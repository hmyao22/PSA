# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from models.utils import get_2d_sincos_pos_embed


class DPTV(nn.Module):

    def __init__(self, img_size=32, patch_size=2, in_chans=272,
                 embed_dim=240, divide_num=4, depth=12, num_heads=12,
                 decoder_embed_dim=240, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.divide_num = divide_num
        self.depth = depth

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.auxiliary_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)


        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)

        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * self.in_chans))
        return x


    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs

    def latent_unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = 1
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p,  self.embed_dim))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0],  self.embed_dim, h * p, h * p))
        return imgs


    def forward_encoder(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        ########################
        N, L, D = x.shape  # batch, length, dim
        assert L % self.divide_num == 0
        mask_ratio = 1 / self.divide_num
        len_keep = int(L * mask_ratio)
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        output = torch.zeros_like(x)
        ########################

        for i in range(self.divide_num):
            ids_keep = torch.cat([ids_shuffle[:, 0: i * len_keep], ids_shuffle[:, (i + 1) * len_keep:]], dim=-1)
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
            mask = torch.ones([N, L], device=x.device)
            mask[:, i * len_keep:(i + 1) * len_keep] = 0
            mask_bool_retain = mask > 0
            mask_bool_pad = mask < 0.5
            auxiliary_token = self.auxiliary_token.repeat(x_masked.shape[0], ids_restore.shape[1] - x_masked.shape[1], 1)
            x_ = torch.cat([x_masked, auxiliary_token], dim=1)
            x_ = x_.masked_scatter_(mask_bool_retain.unsqueeze(-1).repeat(1, 1, x_.shape[2]), x_masked)
            x_ = x_.masked_scatter_(mask_bool_pad.unsqueeze(-1).repeat(1, 1, x_.shape[2]), auxiliary_token)
            x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))


            ###########
            for blk in self.blocks:
                x_masked = blk(x_masked)

            x_masked = self.norm(x_masked)

            mask = torch.gather(mask, dim=1, index=ids_restore)
            mask_pad = mask > 0.5
            mask_pad = mask_pad.unsqueeze(-1).repeat(1, 1, x_masked.shape[2])
            x_masked = x_masked.masked_fill(mask_pad, 0)

            output = output + x_masked



        return output

    def forward_decoder(self, x):
        # embed tokens
        x = self.decoder_embed(x)

        # add pos embed
        x = x + self.decoder_pos_embed[:, 1:, :]

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token

        return x

    def attn(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        ########################
        N, L, D = x.shape  # batch, length, dim
        assert L % self.divide_num == 0
        mask_ratio = 1 / self.divide_num
        len_keep = int(L * mask_ratio)
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        output = torch.zeros_like(x)
        ########################
        all_attn = 0
        for i in range(self.divide_num):
            ids_keep = torch.cat([ids_shuffle[:, 0: i * len_keep], ids_shuffle[:, (i + 1) * len_keep:]], dim=-1)
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
            mask = torch.ones([N, L], device=x.device)
            mask[:, i * len_keep:(i + 1) * len_keep] = 0
            mask_bool_retain = mask > 0
            mask_bool_pad = mask < 0.5
            auxiliary_token = self.auxiliary_token.repeat(x_masked.shape[0], ids_restore.shape[1] - x_masked.shape[1],
                                                          1)
            x_ = torch.cat([x_masked, auxiliary_token], dim=1)
            x_ = x_.masked_scatter_(mask_bool_retain.unsqueeze(-1).repeat(1, 1, x_.shape[2]), x_masked)
            x_ = x_.masked_scatter_(mask_bool_pad.unsqueeze(-1).repeat(1, 1, x_.shape[2]), auxiliary_token)
            x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))


            ###########
            for j, blk in enumerate(self.blocks):

                attn = blk.attention(x_masked)
                # attn[:, :, ids_keep[0], :] =0
                all_attn += attn
                x_masked = blk(x_masked)


            output = output + x_masked

        return all_attn /4



    def forward_loss(self, imgs, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        dis_loss = (pred - target) ** 2
        dis_loss = dis_loss.mean(dim=-1)  # [N, L], mean loss per patch
        dir_loss = 1 - torch.nn.CosineSimilarity(-1)(pred, target)

        loss = 5 * dir_loss.mean() + dis_loss.mean()  # mean loss on removed patches
        return loss


    def forward(self, imgs):
        latent = self.forward_encoder(imgs)

        pred = self.forward_decoder(latent)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred)


        return loss, pred, latent



class TR(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=32, patch_size=2, in_chans=272,
                 embed_dim=240, depth=12, num_heads=12,
                 decoder_embed_dim=240, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        self.in_chans = in_chans
        self.depth = depth

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # w = self.patch_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs


    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)


        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_decoder(self, x):
        # embed tokens
        x = self.decoder_embed(x)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def attn(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]


        attn = 0
        # apply Transformer blocks
        for i, blk in enumerate(self.blocks):

            attn += blk.attention(x)
            x = blk(x)

        return attn

    def forward_loss(self, imgs, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        dis_loss = (pred - target) ** 2
        dis_loss = dis_loss.mean(dim=-1)  # [N, L], mean loss per patch
        dir_loss = 1 - torch.nn.CosineSimilarity(-1)(pred, target)

        loss = 5 * dir_loss.mean() + dis_loss.mean()  # mean loss on removed patches
        return loss

    def forward(self, imgs):
        latent = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent)
        loss = self.forward_loss(imgs, pred)
        return loss, pred,latent


if __name__ =='__main__':
    import time

    model = TR().cuda()
    input_tensor = torch.rand(1, 272, 32, 32).cuda()
    for i in range(10):
        output = model.forward_encoder(input_tensor)
        t1 = time.time()
        output = model.forward_decoder(output)
        t2= time.time()
        print(t2-t1)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

# from functools import partial
# import torch
# import torch.nn as nn
# from timm.models.vision_transformer import PatchEmbed, Block
# from models.utils import get_2d_sincos_pos_embed
#
#
# class DPTV(nn.Module):
#
#     def __init__(self, img_size=32, patch_size=2, in_chans=272,
#                  embed_dim=240, divide_num=2, depth=6, num_heads=12,
#                  decoder_embed_dim=240, decoder_depth=6, decoder_num_heads=16,
#                  mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.in_chans = in_chans
#         self.divide_num = divide_num
#         self.decoder_embed_dim =decoder_embed_dim
#
#         # --------------------------------------------------------------------------
#         # MAE encoder specifics
#         self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
#         num_patches = self.patch_embed.num_patches
#
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
#                                       requires_grad=False)  # fixed sin-cos embedding
#
#         self.blocks = nn.ModuleList([
#             Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
#             for i in range(depth)])
#         self.norm = norm_layer(embed_dim)
#
#         self.auxiliary_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.latent_auxiliary_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#
#
#         # --------------------------------------------------------------------------
#
#         # --------------------------------------------------------------------------
#         # MAE decoder specifics
#         self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
#
#         self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
#                                               requires_grad=False)  # fixed sin-cos embedding
#
#         self.decoder_blocks = nn.ModuleList([
#             Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
#             for i in range(decoder_depth)])
#
#         self.decoder_norm = norm_layer(decoder_embed_dim)
#
#         self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)
#
#         self.decoder_pred1 = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)
#         self.decoder_pred2 = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)
#
#
#
#         # --------------------------------------------------------------------------
#
#         self.norm_pix_loss = norm_pix_loss
#
#         self.initialize_weights()
#
#     def initialize_weights(self):
#         # initialization
#         # initialize (and freeze) pos_embed by sin-cos embedding
#         pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
#                                             cls_token=True)
#         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
#
#         decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
#                                                     int(self.patch_embed.num_patches ** .5), cls_token=True)
#         self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
#
#         # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
#
#         w = self.patch_embed.proj.weight.data
#         torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
#
#         # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
#         torch.nn.init.normal_(self.cls_token, std=.02)
#
#         # initialize nn.Linear and nn.LayerNorm
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             # we use xavier_uniform following official JAX ViT:
#             torch.nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     def patchify(self, imgs):
#         """
#         imgs: (N, 3, H, W)
#         x: (N, L, patch_size**2 *3)
#         """
#         p = self.patch_embed.patch_size[0]
#         assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
#
#         h = w = imgs.shape[2] // p
#         x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
#         x = torch.einsum('nchpwq->nhwpqc', x)
#         x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * self.in_chans))
#         return x
#
#
#     def unpatchify(self, x):
#         """
#         x: (N, L, patch_size**2 *3)
#         imgs: (N, 3, H, W)
#         """
#         p = self.patch_embed.patch_size[0]
#         h = w = int(x.shape[1] ** .5)
#         assert h * w == x.shape[1]
#
#         x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
#         x = torch.einsum('nhwpqc->nchpwq', x)
#         imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
#         return imgs
#
#     def latent_unpatchify(self, x):
#         """
#         x: (N, L, patch_size**2 *3)
#         imgs: (N, 3, H, W)
#         """
#         p = 1
#         h = w = int(x.shape[1] ** .5)
#         assert h * w == x.shape[1]
#
#         x = x.reshape(shape=(x.shape[0], h, w, p, p,  self.embed_dim))
#         x = torch.einsum('nhwpqc->nchpwq', x)
#         imgs = x.reshape(shape=(x.shape[0],  self.embed_dim, h * p, h * p))
#         return imgs
#
#
#     def forward_encoder(self, x):
#         x = self.patch_embed(x)
#         x = x + self.pos_embed[:, 1:, :]
#         ########################
#         N, L, D = x.shape  # batch, length, dim
#         assert L % self.divide_num == 0
#         mask_ratio = 1 / self.divide_num
#         len_keep = int(L * mask_ratio)
#         noise = torch.rand(N, L, device=x.device)
#         ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
#         ids_restore = torch.argsort(ids_shuffle, dim=1)
#         output = torch.zeros(N, L, self.decoder_embed_dim).to(x.device)
#
#
#         ########################
#         for i in range(self.divide_num):
#             ids_keep = ids_shuffle[:, i * len_keep:(i + 1) * len_keep]
#             x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
#             mask = torch.ones([N, L], device=x.device)
#             mask[:, 0: i * len_keep] = 0
#             mask[:, (i + 1) * len_keep:] = 0
#             mask_bool_retain = mask > 0
#             mask_bool_pad = mask < 0.5
#             auxiliary_token = self.auxiliary_token.repeat(x_masked.shape[0], ids_restore.shape[1] - x_masked.shape[1], 1)
#             x_ = torch.cat([x_masked, auxiliary_token], dim=1)
#             x_ = x_.masked_scatter_(mask_bool_retain.unsqueeze(-1).repeat(1, 1, x_.shape[2]), x_masked)
#             x_ = x_.masked_scatter_(mask_bool_pad.unsqueeze(-1).repeat(1, 1, x_.shape[2]), auxiliary_token)
#             x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
#
#             ########### encoder #############
#             for encoder_phase, blk in enumerate(self.blocks):
#                 x_masked = blk(x_masked)
#
#             x_masked = self.norm(x_masked)
#
#             mask = torch.gather(mask, dim=1, index=ids_restore)
#             mask_pad = mask > 0.5
#             mask_pad = mask_pad.unsqueeze(-1).repeat(1, 1, x_masked.shape[2])
#
#             mask_out = mask < 0.5
#             mask_out = mask_out.unsqueeze(-1).repeat(1, 1, x_masked.shape[2])
#
#             x_masked = x_masked.masked_fill(mask_pad, 0)
#             ########### decoder #############
#             x_masked = self.decoder_embed(x_masked)
#             latent_auxiliary_token = self.latent_auxiliary_token.repeat(x_masked.shape[0], x_masked.shape[1], 1)
#             x_masked_dec = x_masked.masked_scatter_(mask_pad, latent_auxiliary_token)
#             x_masked_dec = x_masked_dec + self.decoder_pos_embed[:, 1:, :]
#
#             for decoder_phase, blk in enumerate(self.decoder_blocks):
#                 x_masked_dec = blk(x_masked_dec)
#
#             x_masked_dec = self.norm(x_masked_dec)
#
#             x_masked_dec = x_masked_dec.masked_fill(mask_out, 0)
#
#             output = output + x_masked_dec
#
#         output = self.decoder_pred(output)
#
#         return output
#
#
#
#     def forward_loss(self, imgs, pred):
#         """
#         imgs: [N, 3, H, W]
#         pred: [N, L, p*p*3]
#         mask: [N, L], 0 is keep, 1 is remove,
#         """
#         target = self.patchify(imgs)
#         if self.norm_pix_loss:
#             mean = target.mean(dim=-1, keepdim=True)
#             var = target.var(dim=-1, keepdim=True)
#             target = (target - mean) / (var + 1.e-6) ** .5
#
#         dis_loss = (pred - target) ** 2
#         dis_loss = dis_loss.mean(dim=-1)  # [N, L], mean loss per patch
#         dir_loss = 1 - torch.nn.CosineSimilarity(-1)(pred, target)
#
#         loss = 5 * dir_loss.mean() + dis_loss.mean() # mean loss on removed patches
#         return loss
#
#
#     def forward(self, imgs):
#         pred = self.forward_encoder(imgs)
#         loss1 = self.forward_loss(imgs, pred)
#
#         loss = loss1
#
#         return loss, pred, imgs
#


if __name__ =='__main__':
    import time

    model = DPTV().cuda()
    input_tensor = torch.rand(1, 272, 32, 32).cuda()
    for i in range(10):
        t1 = time.time()
        output = model.forward(input_tensor)
        t2= time.time()
        print(t2-t1)


