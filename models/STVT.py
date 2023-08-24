from models.networks import *
from torch import nn
from models.DPTV_model import *


class S_MAE(nn.Module):
    def __init__(self, opt):
        super(S_MAE, self).__init__()


        if opt.backbone_name =='EfficientNet':
            self.Feature_extractor = EfficientNet().eval()
            self.Roncon_model = DPTV(in_chans=272,patch_size=2, divide_num=4)

        if opt.backbone_name =='IMAGE':
            self.Feature_extractor = IMAGE().eval()
            self.Roncon_model = DPTV(in_chans=3, img_size=256, patch_size=16)

        if opt.backbone_name =='D_VGG':
            self.Feature_extractor = D_VGG().eval()
            self.Roncon_model = DPTV(in_chans=768)

        if opt.backbone_name =='VGG':
            self.Feature_extractor = VGG().eval()
            self.Roncon_model = DPTV(in_chans=960)

        if opt.backbone_name == 'Resnet34':
            self.Feature_extractor = Resnet34().eval()
            self.Roncon_model = DPTV(in_chans=512)

        if opt.backbone_name == 'Resnet50':
            self.Feature_extractor = Resnet50().eval()
            self.Roncon_model = DPTV(in_chans=1856)

        if opt.backbone_name =='WResnet50':
            self.Feature_extractor = WResnet50().eval()
            self.Roncon_model = DPTV(in_chans=1856)
        
        if opt.backbone_name =='Resnet101':
            self.Feature_extractor = Resnet101().eval()
            self.Roncon_model = DPTV(in_chans=1856)

        if opt.backbone_name == 'MobileNet':
            self.Feature_extractor = MobileNet().eval()
            self.Roncon_model = DPTV(in_chans=104)
    
    def forward(self, imgs):
        deep_feature = self.Feature_extractor(imgs)
        loss, pre_feature, _ = self.Roncon_model(deep_feature)
        return deep_feature, pre_feature, loss



    def a_map(self, deep_feature, pre_feature):
        recon_feature = self.Roncon_model.unpatchify(pre_feature)
        batch_size = recon_feature.shape[0]
        dis_map =torch.mean((deep_feature - recon_feature) ** 2, dim=1)
        dis_map = dis_map.reshape(batch_size, 1, 32, 32)
        dis_map = nn.functional.interpolate(dis_map, size=(256, 256), mode="bilinear", align_corners=True).squeeze(1)
        dis_map = dis_map.clone().squeeze(0).cpu().detach().numpy()



        dir_map = 1 - torch.nn.CosineSimilarity()(deep_feature, recon_feature)
        dir_map = dir_map.reshape(batch_size, 1, 32, 32)
        dir_map = nn.functional.interpolate(dir_map, size=(256, 256), mode="bilinear", align_corners=True).squeeze(1)
        dir_map = dir_map.clone().squeeze(0).cpu().detach().numpy()
        return dis_map, dir_map
    
    
    def attention_matrix(self, img):
        deep_feature = self.Feature_extractor(img)
        all_attn = self.Roncon_model.attn(deep_feature)
        return all_attn




class AE(nn.Module):
    def __init__(self, opt):
        super(AE, self).__init__()

        if opt.backbone_name == 'EfficientNet':
            self.Feature_extractor = EfficientNet().eval()
            self.Roncon_model = TR(in_chans=272)

        if opt.backbone_name == 'IMAGE':
            self.Feature_extractor = IMAGE().eval()
            self.Roncon_model = TR(in_chans=3)

        if opt.backbone_name == 'D_VGG':
            self.Feature_extractor = D_VGG().eval()
            self.Roncon_model = TR(in_chans=768)

        if opt.backbone_name == 'VGG':
            self.Feature_extractor = VGG().eval()
            self.Roncon_model = TR(in_chans=960)

        if opt.backbone_name == 'Resnet34':
            self.Feature_extractor = Resnet34().eval()
            self.Roncon_model = TR(in_chans=512)

        if opt.backbone_name == 'Resnet50':
            self.Feature_extractor = Resnet50().eval()
            self.Roncon_model = TR(in_chans=1856)

        if opt.backbone_name == 'Resnet101':
            self.Feature_extractor = Resnet101().eval()
            self.Roncon_model = TR(in_chans=1856)

        if opt.backbone_name == 'MobileNet':
            self.Feature_extractor = MobileNet().eval()
            self.Roncon_model = TR(in_chans=104)

    def forward(self, imgs):
        deep_feature = self.Feature_extractor(imgs)
        loss, pre_feature, _ = self.Roncon_model(deep_feature)
        return deep_feature, pre_feature, loss

    def a_map(self, deep_feature, pre_feature):
        recon_feature = self.Roncon_model.unpatchify(pre_feature)
        batch_size = recon_feature.shape[0]
        dis_map = torch.mean((deep_feature - recon_feature) ** 2, dim=1)
        dis_map = dis_map.reshape(batch_size, 1, 32, 32)
        dis_map = nn.functional.interpolate(dis_map, size=(256, 256), mode="bilinear", align_corners=True).squeeze(1)
        dis_map = dis_map.clone().squeeze(0).cpu().detach().numpy()

        dir_map = 1 - torch.nn.CosineSimilarity()(deep_feature, recon_feature)
        dir_map = dir_map.reshape(batch_size, 1, 32, 32)
        dir_map = nn.functional.interpolate(dir_map, size=(256, 256), mode="bilinear", align_corners=True).squeeze(1)
        dir_map = dir_map.clone().squeeze(0).cpu().detach().numpy()
        return dis_map, dir_map

    def attention_matrix(self, img):
        deep_feature = self.Feature_extractor(img)
        all_attn = self.Roncon_model.attn(deep_feature)
        return all_attn




