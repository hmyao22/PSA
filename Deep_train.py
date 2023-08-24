import torch.optim as opti
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import init
import os
from datasets import TrainData, TestData, UniTrainData,UniTestData
import models.networks
from config import DefaultConfig
import torch
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import feature_create_disjoint_masks
import models.STVT as STVT
from scipy.ndimage import gaussian_filter
from datasets import denormalize
from models.misc import NativeScalerWithGradNormCount as NativeScaler
import measure
from models.utils import adjust_learning_rate
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_normal(m.weight)


def train(opt, show_feature_map=True):
    loss_scaler = NativeScaler()
    device = opt.device
    print(device)

    if opt.is_STVT == 'ST':
        model_name = opt.backbone_name + '_' + opt.class_name + '_' + 'ST.pth'
        STVT_model = STVT.S_MAE(opt)
    if opt.is_STVT == 'TR':
        model_name = opt.backbone_name + '_' + opt.class_name + '_' + 'AE.pth'
        STVT_model = STVT.AE(opt)

    STVT_model.Feature_extractor = STVT_model.Feature_extractor.eval()
    STVT_model.Roncon_model = STVT_model.Roncon_model.train(True)

    

    if opt.use_gpu:
        STVT_model.to(device)
    if os.path.exists(opt.load_model_path + model_name):
        STVT_model.Roncon_model.load_state_dict(torch.load(opt.load_model_path + model_name))
        print("load weights!")


    optimizer = opti.AdamW(STVT_model.Roncon_model.parameters(), lr=opt.lr,betas=(0.9, 0.95))

    trainDataset = TrainData(opt=opt)
    train_dataloader = DataLoader(trainDataset, batch_size=opt.train_batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)


    testDataset = TestData(opt=opt)
    test_dataloader = DataLoader(testDataset, batch_size=1, shuffle=True, drop_last=True, num_workers=0)


    max_auc =0.0
    for epoch in range(opt.max_epoch):
        adjust_learning_rate(optimizer, epoch)
        running_loss = 0.0

        for index, item in enumerate(tqdm(train_dataloader, ncols=80)):
            input_frame = item

            if opt.use_gpu:
                input_frame = input_frame.to(device, non_blocking=True)

            deep_feature, recon_feature, loss = STVT_model(input_frame)
            
            
            loss_scaler(loss, optimizer, parameters=STVT_model.Roncon_model.parameters(), update_grad=(index + 1) % 1 == 0)
            running_loss += loss.item()


            if index == len(train_dataloader)-1:
                print(f"[{epoch}]  F_loss: {(running_loss / (1 * len(trainDataset))):.3f}")


        if epoch % 1 == 0:
            if epoch == 0:
                model_dict = STVT_model.Roncon_model.state_dict()
                torch.save(model_dict, opt.load_model_path + model_name)
            STVT_model.eval()
            item = next(iter(test_dataloader))
            input_frame = item

            if opt.use_gpu:
                input_frame = input_frame.to(device, non_blocking=True)

            deep_feature, recon_feature, loss = STVT_model(input_frame)

            dis_amap,dir_amap = STVT_model.a_map(deep_feature,recon_feature)
            dis_amap = gaussian_filter(dis_amap, sigma=4)
            dir_amap = gaussian_filter(dir_amap, sigma=4)

            input_frame = denormalize(input_frame.clone().squeeze(0).cpu().detach().numpy())
            cv2_input = np.array(input_frame, dtype=np.uint8)

            plt.figure()
            plt.subplot(1, 4, 1)
            plt.axis('off')
            plt.imshow(cv2_input)
            plt.subplot(1, 4, 2)
            plt.axis('off')
            plt.imshow(dis_amap, cmap='jet')
            plt.subplot(1, 4, 3)
            plt.axis('off')
            plt.imshow(dir_amap, cmap='jet')
            plt.subplot(1, 4, 4)
            plt.axis('off')
            plt.imshow(dis_amap*dir_amap, cmap='jet')

            plt.savefig("temp.png")
            plt.close()

            if show_feature_map == True:
                plt.figure()
                plt.subplot(131)
                plt.imshow(cv2_input)
                plt.subplot(132)
                plt.imshow(deep_feature.cpu().detach().numpy()[0, 2, :, :])
                plt.subplot(133)
                plt.imshow(STVT_model.Roncon_model.unpatchify(recon_feature).cpu().detach().numpy()[0, 2, :, :])
                plt.savefig('feature/' + str(epoch) + opt.is_STVT +'.png')
                plt.close()
            #
            # if epoch % 10 == 0:
            #     with torch.cuda.device(device):
            #         obj_list=[opt.class_name]
            #         auc = measure.test(opt, total_list_0, mvtec_path=opt.data_root, checkpoint_path=opt.load_model_path, backbone_name=opt.backbone_name, is_STVT=opt.is_STVT, training_model=STVT_model)
            #         if auc > max_auc:
            #             print('weight updata!')
            #             max_auc = auc
            #             model_dict = STVT_model.Roncon_model.state_dict()
            #             torch.save(model_dict, opt.load_model_path + model_name)
            model_dict = STVT_model.Roncon_model.state_dict()
            torch.save(model_dict, opt.load_model_path + model_name)
            STVT_model.Roncon_model = STVT_model.Roncon_model.train(True)




            
total_list_0 = [
    'grid',
    'carpet',
    'leather',
    'tile',
    'wood',
    'bottle',
    'cable',
    'capsule',
    'hazelnut',
    'metal_nut',
    'toothbrush',
    'pill',
    'screw',
    'zipper',
    'transistor',
    ]


cifar = [
        'horse',
        'ship',
        'airplane',
        'truck',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog'
                 ]

MNIST = [
        '0',
        '1',
        '2',
        '3',
        '4',
        '5',
        '6',
        '7',
        '8',
        '9',
        ]

total_list = [
    'facemask',
]


real = ['ubolt_local']

video = ['ped2', 'avenue']
medical = ['OCT', 'brains']
opt = DefaultConfig()


is_STVT = ['ST', 'TR']

backbone_name = ['WResnet50', 'EfficientNet', 'Resnet50', 'MobileNet', 'VGG', 'Resnet34', 'D_VGG', 'IMAGE']

if __name__ == '__main__':
    opt.parse({'is_STVT': is_STVT[0]})
    opt.parse({'backbone_name': backbone_name[1]})
    for obj in real:
        opt.parse({'class_name': obj})
        print('training_dataset:' + str(opt.class_name))
        train(opt)



# opt.parse({'is_STVT': is_STVT[1]})
# opt.parse({'backbone_name': backbone_name[0]})
# for obj in total_list:
#     opt.parse({'class_name': obj})
#     print('training_dataset:' + str(opt.class_name))
#     train(opt)
#
