import torch.optim as opti
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import init
import os
from datasets import TrainData, TestData
import models.networks
from config import DefaultConfig
import torch
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import feature_create_disjoint_masks
from scipy.ndimage import gaussian_filter
from datasets import normal
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
from datasets import TrainData, TestData
from torch.utils.data import DataLoader
from datasets import transform_x, denormalize
from torchvision import transforms as T
from PIL import Image
import models.STVT as STVT
import time


from fvcore.nn import FlopCountAnalysis,parameter_count_table

#opt.parse_model_root({'load_model_path': './check_points/VIT_SIZE/ckpt_base/'})
#opt.parse_model_root({'load_model_path': './check_points/DFT/without_DFT/base/'})

def load_image(path):
    transform_x = T.Compose([T.Resize(256, Image.ANTIALIAS),
                             T.ToTensor(),
                             T.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])
    x = Image.open(path).convert('RGB')
    x = transform_x(x)
    x = x.unsqueeze(0)
    return x




image_path = r'D:\IMSN-YHM\dataset\cable\test\poke_insulation\000.png'

backbone_name = ['EfficientNet', 'Resnet34', 'VGG', 'Resnet50', 'Resnet101', 'IMAGE']


def model_test(image_path, backbone_name=backbone_name[0], is_STVT='ST', show_feature_map=True):
    opt = DefaultConfig()
    opt.parse({'backbone_name': backbone_name})
    class_name = image_path.split('\\')[-4]
    label_path = os.path.join(r'D:\IMSN-YHM\dataset', class_name, 'ground_truth',
                              image_path.split('\\')[-2], image_path.split('\\')[-1].split('.')[0]+'_mask.png')
    print(label_path)
    label_image = cv2.imread(label_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    original_tensor = load_image(image_path).to(device)
    if is_STVT == 'ST':
        model_name = opt.backbone_name + '_' + class_name + '_' + 'ST.pth'
        STVT_model = STVT.S_MAE(opt)
    if is_STVT == 'TR':
        model_name = opt.backbone_name + '_' + class_name + '_' + 'TR.pth'
        STVT_model = STVT.AE(opt)
    model_name = 'EfficientNet_divide4_ST.pth'

    if opt.use_gpu:
        STVT_model = STVT_model.to(device)
    STVT_model.Roncon_model.load_state_dict(torch.load(opt.load_model_path + model_name, map_location='cuda:0'))


    deep_feature, recon_feature, loss = STVT_model(original_tensor)

    from thop import profile
    from thop import clever_format
    tensor = original_tensor
    flops, params = profile(STVT_model, inputs=(tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops)
    print(params)

    dis_amap, dir_amap = STVT_model.a_map(deep_feature, recon_feature)

    dis_amap = gaussian_filter(dis_amap, sigma=4)

    input_frame = denormalize(original_tensor.clone().squeeze(0).cpu().detach().numpy())
    cv2_input = np.array(input_frame, dtype=np.uint8)
    grey_image = STVT_model.Roncon_model.unpatchify(recon_feature).cpu().detach().numpy()[0, 2, :, :]
    print((dir_amap*dis_amap).std())


    # retval, dst = cv2.threshold(grey_image, grey_image.max()/10, 255, cv2.THRESH_BINARY)
    # dst = gaussian_filter(dst, sigma=1)



    if show_feature_map == True:
        plt.figure()
        plt.subplot(131)
        plt.imshow(cv2_input)
        plt.subplot(132)
        plt.imshow(deep_feature.cpu().detach().numpy()[0, 2, :, :], cmap='gray')
        plt.subplot(133)
        plt.imshow(grey_image, cmap='gray')
        plt.show()

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.imshow(cv2_input)
    plt.title('input image')
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.imshow(dir_amap*dis_amap, cmap='jet')
    plt.title('detection result')
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.imshow(label_image)
    plt.title('label')
    plt.show()


model_test(image_path, is_STVT='ST')