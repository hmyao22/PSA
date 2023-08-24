import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision.models.vgg import vgg16, vgg19,vgg19_bn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101,wide_resnet50_2, wide_resnet101_2
import torch.nn.functional as F
from torchsummary import summary
from torchvision.models import mobilenet_v2
from models.efficientnet import model


class Doubleconv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = Doubleconv(in_channels=272, out_channels=512)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.layer2 = Doubleconv(in_channels=512, out_channels=256)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.layer3 = Doubleconv(in_channels=256, out_channels=128)
        self.layer4 = Doubleconv(in_channels=128, out_channels=64)

        self.ouput_layer = nn.Conv2d(64, 3, kernel_size=3, padding=1, bias=False)



    def forward(self, input_):
        out1 = self.layer1(input_)
        out1 = self.up1(out1)

        out2 = self.layer2(out1)
        out2 = self.up2(out2)

        out3 = self.layer3(out2)
        out3 = self.up2(out3)
        out4 = self.layer4(out3)

        out5 = self.ouput_layer(out4)


        return out5


