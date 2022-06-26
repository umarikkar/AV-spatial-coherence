import datetime
import os
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import torchvision.models as models_vision
import torchaudio.models as models_audio

import core.config as conf
from torchvision.transforms import transforms


class CustomConv(nn.Module):

    def __init__(self, c_in=1):
        super().__init__()

        self.conv1_1 = nn.Conv2d(c_in, 64, kernel_size=3, padding=1)
        self.BN1 = nn.BatchNorm2d(num_features=64)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.BN2 = nn.BatchNorm2d(num_features=128)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BN3 = nn.BatchNorm2d(num_features=256)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BN4 = nn.BatchNorm2d(num_features=512)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BN5 = nn.BatchNorm2d(num_features=512)

    def forward(self, x):

        x = torch.relu(self.BN1(self.conv1_1(x)))
        x = torch.max_pool2d(x, 2) 

        x = torch.relu(self.BN2(self.conv2_1(x)))
        x = torch.max_pool2d(x, 2) 

        x = torch.relu(self.conv3_1(x))
        x = torch.relu(self.BN3(self.conv3_2(x)))
        x = torch.max_pool2d(x,2) 

        x = torch.relu(self.conv4_1(x)) 
        x = torch.relu(self.BN4(self.conv4_2(x)))
        x = torch.max_pool2d(x,2) 

        x = torch.relu(self.conv5_1(x))
        x = torch.relu(self.BN5(self.conv5_2(x)))

        return x


class BackboneVid(nn.Module):

    def __init__(self,  custom=False, pretrained=False, freeze=False):
        super().__init__()

        if not custom:
            self.net = models_vision.vgg11(pretrained=pretrained).features

            for layer in self.net:
                if isinstance(layer, nn.Conv2d):
                    layer.requires_grad_ = not freeze

        else:
            self.net = CustomConv(c_in=3)

    def forward(self, x):

        x = self.net(x)  # returns a (* , 512 , h , w) tensor.
            
        return x



class BackboneAud(nn.Module):

    def __init__(self,  multi_mic=conf.logmelspectro['multi_mic'], custom=True, pretrained=False, freeze=False):
        super().__init__()

        if not custom:
            self.net = None if pretrained else None
        else:
            self.net = CustomConv(c_in=16) if multi_mic else CustomConv(c_in=1)

    def forward(self, x):

        x = self.net(x)  # returns a (* , 512 , h , w) tensor.
            
        return x