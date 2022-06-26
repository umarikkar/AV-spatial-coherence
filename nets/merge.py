import datetime
import os
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import core.config as conf
from torchvision.transforms import transforms

from nets.backbones import BackboneAud, BackboneVid

"""
class SubNet_main(nn.Module):

    def __init__(self, mode, multi_mic=conf.logmelspectro['get_gcc']):
        super().__init__()

        if mode=='audio':
            if multi_mic:
                self.conv1_1 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
            else: 
                self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        else:
            self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.BN1 = nn.BatchNorm2d(num_features=64)
        # maxpool --> 112 x 112

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.BN2 = nn.BatchNorm2d(num_features=128)
        # maxpool --> 56 x 56

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BN3 = nn.BatchNorm2d(num_features=256)
        # maxpool --> 28 x 28

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BN4 = nn.BatchNorm2d(num_features=512)
        # maxpool --> 14 x 14

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BN5 = nn.BatchNorm2d(num_features=512)
        # maxpool --> 7 x 7

        

    def forward(self, x):

        x = torch.relu(self.BN1(self.conv1_1(x)))
        x = torch.max_pool2d(x, 2) # 112 x 112

        x = torch.relu(self.BN2(self.conv2_1(x)))
        x = torch.max_pool2d(x, 2) # 56 x 56

        x = torch.relu(self.conv3_1(x))
        x = torch.relu(self.BN3(self.conv3_2(x)))
        x = torch.max_pool2d(x,2) # 28 x 28

        x = torch.relu(self.conv4_1(x)) 
        x = torch.relu(self.BN4(self.conv4_2(x)))
        x = torch.max_pool2d(x,2) # 14 x 14

        x = torch.relu(self.conv5_1(x))
        x = torch.relu(self.BN5(self.conv5_2(x)))

        return x

"""

class SubnetVid(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(512,128,kernel_size=1)
        self.conv2 = nn.Conv2d(128,128,kernel_size=1)
        self.BN = nn.BatchNorm2d(num_features=128)

    def forward(self, x):

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.BN(self.conv2(x)))

        return x
        

class SubnetAud(nn.Module):
    def __init__(self, cam_size=11):
        super().__init__()

        self.conv1 = nn.Conv2d(512,128,kernel_size=1)
        self.FC1 = nn.Linear(128 + cam_size, 128)
        self.BN = nn.BatchNorm2d(num_features=128)

    def forward(self, x, cam_ID=None):

        if cam_ID is None:
            cam_ID = torch.zeros((x.shape[0], 1, 11))
            device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
            cam_ID = cam_ID.to(device=device)

        cam_ID = cam_ID.squeeze(1).float()

        x = torch.relu(self.BN(self.conv1(x)))

        c, h, w = x.shape[1], x.shape[-2], x.shape[-1]
        x = x.view(-1, c, h*w)
        x = torch.mean(x, dim=-1)

        x = torch.concat((x, cam_ID), dim=-1)
        
        x = torch.relu(self.FC1(x))
        
        return x


class MergeNet(nn.Module):

    def __init__(self, multi_mic=conf.logmelspectro['multi_mic'], heatmap=conf.dnn_arch['heatmap']):
        super().__init__()
        self.heatmap = heatmap
        
        self.VideoNet = BackboneVid(custom=False, pretrained=True, freeze=True)
        self.AudioNet = BackboneAud(multi_mic=multi_mic, custom=True)

        self.VideoMerge = SubnetVid()
        self.AudioMerge = SubnetAud()
        
        self.BN = nn.BatchNorm2d(num_features=128)
        self.conv1 = nn.Conv2d(128, 128, kernel_size=1)
        self.dropout = nn.Dropout(p=0.5)
        self.FC_final = nn.Linear(128, 2)

    def forward(self, x, y, cam):
        
        x = self.VideoNet(x)
        y = self.AudioNet(y)

        x = self.VideoMerge(x)
        y = self.AudioMerge(y, cam_ID=cam).unsqueeze(-1).unsqueeze(-1)

        x += x * y

        x = self.conv1(x)

        if self.heatmap:
            x = torch.mean(x, dim=1) # collapsing along dimension

            h, w = x.shape[-2], x.shape[-1]
            x_class = torch.mean(x.view(-1, h*w), dim=-1)
            x_class = torch.sigmoid(x_class)

            return x, x_class

        else:
            c = x.shape[1]
            h, w = x.shape[-2], x.shape[-1]
            x = torch.mean(x.view(-1,c, h*w), dim=-1)
            x = self.dropout(x)
            x = self.FC_final(x)

            return x

