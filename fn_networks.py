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

# WEIGHT INITIALISER FUNCTION ----------------------------------------------------------------------------

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)


# SHOW FEATURE MAPS

def show_feature_map(x_list, show=False, savefile=True):
    sz = len(x_list)
    plt.figure(figsize = (4*sz, 4))
    for idx, (x, title) in enumerate(x_list):
        plt.subplot(1, sz, idx+1)
        # first, avg pool
        if x.shape[1] != 3:
            y = x.squeeze(0).to(device='cpu')
            y = y.mean(dim=0).detach().numpy()
        else:
            # rgb image
            y =  x.squeeze(0).permute(1,2,0).to(device='cpu').detach().numpy()

        plt.imshow(y, aspect='equal')
        plt.title(title)
    if show:
        plt.show()

    if savefile:
        fig_path = os.path.join(conf.filenames['net_folder_path'], conf.filenames['train_val'])
        num = len(os.listdir(fig_path)) + 1
        plt.savefig(os.path.join(conf.filenames['net_folder_path'], conf.filenames['train_val'], 'img_%s'%num + '.png'))

    return

# BACKBONE FUNCTIONS ------------------------------------------------------------------------------------

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

        self.apply(init_weights)


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

    def __init__(self, 
    custom=conf.dnn_arch['vid_custom'], 
    pretrained=conf.dnn_arch['vid_pretrained'], 
    freeze=conf.dnn_arch['vid_freeze']
    ):
        super().__init__()

        if not custom:
            self.net = models_vision.vgg11(pretrained=pretrained).features
            print('Pretrained VGG11 for Imgs')
            for layer in self.net:
                if isinstance(layer, nn.Conv2d):
                    layer.requires_grad_ = not freeze

        else:
            print('Custom for Video')
            self.net = CustomConv(c_in=3)

    def forward(self, x):

        x = self.net(x)  # returns a (* , 512 , h , w) tensor.
            
        return x



class BackboneAud(nn.Module):

    def __init__(self,  
    multi_mic=conf.logmelspectro['multi_mic'], 
    custom=conf.dnn_arch['aud_custom'], 
    pretrained=conf.dnn_arch['aud_pretrained'], 
    freeze=conf.dnn_arch['aud_freeze']
    ):
        super().__init__()

        if not custom:
            self.net = None if pretrained else None
            self.net = None if freeze else None
            print('Pretrained for Audio')
        else:
            self.net = CustomConv(c_in=16) if multi_mic else CustomConv(c_in=1)
            print('Custom for Audio')

    def forward(self, x):

        x = self.net(x)  # returns a (* , 512 , h , w) tensor.
            
        return x




# MERGING FUNCTIONS --------------------------------------------------------------------------------------------------------------------------

class SubnetVid(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(512,128,kernel_size=1)
        self.conv2 = nn.Conv2d(128,128,kernel_size=1)
        self.BN = nn.BatchNorm2d(num_features=128)

        self.apply(init_weights)

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

        self.apply(init_weights)

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

    def __init__(self, 
    heatmap=conf.dnn_arch['heatmap'], 
    inference = conf.training_param['inference']
    ):
        
        self.inference = inference

        super().__init__()
        self.heatmap = heatmap
        
        self.VideoNet = BackboneVid()
        self.AudioNet = BackboneAud()

        self.VideoMerge = SubnetVid()
        self.AudioMerge = SubnetAud()
        
        self.BN = nn.BatchNorm2d(num_features=128)
        self.conv1 = nn.Conv2d(128, 64, kernel_size=1)
        # self.dropout = nn.Dropout(p=0.5)

        self.conv_final = nn.Conv2d(64, 1, kernel_size=1)
        self.FC_final = nn.Linear(64, 2)

    def forward(self, x_in, y_in, cam):

        x1 = self.VideoNet(x_in)                                                # 512 x 7 x 7 
        y1 = self.AudioNet(y_in)                                                # 512 x H x W 

        x2 = self.VideoMerge(x1)                                              # 128 x 7 x 7 
        y2 = self.AudioMerge(y1, cam_ID=cam).unsqueeze(-1).unsqueeze(-1)      # 128 x 1 x 1 

        x2 = x2*y2                                                             # 128 x 7 x 7 

        x = self.conv1(x2)                                                   # 64 x 7 x 7 

        if self.inference:
            features_ls = [
                (x_in, 'input image'),
                (x1, 'after img backbone'),
                (x2, 'after attention'),
                (x, 'heatmap')
            ]           
            show_feature_map(features_ls)

        if self.heatmap:
 
            x = self.conv_final(x)   
            x = torch.mean(x, dim=1) # collapsing along dimension

            h, w = x.shape[-2], x.shape[-1]
            x_class = torch.mean(x.view(-1, h*w), dim=-1)
            x_class = torch.sigmoid(x_class)

            return x, x_class

        else:
            c = x.shape[1]
            h, w = x.shape[-2], x.shape[-1]
            x = torch.mean(x.view(-1,c, h*w), dim=-1)
            # x = self.dropout(x)
            x = self.FC_final(x)

            return x