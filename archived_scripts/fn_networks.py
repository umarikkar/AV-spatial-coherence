import datetime
import os
import random
from unicodedata import bidirectional
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

def show_feature_map(x_list, BS_pos, show=False, savefile=True ):

    sz = len(x_list)-1
    BS = x_list[0][0].shape[0]

    for im_idx in range(BS):

        plt.figure(figsize = (4*sz, 4))

        # for idx, (x, title) in enumerate(x_list):
        for idx in range(sz):

            x = x_list[idx][0]
            title = x_list[idx][1]

            plt.subplot(1, sz, idx+1)

            # print(title, x.shape)
            # first, avg pool
            if x.shape[1] != 3:
                if len(x.shape) ==3 :
                    y = x[im_idx].to(device='cpu').detach().numpy()
                else:
                    y = x[im_idx].squeeze(0).to(device='cpu')
                    y = y.abs().mean(dim=0).detach().numpy()
            else:
                # rgb image
                y =  x[im_idx].squeeze(0).permute(1,2,0).to(device='cpu').detach().numpy()

            
            if idx == sz-1:
                plt.imshow(y, aspect='equal', vmax=1, vmin=0)
            else:
                plt.imshow(y, aspect='equal')

            if idx != 0:
                plt.colorbar()

            plt.title(title)

            if im_idx < BS_pos:
                smp = 'positive sample'
            else:
                smp = 'negative sample'

            pred = '     prediction: %s'%round(x_list[-1][im_idx].item(), 2)
            sptitle = smp + pred
            plt.suptitle(sptitle)

        if show:
            plt.show()

        if savefile:
            fig_path = os.path.join(conf.filenames['net_folder_path'], conf.filenames['train_val'])
            num = len(os.listdir(fig_path)) + 1
            plt.savefig(os.path.join(conf.filenames['net_folder_path'], conf.filenames['train_val'], 'img_%s'%num + '.png'))

        plt.close()

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
            # self.net = models_vision.vgg11(pretrained=pretrained).features
            self.net = models_vision.vgg11(pretrained=pretrained).features[0:20]
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
    def __init__(self, BN=True):
        super().__init__()

        self.BN = BN

        self.conv1 = nn.Conv2d(512,128,kernel_size=1)
        self.conv2 = nn.Conv2d(128,128,kernel_size=1)
        if self.BN:
            self.BN_layer = nn.BatchNorm2d(num_features=128)

        self.apply(init_weights)

    def forward(self, x):

        x = torch.relu(self.conv1(x))
        
        if self.BN:
            x = torch.relu(self.BN_layer(self.conv2(x)))
        else:
            x = torch.relu(self.conv2(x))

        return x
        

class SubnetAud(nn.Module):
    def __init__(self, BN=True, cam_size=11):
        super().__init__()

        self.BN = BN

        self.conv1 = nn.Conv2d(512,128,kernel_size=1)
        self.FC1 = nn.Linear(128 + cam_size, 128)

        if self.BN:
            self.BN_layer = nn.BatchNorm2d(num_features=128)

        self.apply(init_weights)

    def forward(self, x, cam_ID=None):

        if cam_ID is None:
            cam_ID = torch.zeros((x.shape[0], 1, 11))
            device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
            cam_ID = cam_ID.to(device=device)

        cam_ID = cam_ID.squeeze(1).float()

        if self.BN:
            x = torch.relu(self.BN_layer(self.conv1(x)))
        else:
            x = torch.relu(self.conv1(x))

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

    def forward(self, x_in, y_in, cam, BS_pos=None):

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
                (x, 'heatmap (after attention)')
            ]           
            show_feature_map(features_ls, BS_pos)

        if self.heatmap:
 
            x = self.conv_final(x)   
            x = torch.mean(x, dim=1) # collapsing along dimension

            h, w = x.shape[-2], x.shape[-1]
            x_class = torch.mean(x.view(-1, h*w), dim=-1)
            x_class = torch.sigmoid(x_class)

            return x_class, x

        else:
            c = x.shape[1]
            h, w = x.shape[-2], x.shape[-1]
            x = torch.mean(x.view(-1,c, h*w), dim=-1)
            # x = self.dropout(x)
            x = self.FC_final(x)

            return x


class AVE_Net(nn.Module):


    def __init__(self, 
    heatmap=conf.dnn_arch['heatmap'], 
    inference = conf.training_param['inference'],
    cam_size=11
    ):
        
        self.inference = inference

        super().__init__()
        self.heatmap = heatmap
        
        self.VideoNet = models_vision.vgg11(pretrained=True).features[0:20]

        for layer in self.VideoNet:
                if isinstance(layer, nn.Conv2d):
                    layer.requires_grad_ = False

        self.AudioNet = BackboneAud()
        self.AudioMerge = SubnetAud(BN=False)

        self.FC1_aud = nn.Linear(512, 128).apply(init_weights)
        self.FC2_aud = nn.Linear(128+cam_size, 128)

        self.FC1_img = nn.Linear(512, 128).apply(init_weights)
        self.FC2_img = nn.Linear(128, 128)
        self.FC3 = nn.Linear(1,2)

    def forward(self, x_in, y_in, cam=None, BS_pos=None):

        x1 = self.VideoNet(x_in)                                                # 512 x 14 x 14
        y1 = self.AudioNet(y_in)                                                # 512 x H x W 

        h_img, w_img = x1.shape[-2], x1.shape[-1]
        h_aud, w_aud = y1.shape[-2], y1.shape[-1]

        x1 = torch.max_pool2d(x1, (h_img, w_img), (h_img, w_img)).squeeze(-1).squeeze(-1)               # 512
        y1 = torch.max_pool2d(y1, (h_aud, w_aud), (h_aud, w_aud)).squeeze(-1).squeeze(-1)               # 512

        x1 = self.FC1_img(x1)                   # 128
        x1 = self.FC2_img(x1)                   # 128

        y1 = self.FC1_aud(y1)                   # 128
        y1 = torch.concat((y1, cam.squeeze(1)), dim=-1).float()    # 139
        y1 = self.FC2_aud(y1)                   # 128

        # L2 norm

        x1_norm = torch.linalg.vector_norm(x1, dim=-1, keepdim=True)
        y1_norm = torch.linalg.vector_norm(y1, dim=-1, keepdim=True)

        x1 = x1 / x1_norm
        y1 = y1 / y1_norm

        x = (x1 - y1).pow(2).sum(-1).sqrt().unsqueeze(-1)

        x = self.FC3(x)

        return x


class AVE_Net2(nn.Module):

    def __init__(self, 
    heatmap=conf.dnn_arch['heatmap'], 
    inference = conf.training_param['inference'],
    cam_size=11
    ):
        
        self.inference = inference

        super().__init__()
        self.heatmap = heatmap
        
        self.VideoNet = models_vision.vgg11(pretrained=True).features[0:20]

        for layer in self.VideoNet:
                if isinstance(layer, nn.Conv2d):
                    layer.requires_grad_ = False

        self.AudioNet = BackboneAud()
        self.AudioMerge = SubnetAud(BN=False)

        self.FC1_aud = nn.Linear(512, 128).apply(init_weights)
        self.FC2_aud = nn.Linear(128+cam_size, 128)

        self.FC1_img = nn.Linear(512, 128)
        self.FC3 = nn.Linear(1,2)

    def forward(self, x_in, y_in, cam=None, BS_pos=None):

        x1 = self.VideoNet(x_in)                                                # 512 x 14 x 14
        y1 = self.AudioNet(y_in)                                                # 512 x H x W 

        h_img, w_img = x1.shape[-2], x1.shape[-1]
        h_aud, w_aud = y1.shape[-2], y1.shape[-1]

        x1 = torch.max_pool2d(x1, (h_img, w_img), (h_img, w_img)).squeeze(-1).squeeze(-1)               # 512
        y1 = torch.max_pool2d(y1, (h_aud, w_aud), (h_aud, w_aud)).squeeze(-1).squeeze(-1)               # 512

        x1 = self.FC1_img(x1)                   # 128

        y1 = self.FC1_aud(y1)                   # 128
        y1 = torch.concat((y1, cam.squeeze(1)), dim=-1).float()    # 139
        y1 = self.FC2_aud(y1)                   # 128

        # L2 norm

        x1_norm = torch.linalg.vector_norm(x1, dim=-1, keepdim=True)
        y1_norm = torch.linalg.vector_norm(y1, dim=-1, keepdim=True)

        x1 = x1 / x1_norm
        y1 = y1 / y1_norm

        x = (x1 - y1).pow(2).sum(-1).sqrt().unsqueeze(-1)

        x = self.FC3(x)

        return x



class AVE_Net_temporal(nn.Module):


    def __init__(self, 
    heatmap=conf.dnn_arch['heatmap'], 
    inference = conf.training_param['inference'],
    cam_size=11
    ):
        
        self.inference = inference

        super().__init__()
        self.heatmap = heatmap
        
        self.VideoNet = models_vision.vgg11(pretrained=True).features[0:20]

        for layer in self.VideoNet:
                if isinstance(layer, nn.Conv2d):
                    layer.requires_grad_ = False

        self.AudioNet = BackboneAud()
        self.AudioMerge = SubnetAud(BN=False)

        self.FC1_aud = nn.Linear(512, 128).apply(init_weights)
        self.FC2_aud = nn.Linear(128+cam_size, 128)

        self.FC1_img = nn.Linear(512, 128)
        # self.FC2_img = nn.Linear(128, 128)
        self.gru = nn.GRU(128, 128, 1, batch_first=True, bidirectional=False)

        self.FC3 = nn.Linear(1,2)

    def forward(self, x_in, y_in, cam=None, BS_pos=None):
        
        for frame_idx in range(x_in.shape[1]):
            x1 = self.VideoNet(x_in[:,frame_idx])    # 512 x 7 x 7 
            h_img, w_img = x1.shape[-2], x1.shape[-1]
            x1 = torch.max_pool2d(x1, (h_img, w_img), (h_img, w_img)).squeeze(-1).squeeze(-1)               # 512
            # x1 = self.FC1_img(x1)                   # 128
            x1 = self.FC1_img(x1).unsqueeze(1)                   # 128
            if frame_idx==0:
                x2 = x1
            else:
                x2 = torch.concat((x2, x1), dim=1)

        x2 = self.gru(x2)[0][:,-1,:]        # taking the last output layer
        

        y1 = self.AudioNet(y_in) 
        h_aud, w_aud = y1.shape[-2], y1.shape[-1]
        y1 = torch.max_pool2d(y1, (h_aud, w_aud), (h_aud, w_aud)).squeeze(-1).squeeze(-1)               # 512
        y1 = self.FC1_aud(y1)                   # 128
        y1 = torch.concat((y1, cam.squeeze(1)), dim=-1).float()    # 139
        y1 = self.FC2_aud(y1)                   # 128


        # L2 norm

        x2_norm = torch.linalg.vector_norm(x2, dim=-1, keepdim=True)
        y1_norm = torch.linalg.vector_norm(y1, dim=-1, keepdim=True)

        x2 = x2 / x2_norm
        y1 = y1 / y1_norm

        x = (x2 - y1).pow(2).sum(-1).sqrt().unsqueeze(-1)

        x = self.FC3(x)

        return x


class AVOL_Net(nn.Module):


    def __init__(self, 
    heatmap=conf.dnn_arch['heatmap'], 
    inference = conf.training_param['inference']
    ):
        
        self.inference = inference

        super().__init__()
        self.heatmap = heatmap
        
        self.VideoNet = models_vision.vgg11(pretrained=True).features[0:20]

        for layer in self.VideoNet:
                if isinstance(layer, nn.Conv2d):
                    layer.requires_grad_ = False

        self.AudioNet = BackboneAud()

        self.VideoMerge = SubnetVid(BN=False)
        self.AudioMerge = SubnetAud(BN=False)

        self.conv_final = nn.Conv2d(1,1,kernel_size=1)

    def forward(self, x_in, y_in, cam, BS_pos=None):

        x1 = self.VideoNet(x_in)                                                # 512 x 7 x 7 
        y1 = self.AudioNet(y_in)                                                # 512 x H x W 

        x2 = self.VideoMerge(x1)                                                # 128 x 7 x 7 
        y2 = self.AudioMerge(y1, cam_ID=cam).unsqueeze(-1).unsqueeze(-1)        # 128 x 1 x 1 

        x2 = x2*y2
        
        x2 = x2.mean(dim=1, keepdim=True)                                       # 1 x 14 x 14 

        x2 = self.conv_final(x2).squeeze(1)

        x_map = torch.sigmoid(x2)                                               # 1 x 14 x 14 

        if self.inference:
            features_ls = [
                (x_in, 'input image'),
                (x1, 'after img backbone'),
                (x2, 'after attn'),
                (x_map, 'heatmap')
            ]           
            show_feature_map(features_ls, BS_pos)

        x_class = torch.amax(x_map, (1,2))

        return x_class, x_map


class AVOL_Net_v2(nn.Module):


    def __init__(self, 
    heatmap=conf.dnn_arch['heatmap'], 
    inference = conf.training_param['inference'],
    cam_size=11
    ):
        
        self.inference = inference

        super().__init__()
        self.heatmap = heatmap
        
        self.VideoNet = models_vision.vgg11(pretrained=True).features[0:20]

        for layer in self.VideoNet:
                if isinstance(layer, nn.Conv2d):
                    layer.requires_grad_ = False

        self.AudioNet = BackboneAud()

        self.conv5 = nn.Conv2d(512, 128, kernel_size=1).apply(init_weights)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=1)

        self.FC1_aud = nn.Linear(512, 128).apply(init_weights)
        self.FC2_aud = nn.Linear(128+cam_size, 128)

        self.conv7 = nn.Conv2d(1,1,kernel_size=1)

    def forward(self, x_in, y_in, cam, BS_pos=None):

        x1 = self.VideoNet(x_in)                                                # 512 x 14 x 14 
        y1 = self.AudioNet(y_in)                                                # 512 x H x W 

        h_aud, w_aud = y1.shape[-2], y1.shape[-1]
        y1 = torch.max_pool2d(y1, (h_aud, w_aud), (h_aud, w_aud)).squeeze(-1).squeeze(-1)

        x1 = torch.relu(self.conv5(x1))                                         # 128 x 14 x 14   
        x1 = torch.relu(self.conv6(x1))                                         # 128 x 14 x 14    

        y1 = self.FC1_aud(y1)                                                   # 128
        y1 = torch.concat((y1, cam.squeeze(1)), dim=-1).float()                 # 139
        y1 = self.FC2_aud(y1).unsqueeze(-1).unsqueeze(-1)                       # 128 x 1  x 1

        x2 = x1*y1                                                              # B x 128 x 14 x 14
        
        x2 = x2.mean(dim=1, keepdim=True)                                       # B x 1 x 14 x 14 
        x2 = self.conv7(x2).squeeze(1)                                     # B x 14 x 14                         

        x_map = torch.sigmoid(x2)                                               # B x 14 x 14 

        x_class = torch.amax(x_map, (1,2))

        if self.inference:
            features_ls = [
                (x_in, 'input image'),
                (x1, 'after img backbone'),
                (x2, 'after attn'),
                (x_map, 'heatmap'),
                x_class
            ]           
            show_feature_map(features_ls, BS_pos)

        return x_class, x_map


class AVOL_Net_temporal(nn.Module):

    def __init__(self, 
    heatmap=conf.dnn_arch['heatmap'], 
    inference = conf.training_param['inference']
    ):
        
        self.inference = inference

        super().__init__()
        self.heatmap = heatmap
        
        self.VideoNet = models_vision.vgg11(pretrained=True).features[0:20]

        for layer in self.VideoNet:
                if isinstance(layer, nn.Conv2d):
                    layer.requires_grad_ = False

        self.AudioNet = BackboneAud()

        self.VideoMerge = SubnetVid(BN=False)
        self.AudioMerge = SubnetAud(BN=False)

        self.conv_final = nn.Conv2d(1,1,kernel_size=1)

    def forward(self, x_in, y_in, cam, BS_pos=None):

        x1 = self.VideoNet(x_in)                                                # 512 x 7 x 7 
        y1 = self.AudioNet(y_in)                                                # 512 x H x W 

        x2 = self.VideoMerge(x1)                                                # 128 x 7 x 7 
        y2 = self.AudioMerge(y1, cam_ID=cam).unsqueeze(-1).unsqueeze(-1)        # 128 x 1 x 1 

        x2 = x2*y2
        
        x2 = x2.mean(dim=1, keepdim=True)                                       # 1 x 14 x 14 

        x2 = self.conv_final(x2).squeeze(1)

        x_map = torch.sigmoid(x2)                                               # 1 x 14 x 14 

        if self.inference:
            features_ls = [
                (x_in, 'input image'),
                (x1, 'after img backbone'),
                (x2, 'after attn'),
                (x_map, 'heatmap')
            ]           
            show_feature_map(features_ls, BS_pos, show=True)

        x_class = torch.amax(x_map, (1,2))

        return x_class, x_map