import datetime
import os
import random
from unicodedata import bidirectional
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision.models import resnet18

import core.config as conf
from torchvision.transforms import transforms

from fn_networks import init_weights, show_feature_map, BackboneAud

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange


class backbone(nn.Module):

    def __init__(self,
            mode = 'image',
            freeze = False,
            out_tensor = True,
    ):
        super().__init__()
        self.freeze = freeze
        self.out_tensor = out_tensor

        if mode == 'image':
            self.net = nn.Sequential(*list(resnet18(pretrained=True).children())[:-3])
            for layer in self.net:
                    if isinstance(layer, nn.Conv2d):
                        layer.requires_grad_ = not freeze

        else:  
            self.net = nn.Sequential(*list(resnet18(pretrained=True).children())[:-4])
            self.multi_mic = conf.logmelspectro['multi_mic']
            aud_c = 16 if self.multi_mic else 1
            self.net[0] = nn.Conv2d(aud_c, self.net[0].out_channels, kernel_size=3).apply(init_weights)

    def forward(self, x):

        x = self.net(x)

        return x


class embedder_img(nn.Module):

    def __init__(self,
        sz = 64,
        out_tensor = True,
        get_max = True
    ):
        super().__init__()

        self.out_tensor = out_tensor
        self.get_max = get_max

        if self.out_tensor:
            self.net = nn.Sequential(*list(resnet18(pretrained=True).children())[-3:-2], 
                                    nn.Conv2d(512, sz, kernel_size=1).apply(init_weights))
            self.net[0][0].downsample[0].stride = (1,1)        # making sure its a  14x14 feature map
            self.net[0][0].conv1.stride = (1,1)
        else:
            pool2d = nn.AdaptiveMaxPool2d(output_size=(1,1)) if self.get_max else nn.AdaptiveAvgPool2d(output_size=(1,1)) 
            self.net = nn.Sequential(*list(resnet18(pretrained=True).children())[-3:-2], pool2d)
            self.FC1 = nn.Linear(512, sz).apply(init_weights)
            self.FC2 = nn.Linear(sz,sz).apply(init_weights)


            pass

    def forward(self, x):
       
        x = self.net(x)
        if not self.out_tensor:
            x = self.FC1(x.squeeze(-1).squeeze(-1))
            x = self.FC2(x).unsqueeze(-1).unsqueeze(-1)
           
        return x


class embedder_aud(nn.Module):

    def __init__(self,
        sz = 64,
        sz_cam = 11,
        get_max = False
    ):
        super().__init__()
        self.get_max = get_max

        self.net = nn.Sequential(*list(resnet18(pretrained=True).children())[-4:-1])
        if self.get_max:
            self.net[2] = nn.AdaptiveMaxPool2d(output_size=(1,1))
        self.FC1 = nn.Linear(512, sz).apply(init_weights)
        self.FC2 = nn.Linear(sz + sz_cam, sz)

    def forward(self, x, cam=None):

        x = self.net(x)
        x = self.FC1(x.squeeze(-1).squeeze(-1))
        x = torch.concat((x, cam.squeeze(1)), dim=-1).float()
        x = self.FC2(x).unsqueeze(-1).unsqueeze(-1)

        return x


class embedder_AV(nn.Module):

    def __init__(self,
            sz_FC = 64,
            sz_cam = 11,
            out_tensor = True,
            max_a = False,
            max_v = False,
            img_only = False,

    ):
        super().__init__()

        self.sz_FC, self.sz_cam = sz_FC, sz_cam
        self.out_tensor = out_tensor
        self.img_only = img_only

        self.img_backbone = backbone(mode='image')
        self.img_embedder = embedder_img(sz=sz_FC, out_tensor=self.out_tensor, get_max=max_v)

        if not img_only:
            self.aud_backbone = backbone(mode='audio')
            self.aud_embedder = embedder_aud(sz=sz_FC, sz_cam=sz_cam, get_max=max_a)

    def forward(self, img, aud=None, cam=None, device=conf.training_param['device']):

        img_stem = self.img_backbone(img)
        img_rep = self.img_embedder(img_stem)

        if not self.img_only:

            if cam is None:
                cam = torch.zeros((img.shape[0], 1, self.sz_cam)).to(device=device)
            aud_stem = self.aud_backbone(aud)
            aud_rep = self.aud_embedder(aud_stem, cam)

            return img_rep, aud_rep
        else:

            return img_rep



class AVE(nn.Module):

    def __init__(self,
            sz_FC = 64,
            sz_cam = 11,
            set_train = not conf.training_param['inference'],
            multi_frame = conf.training_param['frame_seq']
    ):
        super().__init__()
        self.sz_FC, self.sz_cam = sz_FC, sz_cam
        self.set_train = set_train
        self.multi_frame = multi_frame

        self.AV_embedder = embedder_AV(sz_FC=sz_FC, sz_cam=sz_cam, out_tensor=False, max_a=True, max_v=True)

        if self.multi_frame:
            self.gru = nn.GRU(sz_FC, sz_FC, 1, batch_first=True, bidirectional=False)
            self.img_embedder = embedder_AV(sz_FC=sz_FC, sz_cam=sz_cam, out_tensor=False, max_a=True, max_v=True, img_only=True)

        self.FC_final = nn.Linear(1,2)

    def forward(self, img, aud, cam=None, device=conf.training_param['device']):

        bs = img.shape[0]

        if self.multi_frame:

            for frame_idx in range(img.shape[1]):
                if frame_idx == 0:
                    img_seq, aud_rep = self.AV_embedder(img[:, frame_idx], aud, cam)
                    img_seq = img_seq.squeeze().unsqueeze(1)
                else:
                    im = self.img_embedder(img[:, frame_idx]).squeeze().unsqueeze(1)
                    img_seq = torch.concat((img_seq, im), dim=1)
                    
            img_rep = self.gru(img_seq)[0][:,-1,:]

        else:
            img_rep, aud_rep = self.AV_embedder(img, aud, cam)

        img_rep = img_rep.squeeze()
        aud_rep = aud_rep.squeeze()

        if self.set_train:

            x_score = torch.zeros((bs, bs)).to(device=device)
            for i, im in enumerate(img_rep):

                im = im / torch.linalg.vector_norm(im, dim=-1)

                for j, au in enumerate(aud_rep):
                    
                    au = au / torch.linalg.vector_norm(au, dim=-1)

                    x_score[i,j] = (im - au).pow(2).sum(-1).sqrt()

            x_score = rearrange(x_score, 'b1 b2 -> (b1 b2)').unsqueeze(-1)
            x_score = self.FC_final(x_score)
            x_score = rearrange(x_score, '(b1 b2) c -> b1 b2 c', b1=bs)

            return x_score

        else:

            im = img_rep / torch.linalg.vector_norm(img_rep, dim=-1, keepdim=True)
            au = aud_rep / torch.linalg.vector_norm(aud_rep, dim=-1, keepdim=True)

            x = (im - au).pow(2).sum(-1).sqrt().unsqueeze(-1)

            x_out = self.FC_final(x)

            return x_out


class AVOL(nn.Module):

    def __init__(self,
            sz_FC = 32,
            sz_cam = 11,
            set_train = True
    ):
        super().__init__()
        self.sz_FC, self.sz_cam = sz_FC, sz_cam
        self.set_train = set_train

        self.AV_embedder = embedder_AV(sz_FC=sz_FC, sz_cam=sz_cam)

        self.conv_final = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, img, aud, cam=None, device=conf.training_param['device']):

        bs = img.shape[0]

        img_rep, aud_rep = self.AV_embedder(img, aud, cam)

        if self.set_train:
            x_score = torch.zeros((bs, bs)).to(device=device)
            for i, im in enumerate(img_rep):
                for j, au in enumerate(aud_rep):
                    x_map = (im * au).mean(dim=0, keepdim=True) 
                    x_score[i,j] = torch.amax(torch.sigmoid(self.conv_final(x_map)).squeeze(0))

            return x_score

        else:
            x_map = img_rep * aud_rep
            x_map = x_map.mean(dim=1, keepdim=True) 

            x_map = torch.sigmoid(self.conv_final(x_map)).squeeze(1)
            x_score = torch.amax(x_map, (1,2))

            return x_score, x_map


class EZ_VSL(nn.Module):

    def __init__(self,
            sz_FC = 32,
            sz_cam = 11,
            set_train = True,
    ):
        super().__init__()
        self.sz_FC, self.sz_cam, self.set_train = sz_FC, sz_cam, set_train

        self.AV_embedder = embedder_AV(sz_FC=sz_FC, sz_cam=sz_cam)

        cos_dim = 0 if self.set_train else 1
        self.cos = nn.CosineSimilarity(dim=cos_dim)

    def forward(self, img, aud, cam=None, device=conf.training_param['device']):

        bs = img.shape[0]

        img_rep, aud_rep = self.AV_embedder(img, aud, cam)

        if self.set_train:
            x_score = torch.zeros((bs, bs)).to(device=device)
            for i, im in enumerate(img_rep):
                for j, au in enumerate(aud_rep):
                    x_map = self.cos(im, au)
                    x_score[i,j] = torch.amax(x_map)

            return x_score

        else:
            x_map = self.cos(img_rep, aud_rep)
            x_score = torch.amax(x_map, (1,2))

            return x_score, x_map



