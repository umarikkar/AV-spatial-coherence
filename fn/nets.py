from cgitb import small
from re import I
from turtle import hideturtle
from unicodedata import bidirectional
from sklearn.datasets import load_files
import torch
import torch.nn as nn
from einops import rearrange
from torchvision.models import resnet18
from fn.losses import loss_AVE, loss_AVOL, loss_VSL, loss_embed
import torch.nn.functional as F
import os
import core.config as conf

import matplotlib.pyplot as plt

from fn.sampling import gcc_collapse

# SET AND LOAD NETWORK  ------------------------------------------------------------------------------------------


def set_network(set_train=True, net_name=conf.dnn_arch['net_name'], print_net=True):
    
    if net_name=='AVE':
        net = AVE(set_train=set_train)
        loss_fn = loss_AVE() 

    elif net_name=='AVOL':
        net = MergeNet(merge_type='AVOL', set_train=set_train)
        loss_fn = loss_AVOL()
    elif net_name=='VSL':
        net = MergeNet(merge_type='VSL', set_train=set_train)
        loss_fn = loss_VSL()
    elif net_name=='SL2':
        net = MergeNet(merge_type='SL2', set_train=set_train)
        loss_fn = loss_AVE()
    elif net_name=='SL2_NCE':
        net = MergeNet(merge_type='SL2', loss_type='NCE', set_train=set_train)
        loss_fn = loss_VSL()

    elif net_name=='SpaceNet':
        net = SpaceNet(merge_type='VSL', set_train=set_train)
        loss_fn = loss_VSL()
    elif net_name=='SpaceNet_split':
        net = SpaceNet(merge_type='VSL', set_train=set_train, split_dims=True)
        loss_fn = loss_VSL()
    elif net_name=='SpaceNet_V':
        net = SpaceNet(merge_type='VSL', set_train=set_train, split_dims=True, vertical_only=True)
        loss_fn = loss_VSL()

    elif net_name=='FactorNet':
        net = FactorNet(merge_type='VSL', set_train=set_train)
        loss_fn = loss_VSL()
    elif net_name=='FAV_sim':   
        net = FactorNet(merge_type='VSL', set_train=set_train, vertical_only=True, simple_aud=True)
        loss_fn = loss_VSL()
    elif net_name=='FactorNet_V':
        net = FactorNet(merge_type='VSL', set_train=set_train, vertical_only=True)
        loss_fn = loss_VSL()

    elif net_name=='MixNet':
        net = MixNet(merge_type='AVOL', set_train=set_train)
        loss_fn = loss_embed()

    if print_net:
        print('net: ', type(net), '\nloss: ', loss_fn)

    return net, loss_fn

def load_network(net_name=conf.dnn_arch['net_name'], ep=16, set_train=True):

    net, _ = set_network(set_train=set_train, net_name=net_name, print_net=False)

    fol_name = conf.set_filename(name=net_name, print_path=False, hard_negatives=False)

    pt_name = 'net_ep_%s.pt'%ep
    net_path = os.path.join(fol_name, pt_name)

    checkpoint = torch.load(net_path)

    net.load_state_dict(checkpoint['model'])

    return net

# WEIGHT INITIALISER FUNCTION ----------------------------------------------------------------------------

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)


# NETWORK BACKBONES (IMAGE AND AUDIO) --------------------------------------------------------------------

class backbone(nn.Module):

    def __init__(self,
            mode = 'image',
            freeze = False,
            AVE_backbone=False,
            small_features = True,
            multi_mic = conf.logmelspectro['multi_mic']

    ):
        super().__init__()
        self.freeze = freeze
        self.AVE_backbone = AVE_backbone
        self.small_features = small_features
        self.multi_mic = multi_mic

        if mode == 'image':

            if not self.AVE_backbone:
                self.net = nn.Sequential(*list(resnet18(pretrained=True).children())[:-3])
                for layer in self.net:
                        if isinstance(layer, nn.Conv2d):
                            layer.requires_grad_ = not freeze

            # if AVE backbone, use the pretrained weights from the ave backbone
            else:
                print('using pretrained AVE backbone weights for image')
                self.net = load_network(net_name='AVE', ep=60, set_train=True).AV_embedder.img_backbone

            if not self.small_features:
                self.net[6][0].conv1.stride = (1,1)
                self.net[6][0].downsample[0].stride = (1,1)


        else:
            if not self.AVE_backbone:
                self.net = nn.Sequential(*list(resnet18(pretrained=True).children())[:-4])
                aud_c = 16 if self.multi_mic else 1
                self.net[0] = nn.Conv2d(aud_c, self.net[0].out_channels, kernel_size=3).apply(init_weights)

            else:
                print('using pretrained AVE backbone weights for audio')
                self.net = load_network(net_name='AVE', ep=60, set_train=True).AV_embedder.aud_backbone


    def forward(self, x):

        x = self.net(x)
        x = torch.layer_norm(x, x.shape[1:])

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
            self.net = nn.Sequential(*list(resnet18(pretrained=False).children())[-3:-2], pool2d)
            self.FC1 = nn.Linear(512, sz).apply(init_weights)
            self.FC2 = nn.Linear(sz,sz).apply(init_weights)

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
        get_max = False,
        compute_cam=True
    ):
        super().__init__()
        self.get_max = get_max
        self.compute_cam = compute_cam

        self.net = nn.Sequential(*list(resnet18(pretrained=True).children())[-4:-1])
        if self.get_max:
            self.net[2] = nn.AdaptiveMaxPool2d(output_size=(1,1))
        self.FC1 = nn.Linear(512, sz).apply(init_weights)

        if self.compute_cam:
            self.FC2 = nn.Linear(sz + sz_cam, sz)
        else:
            self.FC2 = nn.Linear(sz, sz)

    def forward(self, x, cam=None, repeat_aud=conf.contrast_param['flip_img'] and not conf.contrast_param['flip_mic']):

        x = self.net(x)
        x = self.FC1(x.squeeze(-1).squeeze(-1))

        if repeat_aud:
            x = x.repeat(2, 1)

        if self.compute_cam:
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
            AVE_backbone = False,
            small_features = True,
            multi_mic = conf.logmelspectro['multi_mic'],
            compute_cam=True
    ):
        super().__init__()

        self.sz_FC, self.sz_cam = sz_FC, sz_cam
        self.out_tensor = out_tensor
        self.img_only = img_only
        self.AVE_backbone = AVE_backbone
        self.small_features = small_features
        self.multi_mic = multi_mic
        self.compute_cam=compute_cam

        self.img_backbone = backbone(mode='image', AVE_backbone=self.AVE_backbone, small_features=self.small_features)
        self.img_embedder = embedder_img(sz=sz_FC, out_tensor=self.out_tensor, get_max=max_v)

        if not img_only:
            self.aud_backbone = backbone(mode='audio', AVE_backbone=self.AVE_backbone, multi_mic=multi_mic)
            self.aud_embedder = embedder_aud(sz=sz_FC, sz_cam=sz_cam, get_max=max_a, compute_cam=compute_cam)
       

    def forward(self, img, aud=None, cam=None, flip_img=conf.contrast_param['flip_img'], flip_mic=conf.contrast_param['flip_mic'], device=conf.training_param['device'], norm=True):

        img_stem = self.img_backbone(img)
        img_rep = self.img_embedder(img_stem)
        if norm:
            img_rep = torch.layer_norm(img_rep, img_rep.shape[1:])

        if not self.img_only:

            if cam is None:
                cam = torch.zeros((img.shape[0], 1, self.sz_cam)).to(device=device)
            aud_stem = self.aud_backbone(aud)
            aud_rep = self.aud_embedder(aud_stem, cam, repeat_aud=flip_img and not flip_mic)
            if norm:
                aud_rep = torch.layer_norm(aud_rep, aud_rep.shape[1:])

            # return torch.relu(img_rep), torch.relu(aud_rep)
            return img_rep, aud_rep
        else:

            # return torch.relu(img_rep)
            return img_rep

# The spatial audio network ---------------------------------------------------------------------------------------

class aud_spatial(nn.Module):

    def __init__(self,
            sz_FC = 64,
            sz_sq = 64,
            sz_ex = None,
            split_dims=False,
            vertical_only=False,
    ):
        super().__init__()

        self.sz_FC = sz_FC
        self.sz_sq = sz_sq
        self.split_dims = split_dims
        self.v_only = vertical_only

        if not self.v_only:
            if sz_ex is None:
                (self.sz_ex, self.sz_sp) = (14*14, 14) if conf.dnn_arch['small_features'] else (28*28, 28)
            else:
                self.sz_ex = sz_ex
                self.sz_sp = sz_ex**0.5
        else:
            if sz_ex is None:
                (self.sz_ex, self.sz_sp) = (14, 14) if conf.dnn_arch['small_features'] else (28, 28)
            else:
                self.sz_ex = sz_ex
                self.sz_sp = sz_ex

        if self.split_dims:
            self.sz0, self.sz1 = self.sz_sq, self.sz_FC - self.sz_sq
        else:
            self.sz0, self.sz1 = self.sz_FC, self.sz_FC

        self.FC_squeeze = nn.Linear(self.sz0, self.sz0)
        self.FC_squeeze2 = nn.Linear(self.sz0, self.sz_sq)
        self.FC_excite = nn.Linear(self.sz_sq, self.sz_ex)

        self.conv = nn.Conv2d(self.sz1, self.sz_FC, 1)
        self.BN = nn.BatchNorm2d(num_features=self.sz_FC)
        # self.LN = nn.LayerNorm([self.sz_FC, self.sz_sp, self.sz_sp])

    def forward(self, x):

        if self.split_dims:
            x, x0 = torch.split(x, [self.sz1, self.sz0], dim=1)
        else:
            x0 = 1*x

        x0 = self.FC_squeeze(x0.squeeze(-1).squeeze(-1))
        x0 = torch.relu(self.FC_squeeze2(x0))
        x0 = torch.softmax(self.FC_excite(x0), dim=-1).unsqueeze(1)
        x0 = torch.layer_norm(x0, x0.shape[1:])
        x = x.squeeze(-1)

        x = torch.bmm(x, x0)

        x = rearrange(x, 'b c (h w) -> b c h w', w=self.sz_sp)

        x = torch.relu(self.conv(x))

        if self.v_only:
            x = x.repeat(1, 1, self.sz_sp, 1)

        return x

class gcc_spatial(nn.Module):

    def __init__(self,
            sz_FC1 = 480,
            sz_FC = 64,
            sz_cam = 11,
            sz_ex = None,
    ):
        super().__init__()

        self.sz_FC1 = sz_FC1
        self.sz_FC2 = sz_FC
        self.sz_cam = sz_cam

        if sz_ex is None:
            (self.sz_ex, self.sz_sp) = (14*14, 14) if conf.dnn_arch['small_features'] else (28*28, 28)
        else:
            self.sz_ex = sz_ex
            self.sz_sp = sz_ex**0.5

        # self.LN = nn.LayerNorm([15, 64])

        self.conv = nn.Sequential(*list(resnet18(pretrained=True).children())[:-3])

        self.FC1 = nn.Linear(self.sz_FC1, self.sz_FC2)
        self.BN = nn.BatchNorm1d(num_features=self.sz_FC2)
        self.FC2 = nn.Linear(self.sz_FC2+sz_cam, self.sz_ex)
        # self.conv = nn.Conv2d(1,1,1)

    def forward(self, x_ref, x, cam=None, device=conf.training_param['device']):

        if cam is None:
            cam = torch.zeros((x.shape[0], self.sz_cam)).to(device)
        else:
            cam = cam.squeeze(1).float()

        # a = x[0].cpu().detach().numpy()
        # plt.imshow(a, aspect='auto')
        # plt.show()

        x = torch.max_pool2d(x, (1,2), (1,2))

        x = rearrange(x, 'b h w -> b (h w)')

        x = self.FC1(x)
        x = self.FC2(torch.cat((x, cam), dim=-1))


        # x = self.FC3(torch.cat((x, cam), dim=-1))
        # x = self.FC3(torch.cat((x, cam), dim=-1))

        x = torch.layer_norm(x, [x.shape[-2], x.shape[-1]])

        x = torch.softmax(x, dim=-1).unsqueeze(1)

        x_ref = x_ref.squeeze(-1)

        x = torch.bmm(x_ref, x)

        x = rearrange(x, 'b c (h w) -> b c h w', h=self.sz_sp)

        return x

class gcc_simple(nn.Module):

    def __init__(self,
            sz_FC = 128,
            sz_FC1 = 128,
            sz_FC2 = 64,
            sz_cam = 11,
            sz_ex = None,
            vertical_only=False, 
            
    ):
        super().__init__()

        self.sz_FC1 = sz_FC1
        self.sz_FC2 = sz_FC2
        self.sz_cam = sz_cam
        self.v_only = vertical_only

        if not self.v_only:
            if sz_ex is None:
                (self.sz_ex, self.sz_sp) = (14*14, 14) if conf.dnn_arch['small_features'] else (28*28, 28)
            else:
                self.sz_ex = sz_ex
                self.sz_sp = sz_ex**0.5
        else:
            if sz_ex is None:
                (self.sz_ex, self.sz_sp) = (14, 14) if conf.dnn_arch['small_features'] else (28, 28)
            else:
                self.sz_ex = sz_ex
                self.sz_sp = sz_ex

        self.FC1 = nn.Linear(15, 32)
        self.FC2 = nn.Linear(32, 32)
        self.FC3 = nn.Linear(43, 32)
        self.FC4 = nn.Linear(32, 32)
        self.FC5 = nn.Linear(32, self.sz_sp)

        self.BN1 = nn.BatchNorm1d(num_features=32)
        self.BN2 = nn.BatchNorm1d(num_features=self.sz_sp)

    def forward(self, x_ref, x, cam=None, device=conf.training_param['device']):

        if cam is None:
            cam = torch.zeros((x.shape[0], self.sz_cam)).to(device)
        else:
            cam = cam.squeeze(1).float()

        x = self.FC1(x)
        # x = torch.layer_norm(x, [x.shape[-3], x.shape[-2], x.shape[-1]])

        x = torch.layer_norm(torch.relu(self.FC2(x)), x.shape[1:])

        x = self.BN1(torch.relu(self.FC3(torch.cat((x, cam), dim=-1))))


        x = torch.relu(self.FC4(x))
        x = torch.relu(self.FC5(x))
        # x = torch.layer_norm(x, x.shape[1:]).unsqueeze(1)

        x = self.BN2(x).unsqueeze(1)

        # x = torch.sigmoid(x)

        x = torch.softmax(x, dim=-1)

        # for i in range(10):

        #     a = x[i].cpu().detach().numpy()
        #     plt.subplot(10, 1, i+1)
        #     plt.axis('off')
        #     plt.imshow(a)
        #     # plt.colorbar()

        # plt.show()

        # print(torch.argmax(x, dim=-1))

        x_ref = x_ref.squeeze(-1)

        x = torch.bmm(x_ref, x)

        x = rearrange(x, 'b c (h w) -> b c h w', w=self.sz_sp)

        if self.v_only:
            x = x.repeat(1, 1, self.sz_sp, 1)

        # a = x[0].mean(0).cpu().detach().numpy()
        # plt.subplot(122)
        # plt.imshow(a)

        return x


class gcc_spatial_v2(nn.Module):

    def __init__(self,
            sz_FC = 128,
            sz_FC1 = 128,
            sz_FC2 = 64,
            sz_cam = 11,
            sz_ex = None,
            vertical_only=False, 
            
    ):
        super().__init__()

        self.sz_FC1 = sz_FC1
        self.sz_FC2 = sz_FC2
        self.sz_cam = sz_cam
        self.v_only = vertical_only

        if not self.v_only:
            if sz_ex is None:
                (self.sz_ex, self.sz_sp) = (14*14, 14) if conf.dnn_arch['small_features'] else (28*28, 28)
            else:
                self.sz_ex = sz_ex
                self.sz_sp = sz_ex**0.5
        else:
            if sz_ex is None:
                (self.sz_ex, self.sz_sp) = (14, 14) if conf.dnn_arch['small_features'] else (28, 28)
            else:
                self.sz_ex = sz_ex
                self.sz_sp = sz_ex

        self.conv1 = nn.Conv2d(1, 64, (7,7), (1,1), (0,0), bias=False)
        self.conv2 = nn.Conv2d(64, 128, (5,5), (1,1), (0,0))

        self.pool = nn.AdaptiveMaxPool2d(output_size=(1,1))

        self.FC1 = nn.Linear(self.sz_FC1, self.sz_FC2)
        self.FC2 = nn.Linear(self.sz_FC2+sz_cam, self.sz_FC2)
        self.FC3 = nn.Linear(self.sz_FC2, self.sz_FC2)
        self.FC32 = nn.Linear(self.sz_FC2, self.sz_FC2)
        self.FC4 = nn.Linear(self.sz_FC2, self.sz_ex)

    def forward(self, x_ref, x, cam=None, device=conf.training_param['device']):

        if cam is None:
            cam = torch.zeros((x.shape[0], self.sz_cam)).to(device)
        else:
            cam = cam.squeeze(1).float()

        cam = torch.layer_norm(cam, [11])

        x = torch.relu(self.conv1(x.unsqueeze(1)))
        x = torch.layer_norm(x, [x.shape[-3], x.shape[-2], x.shape[-1]])

        x = torch.relu(self.conv2(x))
        x = torch.layer_norm(x, [x.shape[-3], x.shape[-2], x.shape[-1]])

        x = self.pool(x).squeeze(-1).squeeze(-1)

        x = torch.layer_norm(torch.relu(self.FC1(x)), [self.sz_FC2])
        x = torch.relu(self.FC2(torch.cat((x, cam), dim=-1)))
        x = torch.relu(self.FC3(x))
        x = torch.relu(self.FC32(x))
        x = torch.relu(self.FC4(x))

        x = torch.softmax(x/0.2, dim=-1)
        x = x.unsqueeze(1)
        x = torch.relu(torch.layer_norm(x, x.shape[1:]))

        x_ref = x_ref.squeeze(-1)

        x = torch.bmm(x_ref, x)

        x = rearrange(x, 'b c (h w) -> b c h w', w=self.sz_sp)

        if self.v_only:
            x = x.repeat(1, 1, self.sz_sp, 1)

        return x

# MERGING NETWORKS ------------------------------------------------------------------------------------------------

class AVE(nn.Module):

    def __init__(self,
            sz_FC = conf.dnn_arch['FC_size'],
            sz_cam = 11,
            set_train = not conf.training_param['inference'],
            multi_frame = conf.training_param['frame_seq'],
            heatmap = conf.dnn_arch['heatmap'],
            merge_L2 = conf.dnn_arch['ave_L2'],
    ):
        super().__init__()
        self.sz_FC, self.sz_cam = sz_FC, sz_cam
        self.set_train = set_train
        self.multi_frame = multi_frame
        self.heatmap = heatmap
        self.merge_L2 = merge_L2


        out_tensor = True if self.heatmap else False

        self.AV_embedder = embedder_AV(sz_FC=sz_FC, sz_cam=sz_cam, out_tensor=out_tensor, 
                                max_a=True, max_v=True, AVE_backbone=False)

        # for multiple frames, out_tensor is false because I have not found a way to spatially encode information yet.
        if self.multi_frame:
            self.gru_layers = 2
            self.gru_hidden = sz_FC
            self.gru = nn.GRU(sz_FC, self.gru_hidden, self.gru_layers, batch_first=True, bidirectional=False)
            self.img_embedder = embedder_AV(sz_FC=sz_FC, sz_cam=sz_cam, out_tensor=out_tensor, max_a=True, max_v=True, img_only=True)
            self.FC_time = nn.Linear(sz_FC, sz_FC)

        self.FC_final = nn.Linear(1,2)

    def forward(self, img, aud, cam=None, device=conf.training_param['device']):

        bs = img.shape[0]

        if self.multi_frame:

            img_seq = torch.zeros(bs, img.shape[1], self.sz_FC).to(device)

            for frame_idx in range(img.shape[1]):
                
                if frame_idx == 0:
                    im_ref, aud_rep = self.AV_embedder(img[:, frame_idx], aud, cam)
                    im_ref = im_ref.squeeze(-1).squeeze(-1)

                    img_seq[:, frame_idx, :] = 1*im_ref
                    im_prev = 1*im_ref
                else:
                    im_new = self.img_embedder(img[:, frame_idx]).squeeze(-1).squeeze(-1)

                    if conf.dnn_arch['seq_res']:
                        im_res = im_new - im_prev
                    else:
                        im_res = im_new

                    img_seq[:, frame_idx, :] = 1*im_res

                    # img_all = torch.concat((img_all, im_res), dim=1)
                    im_prev = 1*im_new

            #     im_features[:,frame_idx,:] = im_prev.squeeze(1)

            # im_features = img_seq[0].cpu().detach().numpy()

            # plt.figure()
            # for i in range(5):
            #     plt.plot(im_features[i])
            # plt.show()

            h0 = torch.zeros(self.gru_layers, img_seq.shape[0], self.gru_hidden).to(device)

            if conf.dnn_arch['seq_res']:
                img_gru = self.gru(img_seq[:,1:,:], h0)[0]
            else:
                img_gru = self.gru(img_seq, h0)[0]

            img_rep = img_gru[:,-1,:] 

            if conf.dnn_arch['seq_skip'] or conf.dnn_arch['seq_res']:
                img_rep = self.FC_time(img_rep) + im_ref.squeeze(-1).squeeze(-1)
            else:
                img_rep = self.FC_time(img_rep)

        else:

            img_rep, aud_rep = self.AV_embedder(img, aud, cam)

        img_rep = torch.layer_norm(img_rep.squeeze(-1).squeeze(-1), normalized_shape=[self.sz_FC])
        aud_rep = torch.layer_norm(aud_rep.squeeze(-1).squeeze(-1), normalized_shape=[self.sz_FC])

        if self.set_train:

            x_score = torch.zeros((bs, bs)).to(device=device)
            for i, im in enumerate(img_rep):
                for j, au in enumerate(aud_rep):
                    if self.merge_L2:
                        im_norm = im / torch.linalg.vector_norm(im, dim=-1)
                        au_norm = au / torch.linalg.vector_norm(au, dim=-1)
                        x_score[i,j] = (im_norm - au_norm).pow(2).sum(-1).sqrt()

                    else:

                        x_score[i,j] = F.cosine_similarity(im, au, dim=0, eps=1e-08)


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


class MergeNet(nn.Module):

    def __init__(self,
            merge_type = conf.dnn_arch['net_name'],
            sz_FC = conf.dnn_arch['FC_size'],
            sz_cam = 11,
            set_train = True,
            ave_backbone = conf.dnn_arch['ave_backbone'],
            small_features = conf.dnn_arch['small_features'],
            loss_type = 'class'
    ):
        super().__init__()
        self.merge_type = merge_type
        self.sz_FC, self.sz_cam = sz_FC, sz_cam
        self.set_train = set_train
        self.small_features = small_features

        
        self.loss_type = loss_type

        self.AV_embedder = embedder_AV(sz_FC=sz_FC, sz_cam=sz_cam, AVE_backbone=ave_backbone, small_features=self.small_features)
        
        if merge_type=='AVOL':
            self.conv_final = nn.Conv2d(1, 1, kernel_size=1)
        elif merge_type=='VSL':
            cos_dim = 0 if self.set_train else 1
            self.cos = nn.CosineSimilarity(dim=cos_dim)
        elif merge_type=='SL2':
            self.conv_final = nn.Conv2d(1, 1, kernel_size=1)
            self.FC_final = nn.Linear(1,2)


    def forward(self, img, aud, cam=None, flip_img=conf.contrast_param['flip_img'], flip_mic=conf.contrast_param['flip_img'], device=conf.training_param['device'], hard_negatives=False):

        bs = img.shape[0]

        img_rep, aud_rep = self.AV_embedder(img, aud, cam, flip_img=flip_img and not flip_mic)


        if self.set_train:
    
            x_score = torch.zeros((bs, bs)).to(device=device)

            for i, im in enumerate(img_rep):
                for j, au in enumerate(aud_rep):

                    if self.merge_type == 'AVOL':
                        x_map = (im * au).mean(dim=0, keepdim=True)
                        x_map = torch.sigmoid(self.conv_final(x_map)).squeeze(0)
                        x_score[i,j] = torch.amax(x_map)
                    elif self.merge_type=='VSL':
                        x_map = self.cos(im, au)
                        x_score[i,j] = torch.amax(x_map)
                    elif self.merge_type == 'SL2':
                        im_norm = im / torch.linalg.vector_norm(im, dim=(-3))
                        au_norm = au / torch.linalg.vector_norm(au, dim=(-3))
                        x_map = (im_norm - au_norm).pow(2).sum(0).sqrt().unsqueeze(0)
                        if self.loss_type == 'class':
                            x_score[i,j] = torch.amin(x_map)
                        else:
                            x_map = -1*x_map
                            # x_map = torch.layer_norm(-1*x_map, x_map.shape)
                            x_score[i,j] = torch.amax(x_map)
            
            if self.merge_type == 'SL2' and self.loss_type=='class':
                x_score = rearrange(x_score, 'b1 b2 -> (b1 b2)').unsqueeze(-1)
                x_score = self.FC_final(x_score)
                x_score = rearrange(x_score, '(b1 b2) c -> b1 b2 c', b1=bs)
                

            return x_score

        else:
            if self.merge_type == 'AVOL':
                x_map = (img_rep * aud_rep).mean(dim=1, keepdim=True)
                x_map = torch.sigmoid(self.conv_final(x_map)).squeeze(1)
                # x_map = torch.layer_norm(x_map, x_map.shape)
            elif  self.merge_type=='VSL':
                x_map = self.cos(img_rep, aud_rep)
                # x_map = torch.layer_norm(x_map, x_map.shape)
            elif self.merge_type == 'SL2':
                if img_rep.shape[0]==1:
                    img_rep, aud_rep = img_rep.squeeze(0), aud_rep.squeeze(0)
                    im_norm = img_rep / torch.linalg.vector_norm(img_rep, dim=(-3)).unsqueeze(0)
                    au_norm = aud_rep / torch.linalg.vector_norm(aud_rep, dim=(-3)).unsqueeze(0)
                    x_map = -1*(im_norm - au_norm).pow(2).sum(0).sqrt().unsqueeze(0)
                else:
                    for i in range(bs):
                        au = aud_rep[i]
                        im = img_rep[i]
                        im_norm = im / torch.linalg.vector_norm(im, dim=(-3))
                        au_norm = au / torch.linalg.vector_norm(au, dim=(-3))
                        x_m = -1*(im_norm - au_norm).pow(2).sum(0).sqrt().unsqueeze(0)
                        if i==0:
                            x_map = x_m
                        else:
                            x_map = torch.cat([x_map, x_m], dim=0)


            x_score = torch.amax(x_map, (1,2))

            return x_score, x_map


class SpaceNet(nn.Module):

    def __init__(self,
            merge_type = conf.dnn_arch['net_name'],
            sz_FC = conf.dnn_arch['FC_size'],
            sz_cam = 11,
            set_train = True,
            ave_backbone = conf.dnn_arch['ave_backbone'],
            small_features = conf.dnn_arch['small_features'],
            split_dims=False,
            vertical_only=False,
    ):
        super().__init__()
        self.merge_type = merge_type
        self.sz_FC, self.sz_cam = sz_FC, sz_cam
        self.set_train = set_train
        self.small_features = small_features
        self.split_dims = split_dims
        self.vertical_only = vertical_only

        self.AV_embedder = embedder_AV(sz_FC=sz_FC, sz_cam=sz_cam, AVE_backbone=ave_backbone, small_features=self.small_features)
        self.space_embed = aud_spatial(sz_FC=sz_FC, split_dims=self.split_dims, vertical_only=self.vertical_only)


        if merge_type=='AVOL':
            self.conv_final = nn.Conv2d(1, 1, kernel_size=1)
        elif merge_type=='VSL':
            cos_dim = 0 if self.set_train else 1
            self.cos = nn.CosineSimilarity(dim=cos_dim)


    def forward(self, img, aud, cam=None, flip_img=conf.contrast_param['flip_img'], flip_mic=conf.contrast_param['flip_img'], device=conf.training_param['device'], hard_negatives=False):

        bs = img.shape[0]

        img_rep, aud_rep = self.AV_embedder(img, aud, cam, flip_img=flip_img and not flip_mic)

        aud_rep = self.space_embed(aud_rep)

        if self.set_train:
    
            x_score = torch.zeros((bs, bs)).to(device=device)

            for i, im in enumerate(img_rep):
                for j, au in enumerate(aud_rep):
                    if self.merge_type == 'AVOL':
                        x_map = (im * au).mean(dim=0, keepdim=True)
                        x_map = torch.sigmoid(self.conv_final(x_map)).squeeze(0)
                    elif self.merge_type=='VSL':
                        x_map = self.cos(im, au)

                    x_score[i,j] = torch.amax(x_map)

            return x_score

        else:
            if self.merge_type == 'AVOL':
                x_map = (img_rep * aud_rep).mean(dim=1, keepdim=True)
                x_map = torch.sigmoid(self.conv_final(x_map)).squeeze(1)
            elif  self.merge_type=='VSL':
                x_map = self.cos(img_rep, aud_rep)

            x_score = torch.amax(x_map, (1,2))

            return x_score, x_map


class FactorNet(nn.Module):

    def __init__(self,
            merge_type = conf.dnn_arch['vsl_merge_type'],
            sz_FC = conf.dnn_arch['FC_size'],
            sz_cam = 11,
            set_train = True,
            ave_backbone = conf.dnn_arch['ave_backbone'],
            small_features = conf.dnn_arch['small_features'],
            vertical_only=False,
            simple_aud = False,
    ):
        super().__init__()
        self.merge_type = merge_type
        self.sz_FC, self.sz_cam = sz_FC, sz_cam
        self.set_train = set_train
        self.small_features = small_features
        self.vertical_only=vertical_only
        self.simple_aud = simple_aud

        self.AV_embedder = embedder_AV(sz_FC=sz_FC, sz_cam=sz_cam, 
                                    AVE_backbone=ave_backbone, small_features=self.small_features, 
                                    multi_mic=False, compute_cam=False)

        if self.simple_aud:
            self.space_embed = gcc_simple(sz_FC=sz_FC, 
                                vertical_only=self.vertical_only
                                )
        else:
            self.space_embed = gcc_spatial_v2(sz_FC=sz_FC, 
                                vertical_only=self.vertical_only
                                )

        if merge_type=='AVOL':
            self.conv_final = nn.Conv2d(1, 1, kernel_size=1)
        elif merge_type=='VSL':
            cos_dim = 0 if self.set_train else 1
            self.cos = nn.CosineSimilarity(dim=cos_dim)
        elif merge_type=='L2':
            self.FC_final = nn.Linear(1,2)


    def forward(self, img, aud, cam=None, flip_img=conf.contrast_param['flip_img'], flip_mic=conf.contrast_param['flip_img'], device=conf.training_param['device'], hard_negatives=False):

        bs = img.shape[0]

        aud_ref, aud_gcc = gcc_collapse(aud, simple_aud=self.simple_aud)

        img_rep, aud_rep = self.AV_embedder(img, aud_ref, cam, flip_img=flip_img and not flip_mic)

        aud_rep = self.space_embed(aud_rep, aud_gcc, cam)

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


class MixNet(nn.Module):

    def __init__(self,
            merge_type = conf.dnn_arch['net_name'],
            sz_FC = conf.dnn_arch['FC_size'],
            sz_cam = 11,
            set_train = True,
            ave_backbone = conf.dnn_arch['ave_backbone'],
            small_features = conf.dnn_arch['small_features'],
    ):
        super().__init__()
        self.merge_type = merge_type
        self.sz_FC, self.sz_cam = sz_FC, sz_cam
        self.set_train = set_train
        self.small_features = small_features

        self.AV_embedder = embedder_AV(sz_FC=sz_FC, sz_cam=sz_cam, 
                                    AVE_backbone=ave_backbone, small_features=self.small_features, 
                                    multi_mic=False, compute_cam=False)

        self.space_embed = gcc_spatial(sz_FC=sz_FC)

        self.conv_final = nn.Conv2d(1, 1, kernel_size=1)
        self.cos = nn.CosineSimilarity(dim=1)


    def forward(self, img, aud, cam=None, flip_img=conf.contrast_param['flip_img'], flip_mic=conf.contrast_param['flip_img'], device=conf.training_param['device'], hard_negatives=False):

        bs = img.shape[0]

        aud_ref, aud_gcc = gcc_collapse(aud)

        img_rep, aud_rep = self.AV_embedder(img, aud_ref, cam, flip_img=flip_img and not flip_mic)

        aud_rep = self.space_embed(aud_rep, aud_gcc, cam)

        if self.set_train:
    
            x_score = self.cos(img_rep, aud_rep)

            return x_score

        else:

            x_map = self.cos(img_rep, aud_rep)

            x_score = torch.amax(x_map, (1,2))

            return x_score, x_map


