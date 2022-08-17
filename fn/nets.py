from cgitb import small
from sklearn.datasets import load_files
import torch
import torch.nn as nn
from einops import rearrange
from torchvision.models import resnet18
from fn.losses import loss_AVE, loss_AVOL, loss_AVOL_flip, loss_VSL, loss_VSL_flip
import os
import core.config as conf

# SET AND LOAD NETWORK  ------------------------------------------------------------------------------------------


def set_network(set_train=True, net_name=conf.dnn_arch['net_name'], print_net=True):

    if net_name=='AVOL':
        net = MergeNet(merge_type='AVOL', set_train=set_train)
        if conf.contrast_param['flip_img'] or conf.contrast_param['flip_mic']:
            loss_fn = loss_AVOL_flip()
        else:
            loss_fn = loss_AVOL()
    elif net_name=='AVE':
        net = AVE(set_train=set_train)
        loss_fn = loss_AVE() 
    elif net_name=='VSL':
        net =  MergeNet(merge_type='VSL', set_train=set_train)
        # net = VSL_large(set_train=set_train)
        if conf.contrast_param['flip_img'] or conf.contrast_param['flip_mic']:
            loss_fn = loss_VSL_flip()
        else:
            loss_fn = loss_VSL()

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

    ):
        super().__init__()
        self.freeze = freeze
        self.AVE_backbone = AVE_backbone
        self.small_features = small_features

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
                self.multi_mic = conf.logmelspectro['multi_mic']
                aud_c = 16 if self.multi_mic else 1
                self.net[0] = nn.Conv2d(aud_c, self.net[0].out_channels, kernel_size=3).apply(init_weights)

            else:
                print('using pretrained AVE backbone weights for audio')
                self.net = load_network(net_name='AVE', ep=60, set_train=True).AV_embedder.aud_backbone


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

    def forward(self, x, cam=None, repeat_aud=conf.contrast_param['flip_img'] and not conf.contrast_param['flip_mic']):

        x = self.net(x)
        x = self.FC1(x.squeeze(-1).squeeze(-1))

        if repeat_aud:
            x = x.repeat(2, 1)

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
    ):
        super().__init__()

        self.sz_FC, self.sz_cam = sz_FC, sz_cam
        self.out_tensor = out_tensor
        self.img_only = img_only
        self.AVE_backbone = AVE_backbone
        self.small_features = small_features

        self.img_backbone = backbone(mode='image', AVE_backbone=self.AVE_backbone, small_features=self.small_features)
        self.img_embedder = embedder_img(sz=sz_FC, out_tensor=self.out_tensor, get_max=max_v)

        if not img_only:
            self.aud_backbone = backbone(mode='audio', AVE_backbone=self.AVE_backbone)
            self.aud_embedder = embedder_aud(sz=sz_FC, sz_cam=sz_cam, get_max=max_a)
       

    def forward(self, img, aud=None, cam=None, flip_img=conf.contrast_param['flip_img'], flip_mic=conf.contrast_param['flip_mic'], device=conf.training_param['device']):

        img_stem = self.img_backbone(img)
        img_rep = self.img_embedder(img_stem)

        if not self.img_only:

            if cam is None:
                cam = torch.zeros((img.shape[0], 1, self.sz_cam)).to(device=device)
            aud_stem = self.aud_backbone(aud)
            aud_rep = self.aud_embedder(aud_stem, cam, repeat_aud=flip_img and not flip_mic)

            return img_rep, aud_rep
        else:

            return img_rep


# MERGING NETWORKS ------------------------------------------------------------------------------------------------

class AVE(nn.Module):

    def __init__(self,
            sz_FC = 64,
            sz_cam = 11,
            set_train = not conf.training_param['inference'],
            multi_frame = conf.training_param['frame_seq'],
            heatmap = conf.dnn_arch['heatmap'],
    ):
        super().__init__()
        self.sz_FC, self.sz_cam = sz_FC, sz_cam
        self.set_train = set_train
        self.multi_frame = multi_frame
        self.heatmap = heatmap

        out_tensor = True if self.heatmap else False

        self.AV_embedder = embedder_AV(sz_FC=sz_FC, sz_cam=sz_cam, out_tensor=out_tensor, 
                                max_a=True, max_v=True, AVE_backbone=False)

        # for multiple frames, out_tensor is false because I have not found a way to spatially encode information yet.
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


class MergeNet(nn.Module):

    def __init__(self,
            merge_type = conf.dnn_arch['net_name'],
            sz_FC = conf.dnn_arch['FC_size'],
            sz_cam = 11,
            set_train = True,
            ave_backbone = conf.dnn_arch['ave_backbone'],
            small_features = conf.dnn_arch['small_features']
    ):
        super().__init__()
        self.merge_type = merge_type
        self.sz_FC, self.sz_cam = sz_FC, sz_cam
        self.set_train = set_train
        self.small_features = small_features

        self.AV_embedder = embedder_AV(sz_FC=sz_FC, sz_cam=sz_cam, AVE_backbone=ave_backbone, small_features=self.small_features)
        
        if merge_type=='AVOL':
            self.conv_final = nn.Conv2d(1, 1, kernel_size=1)
        elif merge_type=='VSL':
            cos_dim = 0 if self.set_train else 1
            self.cos = nn.CosineSimilarity(dim=cos_dim)


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

