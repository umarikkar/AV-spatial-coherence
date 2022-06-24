import datetime
import os
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import core.config as conf
from torchvision.transforms import transforms


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


class SubNet_vid(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(512,128,kernel_size=1)
        self.conv2 = nn.Conv2d(128,128,kernel_size=1)
        self.BN = nn.BatchNorm2d(num_features=128)

    def forward(self, x):

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.BN(self.conv2(x)))

        return x


class SubNet_aud(nn.Module):
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

    def __init__(self, multi_mic=conf.logmelspectro['get_gcc'], heatmap=conf.dnn_arch['heatmap']):
        super().__init__()
        self.heatmap = heatmap
        self.VideoNet = SubNet_main(mode='video')
        self.AudioNet = SubNet_main(mode='audio', multi_mic=multi_mic)

        self.VideoMerge = SubNet_vid()
        self.AudioMerge = SubNet_aud()

        self.conv1 = nn.Conv2d(128, 62, kernel_size=1)

        self.BN = nn.BatchNorm2d(num_features=128)

        self.FC_final = nn.Linear(62, 2)

    def forward(self, im, au, cam):

        im = self.VideoNet(im)
        au = self.AudioNet(au)

        im = self.VideoMerge(im)
        au = self.AudioMerge(au, cam_ID=cam).unsqueeze(-1).unsqueeze(-1)

        # x = im + im*au
        x = im*au 

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

            x = self.FC_final(x)

            return x


def create_samples(data, device='cpu', augment=True, take_to_device=True):

    batch_size = data[0].shape[0]
    all_frames = data[-1]
    cam = data[1]

    imgs_pos = all_frames.squeeze(1)
    audio = data[0]

    # create contrastive batch (shift by some n)
    if batch_size > 1:
        roll_idx = random.randint(1, batch_size-1)
    else:
        roll_idx = 1

    imgs_neg = torch.roll(imgs_pos, roll_idx, dims=0)
    cam_neg = torch.roll(cam, roll_idx, dims=0)

    imgs_all = torch.concat((imgs_pos, imgs_neg), dim=0)
    audio_all = torch.concat((audio, audio), dim=0)
    cam_all = torch.concat((cam, cam_neg), dim=0)

    if augment:
        for i in range(imgs_all.shape[0]):
            imgs_all[i] = transforms.ColorJitter((0.8, 1.2), (0.8, 1.2), (0.8, 1.2), (-0.1, 0.1))(imgs_all[i])
            imgs_all[i] = transforms.RandomGrayscale(0.2)(imgs_all[i])

    if take_to_device:
        imgs_all = imgs_all.to(device=device)
        audio_all = audio_all.to(device=device)
        cam_all = cam_all.to(device=device)

    return imgs_all, audio_all, cam_all


def create_labels(BS, device='cpu', heatmap=conf.dnn_arch['heatmap']):

    if heatmap:
        labels_pos = torch.ones(BS).to(device=device)
        labels_neg = torch.zeros(BS).to(device=device)
        labels_all = torch.concat((labels_pos, labels_neg), dim=0).to(device=device)
    else:
        one = torch.ones((BS,1))
        zer = torch.zeros((BS,1))
        labels_pos = torch.concat((one, zer), dim=-1)
        labels_neg = torch.concat((zer, one), dim=-1)
        labels_all = torch.concat((labels_pos, labels_neg), dim=0).to(device=device)

    return labels_all



def Trainer(net, epochs, loss_fn, optimiser, train_loader, val_loader, multi_mic=conf.logmelspectro['get_gcc'], heatmap=conf.dnn_arch['heatmap']):

    device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
    print(f"Training on device {device}.")

    for epoch in range(1, epochs + 1):

        loss_train = 0.0
        acc_train = 0.0
        loss_l2 = 0.0
        net.train()

        for data in train_loader:

            BS = data[0].shape[0] # batch size
            
            imgs_all, audio_all, cam_all = create_samples(data, device=device, augment=True)

            out_all = net(imgs_all, audio_all, cam_all)
            out = out_all[-1] if heatmap else out_all

            labels_all = create_labels(BS, device=device, heatmap=heatmap)

            if heatmap:
                idx1 = torch.zeros_like(out)
                idx1[out >= 0.5] = 1
                idx2 = labels_all

            else: # outputs with FC out as final layer. Learns a bit faster                
                _, idx1 = torch.max(out, dim=-1)
                _, idx2 = torch.max(labels_all, dim=-1)

            acc_train += 100*(1 - torch.abs(idx1-idx2).sum() / len(idx1))
            
            # loss function --->
            loss = loss_fn(out, labels_all)
            l2_lambda = 0.001
            l2_reg = l2_lambda*sum(p.pow(2.0).sum() for p in net.parameters())**0.5 # L2 reg for all the weights
            loss += l2_reg

            optimiser.zero_grad() 
            loss.backward()
            optimiser.step()

            loss_train += loss.item()
            loss_l2 += l2_reg

        # validations -----------------------------------------------------------------------------
        loss_val = 0.0
        acc_val = 0.0
        net.eval()

        with torch.no_grad():

            for data in val_loader:

                BS = data[0].shape[0]

                imgs_all, audio_all, cam_all = create_samples(data, device=device, augment=False)
                out_all = net(imgs_all, audio_all, cam_all)
                out = out_all[-1] if heatmap else out_all

                labels_all = create_labels(BS, device=device, heatmap=heatmap)

                if heatmap:
                    idx1 = torch.zeros_like(out)
                    idx1[out >= 0.5] = 1
                    idx2 = labels_all

                else: # outputs with FC out as final layer. Learns a bit faster                
                    _, idx1 = torch.max(out, dim=-1)
                    _, idx2 = torch.max(labels_all, dim=-1)

                acc_val += 100*(1 - torch.abs(idx1-idx2).sum() / len(idx1))

                # loss function --->
                lossVal = loss_fn(out, labels_all)
                loss_val += lossVal.item()

                
        if epoch == 1 or epoch % 4 == 0:
            
            dt = datetime.datetime.now()  

            net_path = os.path.join(os.getcwd(), 'results', 'checkpoints', 'MultiChannel_augment_11k_ep_' + str(epoch) + '_checkpoint.pt')
            file_path =  os.path.join(os.getcwd(), 'results')
            torch.save({ 
                'epoch': epoch,
                'model': net.state_dict(),
                'optimizer': optimiser.state_dict()}, net_path)


            print('{} Epoch {}, Train loss {}, Train acc {}%, Val loss {}, Val acc {}%'.format(
                dt, 
                epoch,
                round(loss_train / len(train_loader), 4),
                round(float(acc_train/len(train_loader)), 2),
                round(loss_val / len(val_loader), 4),
                round(float(acc_val/len(val_loader)), 2),
            ),
            file=open(os.path.join(file_path, 'MultiChannel_augment_11k.txt'), "a")
            )


    return

