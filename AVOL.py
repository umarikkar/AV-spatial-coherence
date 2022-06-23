import datetime
import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class SubNet_main(nn.Module):

    def __init__(self, mode, multi_mic=True):
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

    def __init__(self, multi_mic=True):
        super().__init__()

        self.VideoNet = SubNet_main(mode='video')
        if not multi_mic:
            self.AudioNet = SubNet_main(mode='audio', multi_mic=False)
        else:
            self.AudioNet = SubNet_main(mode='audio', multi_mic=True)

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

        # ------ FINAL LAYER FC (NO HEATMAP) ---------------

        c = x.shape[1]
        h, w = x.shape[-2], x.shape[-1]
        x = torch.mean(x.view(-1,c, h*w), dim=-1)

        x = self.FC_final(x)

        return x


        # ------ WITH HEATMAP  ---------------

        # x = torch.mean(x, dim=1) # collapsing along dimension

        # h, w = x.shape[-2], x.shape[-1]
        # x_class = torch.mean(x.view(-1, h*w), dim=-1)

        # x_class = torch.sigmoid(x_class)

        # # GAP for au only.
        # return x, x_class




def Trainer(net, epochs, loss_fn, optimiser, train_loader, val_loader, multi_mic=True):

    device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
    print(f"Training on device {device}.")

    for epoch in range(1, epochs + 1):

        loss_train = 0.0
        acc = 0.0
        net.train()

        for data in train_loader:

            
            BS = data[0].shape[0]
            all_frames = data[-1]
            cam = data[1]

            imgs_pos = all_frames[torch.randint(len(all_frames), (1,1))]

            # multi mic false!
            if not multi_mic:
                audio = data[0][:,-1,:,:].unsqueeze(1)
            else:
                audio = data[0]

            # create contrastive batch (shift by some n)
            roll_idx = int(torch.randint(low=1, high=BS, size=(1,1)))
            roll_idx = 1
            imgs_neg = torch.roll(imgs_pos, roll_idx, dims=0)
            cam_neg = torch.roll(cam, roll_idx, dims=0)

            # imgs_pos = torch.ones_like(imgs_pos)
            # imgs_neg = torch.zeros_like(imgs_neg)

            imgs_all = torch.concat((imgs_pos, imgs_neg), dim=0).to(device=device)
            audio_all = torch.concat((audio, audio), dim=0).to(device=device)
            cam_all = torch.concat((cam, cam_neg), dim=0).to(device=device)
            
            """
            # HEATMAP LOSS ONE ------------------------------------------------------------

            heat_pos, out_pos = net(imgs_pos, audio)
            heat_neg, out_neg = net(imgs_neg, audio)

            labels_pos = torch.ones(BS).to(device=device)
            labels_neg = torch.zeros(BS).to(device=device)

            """
            # # FC layer loss one ----------------------------------------------------------- 

            out = net(imgs_all, audio_all, cam_all)
            
            one = torch.ones((BS,1))
            zer = torch.zeros((BS,1))

            labels_pos = torch.concat((one, zer), dim=-1)
            labels_neg = torch.concat((zer, one), dim=-1)

            labels_all = torch.concat((labels_pos, labels_neg), dim=0).to(device=device)

            # # CALCULATING ACCURACY -------------------------------------------------------
            
            _, idx1 = torch.max(out, dim=-1)
            _, idx2 = torch.max(labels_all, dim=-1)

            acc += 100*(1 - torch.abs(idx1-idx2).sum() / len(idx1))

            ## LOSS FN ---------------------------------------------------------------------

            loss = loss_fn(out, labels_all)

            l2_lambda = 0.0001
            l2_norm = sum(p.pow(2.0).sum() for p in net.parameters())**0.5 # L2 reg for all the weights

            loss = loss + l2_lambda*l2_norm

            optimiser.zero_grad() 
            loss.backward()
            optimiser.step()

            loss_train += loss.item()
            # print(loss.item())


        # # validations
        # loss_val = 0.0
        # net.eval()
        # with torch.no_grad():
        #     for imgs, labels in val_loader:

        #         imgs = imgs.to(device=device)
        #         labels = labels.to(device=device)
        #         outputs = net(imgs)

        #         loss = loss_fn(outputs, labels)
        #         loss_val += loss.item()
            
                
        if epoch == 1 or epoch % 1 == 0:
            
            dt = datetime.datetime.now()  

            # print('{} Epoch {}, Training loss {}, val loss {}'.format(
            #     dt, epoch,
            #     round(loss_train / len(train_loader), 4),
            #     round(loss_val / len(val_loader), 4),
            # ))


            # path = os.path.join(os.getcwd(), 'checkpoints', 'SepNetViT_nopointconv_ep_' + str(epoch) + '_checkpoint.pt')
            # # print(path)
            # torch.save({ 
            #     'epoch': epoch,
            #     'model': net.state_dict(),
            #     'optimizer': optimiser.state_dict()}, path)


            print('{} n_batches {}, Epoch {}, Training loss {} Training acc {}%'.format(
                dt, 
                len(train_loader),
                epoch,
                round(loss_train / len(train_loader), 4),
                round(float(acc/len(train_loader)), 2),
            ))


            # path = os.path.join(os.getcwd(), 'checkpoints', 'AVOL_scratch_' + str(epoch) + '_checkpoint.pt')
            # # print(path)
            # torch.save({ 
            #     'epoch': epoch,
            #     'model': net.state_dict(),
            #     'optimizer': optimiser.state_dict()}, path)

    return

