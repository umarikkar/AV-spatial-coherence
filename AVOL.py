import datetime
import os

import torch
import torch.nn as nn


class SubNet_main(nn.Module):

    def __init__(self, mode):
        super().__init__()

        if mode=='audio':
            self.conv1_1 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        else:
            self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        # maxpool --> 112 x 112

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # maxpool --> 56 x 56

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # maxpool --> 28 x 28

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # maxpool --> 14 x 14

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # maxpool --> 7 x 7

    def forward(self, x):

        x = torch.max_pool2d(torch.relu(self.conv1_1(x)), 2) # 112 x 112
        x = torch.max_pool2d(torch.relu(self.conv2_1(x)), 2) # 56 x 56

        x = torch.relu(self.conv3_1(x)) # 56 x 56
        x = torch.max_pool2d(torch.relu(self.conv3_2(x)),2) # 28 x 28

        x = torch.relu(self.conv4_1(x)) # 28 x 28
        x = torch.max_pool2d(torch.relu(self.conv4_2(x)),2) # 14 x 14

        x = torch.relu(self.conv5_1(x)) # 14 x 14
        x = torch.relu(self.conv5_2(x)) # 14 x 14

        return x


class SubNet_vid(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(512,128,kernel_size=1)
        self.conv2 = nn.Conv2d(128,128,kernel_size=1)

    def forward(self, x):

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

        return x


class SubNet_aud(nn.Module):
    def __init__(self, cam_size=11):
        super().__init__()

        self.conv1 = nn.Conv2d(512,128,kernel_size=1)
        self.FC1 = nn.Linear(128 + cam_size, 128)

    def forward(self, x, cam_ID=None):

        if cam_ID is None:
            cam_ID = torch.zeros((x.shape[0], 11))

        x = torch.relu(self.conv1(x))

        c, h, w = x.shape[1], x.shape[-2], x.shape[-1]
        x = x.view(-1, c, h*w)
        x = torch.mean(x, dim=-1)

        x = torch.concat((x, cam_ID), dim=-1)
        
        x = torch.relu(self.FC1(x))
        
        return x



class MergeNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.AudioNet = SubNet_main(mode='audio')
        self.VideoNet = SubNet_main(mode='video')

        self.AudioMerge = SubNet_aud()
        self.VideoMerge = SubNet_vid()

        self.conv1 = nn.Conv2d(128, 62, kernel_size=1)

    def forward(self, im, au):

        im = self.VideoNet(im)
        au = self.AudioNet(au)

        im = self.VideoMerge(im)
        au = self.AudioMerge(au).unsqueeze(-1).unsqueeze(-1)

        print(im.shape, au.shape)

        x = im + im*au
        # norm here!

        x = torch.relu(self.conv1(x))

        x = torch.mean(x, dim=1) # collapsing along dimension

        h, w = x.shape[-2], x.shape[-1]
        x_class = torch.mean(x.view(-1, h*w), dim=-1)

        x_class = torch.sigmoid(x_class)

        # GAP for au only.
        return x, x_class




def Trainer(net, epochs, loss_fn, optimiser, train_loader, val_loader):

    device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
    print(f"Training on device {device}.")

    for epoch in range(1, epochs + 1):

        loss_train = 0.0
        net.train()

        for data in train_loader:
            
            BS = data[0].shape[0]
            all_frames = data[-1]

            imgs_pos = all_frames[torch.randint(len(all_frames), (1,1))]
            audio = data[0]

            # create contrastive batch (shift by some n)
            imgs_neg = torch.roll(imgs_pos, int(torch.randint(low=1, high=BS, size=(1,1))), dims=0)

            # TODO: CAM ID

            imgs_pos = imgs_pos.to(device=device)
            imgs_neg = imgs_neg.to(device=device)
            audio = audio.to(device=device)

            out_pos = net(imgs, audio)
            out_neg = net(imgs_neg, audio)

            # print(labels.shape, outputs.shape)
            loss_pos = loss_fn(out_pos, torch.ones((BS, 1)))
            loss_neg = loss_fn(out_neg, torch.zeros((BS, 1)))

            loss = loss_pos + loss_neg

            # l2_lambda = 0.0001
            # l2_norm = sum(p.pow(2.0).sum() for p in net.parameters())**0.5 # L2 reg for all the weights

            # loss = loss + l2_lambda*l2_norm

            optimiser.zero_grad() 
            loss.backward()
            optimiser.step()

            loss_train += loss.item()

        # validations
        loss_val = 0.0
        net.eval()
        with torch.no_grad():
            for imgs, labels in val_loader:

                imgs = imgs.to(device=device)
                labels = labels.to(device=device)
                outputs = net(imgs)

                loss = loss_fn(outputs, labels)
                loss_val += loss.item()
            
                
            if epoch == 1 or epoch % 4 == 0:
                
                dt = datetime.datetime.now()  

                print('{} Epoch {}, Training loss {}, val loss {}'.format(
                    dt, epoch,
                    round(loss_train / len(train_loader), 4),
                    round(loss_val / len(val_loader), 4),
                ))


                path = os.path.join(os.getcwd(), 'checkpoints', 'SepNetViT_nopointconv_ep_' + str(epoch) + '_checkpoint.pt')
                # print(path)
                torch.save({ 
                    'epoch': epoch,
                    'model': net.state_dict(),
                    'optimizer': optimiser.state_dict()}, path)

