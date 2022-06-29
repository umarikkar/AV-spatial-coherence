import datetime
import os
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import core.config as conf
from torchvision.transforms import transforms


def create_samples(data, device='cpu', augment=False, take_to_device=True):

    batch_size = data[0].shape[0]
    all_frames = data[-1]
    cam = data[1]

    imgs_pos = all_frames.squeeze(1)

    if augment:
        for i in range(batch_size):
            imgs_pos[i] = transforms.ColorJitter((0.8, 1.2), (0.8, 1.2), (0.8, 1.2), (-0.1, 0.1))(imgs_pos[i])
            imgs_pos[i] = transforms.RandomGrayscale(0.2)(imgs_pos[i])
            if torch.rand(1) > 0.5:
                imgs_pos[i] = 1 - imgs_pos[i]

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



def Trainer(net, 
            epochs, 
            loss_fn, 
            optimiser, 
            train_loader, 
            val_loader, 
            heatmap=conf.dnn_arch['heatmap']):

    device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
    print(f"Training on device {device}.")


    for epoch in range(epochs[0], epochs[1] + 1):

        loss_train = 0.0
        acc_train = 0.0
        loss_l2 = 0.0

        net.train()
        # torch.autograd.set_detect_anomaly(True)

        for data in train_loader:

            BS = data[0].shape[0] # batch size
            
            imgs_all, audio_all, cam_all = create_samples(data, augment=True, device=device)

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
            l2_lambda = 0.0001
            l2_reg = l2_lambda*sum(p.pow(2.0).sum() for p in net.parameters())**0.5 # L2 reg for all the weights
            loss += l2_reg

            optimiser.zero_grad() 
            loss.backward()
            optimiser.step()

            loss_train += loss.item()
            loss_l2 += l2_reg

        if val_loader is not None:

            # validations -----------------------------------------------------------------------------
            loss_val = 0.0
            acc_val = 0.0
            net.eval()

            with torch.no_grad():

                for data in val_loader:

                    BS = data[0].shape[0]

                    imgs_all, audio_all, cam_all = create_samples(data, augment=False, device=device)
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

            len_val = len(val_loader)

        else:
            loss_val = 0.0
            acc_val = 0.0

            len_val = 1

        # verbose
        dt = datetime.datetime.now()  

        net_name = 'net_ep_' + str(epoch) # net_ep_X

        file_path = conf.filenames['net_folder_path']

        net_path = os.path.join(file_path, net_name + '.pt') # results/checkpoints/MultiChannel_sz/net_ep_X.pt

        verbose_str = '{} Epoch {}, Train loss {}, Train acc {}%, Val loss {}, Val acc {}%'.format(
            dt, 
            epoch,
            round(loss_train / len(train_loader), 4),
            round(float(acc_train/len(train_loader)), 2),
            round(loss_val / len_val, 4),
            round(float(acc_val/len_val), 2),
        )

        print(verbose_str, 
        file=open(os.path.join(file_path, 'train_logs.txt'), "a") # results/checkpoints/MultiChannel_sz/train_logs.txt
        )

        print(verbose_str)
                
        if epoch == 1 or epoch % 4 == 0:
            
            torch.save({ 
                'epoch': epoch,
                'model': net.state_dict(),
                'optimizer': optimiser.state_dict()}, net_path)


    return

