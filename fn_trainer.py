import datetime
import os
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import core.config as conf
from torchvision.transforms import transforms


def create_samples(data, device='cpu', augment=False, contrast_vid=False, take_to_device=True, t_image=conf.data_param['t_image']):

    def rotate(l, n): # rotate sequence list
        return l[n:] + l[:n]

    batch_size = data[0].shape[0]

    audio = data[0]
    cam = data[1]
    all_frames = data[2]
    seq_pos = data[3]

    imgs_pos = all_frames.squeeze(1)

    if augment:
        for i in range(batch_size):
            imgs_pos[i] = t_image(imgs_pos[i])

    # create contrastive batch (shift by some n)
    if batch_size > 1:
        roll_idx = random.randint(1, batch_size-1)
    else:
        roll_idx = 1

    imgs_neg = torch.roll(imgs_pos, roll_idx, dims=0)
    cam_neg = torch.roll(cam, roll_idx, dims=0)
    seq_neg = rotate(seq_pos, roll_idx)

    imgs_all = torch.concat((imgs_pos, imgs_neg), dim=0)
    audio_all = torch.concat((audio, audio), dim=0)
    cam_all = torch.concat((cam, cam_neg), dim=0)
    seq_all = seq_pos

    # choosing which indexes to drop for contrastive learning.
    rem_count = 0
    for idx, seq in enumerate(seq_neg):

        seq1 = seq
        seq2 = seq_pos[idx]

        if contrast_vid:
            seq1 = seq1.decode("utf-8")
            seq1 = seq1[:seq1.find('-cam')]

            seq2 = seq2.decode("utf-8")
            seq2 = seq2[:seq2.find('-cam')]

        if seq1 == seq2:
            idx_rem = batch_size + idx - rem_count

            imgs_all = imgs_all[torch.arange(imgs_all.size(0))!=idx_rem] 
            audio_all = audio_all[torch.arange(audio_all.size(0))!=idx_rem] 
            cam_all = cam_all[torch.arange(cam_all.size(0))!=idx_rem] 

            rem_count +=1
        else:
            seq_all.append(seq)
 
    if take_to_device:
        imgs_all = imgs_all.to(device=device)
        audio_all = audio_all.to(device=device)
        cam_all = cam_all.to(device=device)

    return imgs_all, audio_all, cam_all, seq_all, rem_count


def create_labels(BS, rem_count, device='cpu', heatmap=conf.dnn_arch['heatmap']):

    neg_BS = BS - rem_count

    if heatmap:
        labels_pos = torch.ones(BS).to(device=device)
        labels_neg = torch.zeros(neg_BS).to(device=device)
        labels_all = torch.concat((labels_pos, labels_neg), dim=0).to(device=device)
        
    else:
        one = torch.ones((BS,1))
        zer = torch.zeros((neg_BS,1))

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
            
            imgs_all, audio_all, cam_all, seq_all, rem_count = create_samples(data, augment=True, device=device)

            out_all = net(imgs_all, audio_all, cam_all)
            out = out_all[0] if heatmap else out_all

            labels_all = create_labels(BS, rem_count, device=device, heatmap=heatmap)

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

                    imgs_all, audio_all, cam_all, rem_count = create_samples(data, augment=True, device=device)
                    out_all = net(imgs_all, audio_all, cam_all)
                    out = out_all[0] if heatmap else out_all

                    labels_all = create_labels(BS, rem_count, device=device, heatmap=heatmap)

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



def Trainer_AVOL(net, 
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
            
            imgs_all, audio_all, cam_all, rem_count = create_samples(data, augment=True, device=device)

            out, _ = net(imgs_all, audio_all, cam_all)

            labels_all = create_labels(BS, rem_count, device=device, heatmap=True)

            idx1 = torch.zeros_like(out)
            idx1[out >= 0.5] = 1
            idx2 = labels_all

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

                    imgs_all, audio_all, cam_all, rem_count = create_samples(data, augment=False, device=device)
                    out, _ = net(imgs_all, audio_all, cam_all)

                    labels_all = create_labels(BS, rem_count, device=device, heatmap=heatmap)

                    idx1 = torch.zeros_like(out)
                    idx1[out >= 0.5] = 1
                    idx2 = labels_all

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


def Evaluator(net, loader, 
            epochs=1, 
            contrast_vid=False,
            heatmap=conf.dnn_arch['heatmap'],
            verbose=True
            ):

    device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))

    
    net.to(device=device)
    net.eval()
    # torch.autograd.set_detect_anomaly(True)

    acc_avg = 0.0
    acc_pos_avg = 0.0
    acc_neg_avg = 0.0

    for _ in range(1, epochs+1):

        err = 0.0
        err_pos = 0.0
        err_neg = 0.0

        total_n = 0.0
        total_pos = 0.0
        total_neg = 0.0

        with torch.no_grad():

            for data in loader:

                BS = data[0].shape[0] # batch size
                
                imgs_all, audio_all, cam_all, seq_all, rem_count = create_samples(data, augment=False, contrast_vid=contrast_vid, device=device)

                if conf.dnn_arch['AVOL'] or heatmap:

                    labels_all = create_labels(BS, rem_count, device=device, heatmap=True)
                    out, _ = net(imgs_all, audio_all, cam_all)

                    idx1 = torch.zeros_like(out)
                    idx1[out >= 0.5] = 1
                    idx2 = labels_all

                else:

                    labels_all = create_labels(BS, rem_count, device=device, heatmap=False)
                    out = net(imgs_all, audio_all, cam_all)

                    _, idx1 = torch.max(out, dim=-1)
                    _, idx2 = torch.max(labels_all, dim=-1)

                idx1_pos, idx2_pos = idx1[0:BS], idx2[0:BS]
                idx1_neg, idx2_neg = idx1[BS:], idx2[BS:]

                n_pos = len(idx1_pos)
                n_neg = len(idx1_neg)

                err += torch.abs(idx1-idx2).sum()
                err_pos += torch.abs(idx1_pos-idx2_pos).sum()
                err_neg += torch.abs(idx1_neg-idx2_neg).sum()

                total_n += n_pos + n_neg
                total_pos += n_pos
                total_neg += n_neg

                # acc += 100*(1 - torch.abs(idx1-idx2).sum() / len(idx1))
                # acc_pos += 100*(1 - torch.abs(idx1_pos-idx2_pos).sum() / len(idx1_pos))
                # acc_neg += 100*(1 - torch.abs(idx1_neg-idx2_neg).sum() / len(idx1_neg))

        acc = 100*((total_n - err) / total_n)
        acc_pos = 100*((total_pos - err_pos) / total_pos)
        acc_neg = 100*((total_neg - err_neg) / total_neg)
        
        # acc = round(float(acc/len(loader)), 2)
        # acc_pos = round(float(acc_pos/len(loader)), 2)
        # acc_neg = round(float(acc_neg/len(loader)), 2)

        if verbose:
            verbose_str = 'acc {}%'.format(acc)
            print(verbose_str)

        acc_avg += acc
        acc_pos_avg += acc_pos
        acc_neg_avg += acc_neg

    acc_avg = round(float(acc_avg / epochs) , 2)
    acc_pos_avg = round(float(acc_pos_avg / epochs) , 2)
    acc_neg_avg = round(float(acc_neg_avg / epochs) , 2)

    return acc_avg, acc_pos_avg, acc_neg_avg


