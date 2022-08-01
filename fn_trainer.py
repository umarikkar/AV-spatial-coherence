import datetime
import os
import random
from venv import create
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import core.config as conf
from torchvision.transforms import transforms

from einops import rearrange, reduce, repeat


def create_samples(data, 
    device=conf.training_param['device'], 
    augment=False, 
    contrast_vid=conf.training_param['vid_contrast'], 
    t_image=conf.data_param['t_image'], 
    create_contrast=True):

    def rotate(l, n): # rotate sequence list
        return l[n:] + l[:n]

    batch_size = data[0].shape[0]

    audio = data[0]
    cam = data[1]
    all_frames = data[2]
    seq_pos = data[3]

    speaking = data[5] if conf.training_param['inference'] else data[6]


    if conf.training_param['frame_seq']:
        imgs_pos = all_frames
    else:
        imgs_pos = all_frames.squeeze(1)

    if augment:
        for i in range(batch_size):
            imgs_pos[i] = t_image(imgs_pos[i])
    
    
    if create_contrast:
        rem_count = 0
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
        for idx, seq in enumerate(seq_neg):

            seq1 = seq
            seq2 = seq_pos[idx]

            if speaking[idx] == b'NOT_SPEAKING':

                idx_rem = idx

                imgs_all = imgs_all[torch.arange(imgs_all.size(0))!=idx_rem] 
                audio_all = audio_all[torch.arange(audio_all.size(0))!=idx_rem] 
                cam_all = cam_all[torch.arange(cam_all.size(0))!=idx_rem]

                rem_count +=1

            else:
                if contrast_vid:
                    seq1 = seq1.decode("utf-8")
                    seq1 = seq1[:seq1.find('_t')]

                    seq2 = seq2.decode("utf-8")
                    seq2 = seq2[:seq2.find('_t')]


                if seq1 == seq2:
                    idx_rem = batch_size + idx - rem_count

                    imgs_all = imgs_all[torch.arange(imgs_all.size(0))!=idx_rem] 
                    audio_all = audio_all[torch.arange(audio_all.size(0))!=idx_rem] 
                    cam_all = cam_all[torch.arange(cam_all.size(0))!=idx_rem] 

                    rem_count +=1
                else:
                    seq_all.append(seq)

    else:
        cam_all = cam
        imgs_all = imgs_pos
        audio_all = audio
        seq_all = data[3]

        rem_count = torch.ones((batch_size, batch_size)).to(device=device)

        for idx1, seq1 in enumerate(seq_all):

            if speaking[idx1] == b'NOT_SPEAKING':

                rem_count[idx1, :] = 0
                rem_count[:, idx1] = 0

            else:
                for idx2, seq2 in enumerate(seq_all):

                    if idx1 != idx2:
                        if seq1 == seq2:
                            rem_count[idx1, idx2] = 0


    imgs_all = imgs_all.to(device=device)
    audio_all = audio_all.to(device=device)
    cam_all = cam_all.to(device=device)

    return imgs_all, audio_all, cam_all, seq_all, rem_count


def create_labels(BS, rem_count, device='cpu', heatmap=conf.dnn_arch['heatmap']):

    neg_BS = BS - rem_count

    if heatmap or conf.dnn_arch['AVOL']:
        labels_pos = torch.ones(BS).to(device=device)
        labels_neg = torch.zeros(neg_BS).to(device=device)
        labels_all = torch.concat((labels_pos, labels_neg), dim=0).to(device=device)
        
    else:
        one = torch.ones((BS,1))
        zer = torch.zeros((neg_BS,1))

        labels_pos = torch.concat((one, zer), dim=0)
        labels_neg = (-1.0)*labels_pos + 1.0

        labels_neg = torch.concat((zer, one), dim=0)
        labels_all = torch.concat((labels_pos, labels_neg), dim=-1).to(device=device)

    return labels_all


class loss_VSL(nn.Module):

    def __init__(self, tau=0.8, eps=1e-8, alpha=0.5, beta=0.5):
        super().__init__()
        self.tau=tau
        self.eps=eps
        self.alpha=alpha
        self.beta=beta

    def forward(self, scores_raw, neg_mask):

        scores = torch.exp(scores_raw/self.tau)*neg_mask

        s_pos = scores.diag()
        s_neg_a = scores.sum(dim=1)
        s_neg_v = scores.sum(dim=0)

        L_av = -1 * torch.log10(s_pos / (s_neg_a + self.eps))
        L_va = -1 * torch.log10(s_pos / (s_neg_v + self.eps))

        loss = (self.alpha*L_av + self.beta*L_va).sum() / (neg_mask.sum() + self.eps)

        return loss


class loss_AVOL(nn.Module):

    def __init__(self, eps=1e-8):
        super().__init__()
        self.loss_fn = nn.BCELoss(reduction='none')
        self.eps=eps

    def forward(self, scores_raw, neg_mask, device=conf.training_param['device'], return_acc=False):

        bs = scores_raw.shape[0]

        labels = torch.eye(len(scores_raw)).to(device=device).view(-1,1)
        scores = scores_raw.view(-1,1)

        loss = self.loss_fn(scores, labels).view(bs,bs)
        loss = loss*neg_mask
        loss = (torch.ones((bs,bs)) + (bs-1)*torch.eye(bs)).to(device=device)*loss

        loss = loss.sum() / (neg_mask.sum() + self.eps)

        return loss


class loss_AVE(nn.Module):

    def __init__(self, eps=1e-8):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.eps=eps

    def forward(self, scores_raw, neg_mask, device=conf.training_param['device'], return_err = False):

        bs = scores_raw.shape[0]

        labels_pos = torch.eye(bs).unsqueeze(-1)
        labels_neg = (-1*torch.eye(bs) + 1).unsqueeze(-1)

        labels = torch.concat((labels_pos, labels_neg), dim=-1).to(device=device)

        scores = rearrange(scores_raw, 'a1 a2 c -> (a1 a2) c')
        labels = rearrange(labels, 'a1 a2 c -> (a1 a2) c')

        if not return_err:

            loss = rearrange(self.loss_fn(scores, labels), '(a1 a2) -> a1 a2', a1=bs)
            loss = loss*neg_mask
            loss = (torch.ones((bs,bs)) + (bs-1)*torch.eye(bs)).to(device=device)*loss

            loss = loss.sum() / (neg_mask.sum() + self.eps)

            return loss

        else:

            c_pos, c_neg, n_pos, n_neg, c_total, n_total = 0, 0, 0, 0, 0, 0

            _, idx1 = torch.max(scores, dim=-1)
            _, idx2 = torch.max(labels, dim=-1)

            idx1 = rearrange(idx1, '(a1 a2) -> a1 a2', a1=bs)
            idx2 = rearrange(idx2, '(a1 a2) -> a1 a2', a1=bs)

            err_idxs = torch.abs(idx1-idx2)

            n_total = neg_mask.sum()
            
            e_total = (err_idxs*neg_mask).sum()

            return e_total, n_total


def Trainer_binary(net, 
            epochs, 
            loss_fn, 
            optimiser, 
            train_loader, 
            val_loader, 
            heatmap=conf.dnn_arch['heatmap']):

    device = conf.training_param['device']

    for epoch in range(epochs[0], epochs[1] + 1):

        loss_train = 0.0
        acc_train = 0.0
        loss_l2 = 0.0

        net.train()
        # torch.autograd.set_detect_anomaly(True)

        for data in train_loader:

            BS = data[0].shape[0] # batch size

            if BS != 1:
            
                imgs_all, audio_all, cam_all, _, rem_count = create_samples(data, augment=True, device=device)

                out_all = net(imgs_all, audio_all, cam_all, BS, inference=False)

                labels_all = create_labels(BS, rem_count, device=device, heatmap=heatmap)

                if heatmap or conf.dnn_arch['AVOL']:
                    out = out_all[0]
                    idx1 = torch.zeros_like(out)
                    idx1[out >= 0.5] = 1
                    idx2 = labels_all

                else: # outputs with FC out as final layer. Learns a bit faster         
                    out = out_all       
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

            else:
                pass
            

        if val_loader is not None:

            # validations -----------------------------------------------------------------------------
            loss_val = 0.0
            acc_val = 0.0
            net.eval()

            with torch.no_grad():

                for data in val_loader:

                    BS = data[0].shape[0]

                    imgs_all, audio_all, cam_all, _, rem_count = create_samples(data, augment=True, device=device)
                    out_all = net(imgs_all, audio_all, cam_all)

                    labels_all = create_labels(BS, rem_count, device=device, heatmap=heatmap)

                    if heatmap or conf.dnn_arch['AVOL']:
                        out = out_all[0]
                        idx1 = torch.zeros_like(out)
                        idx1[out >= 0.5] = 1
                        idx2 = labels_all

                    else: # outputs with FC out as final layer. Learns a bit faster 
                        out = out_all               
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


def Trainer_contrast(net,
            epochs, 
            loss_fn,
            optimiser, 
            train_loader, 
            val_loader,
            binary_samples = conf.training_param['train_binary']):

    device = conf.training_param['device']

    for epoch in range(epochs[0], epochs[1] + 1):

        loss_train = 0.0
        loss_l2 = 0.0
        net.train()

        for data in train_loader:

            BS = data[0].shape[0] # batch size

            if BS != 1:
            
                imgs_all, audio_all, cam_all, _, neg_mask = create_samples(data, augment=True, device=device, create_contrast=binary_samples)
                scores_raw = net(imgs_all, audio_all, cam_all)

                loss = loss_fn(scores_raw, neg_mask)
                # loss function --->
                l2_lambda = 0.0001
                l2_reg = l2_lambda*sum(p.pow(2.0).sum() for p in net.parameters())**0.5 # L2 reg for all the weights
                loss += l2_reg

                optimiser.zero_grad() 
                loss.backward()
                optimiser.step()

                loss_train += loss.item()
                loss_l2 += l2_reg

            else:
                pass
            

        if val_loader is not None:

            # validations -----------------------------------------------------------------------------
            loss_val = 0.0
            net.eval()

            with torch.no_grad():

                for data in val_loader:

                    BS = data[0].shape[0]

                    if BS != 1:

                        imgs_all, audio_all, cam_all, _, neg_mask = create_samples(data, augment=False, device=device, create_contrast=binary_samples)
                        scores_raw = net(imgs_all, audio_all, cam_all)
                        
                        lossVal = loss_fn(scores_raw, neg_mask)
                        loss_val += lossVal.item()

                    else:
                        pass

            len_val = len(val_loader)

        else:
            loss_val = 0.0
            len_val = 1

        # verbose
        dt = datetime.datetime.now()  

        net_name = 'net_ep_' + str(epoch) # net_ep_X
        file_path = conf.filenames['net_folder_path']
        net_path = os.path.join(file_path, net_name + '.pt') # results/checkpoints/MultiChannel_sz/net_ep_X.pt

        verbose_str = '{} Epoch {}, Train loss {}, Val loss {}'.format(
            dt, 
            epoch,
            round(loss_train / len(train_loader), 4),
            round(loss_val / len_val, 4),
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


def Eval(net, loader, 
            epochs=1, 
            contrast_vid=False,
            heatmap=conf.dnn_arch['heatmap'],
            verbose=True,
            count=None,
            inference=conf.training_param['inference']
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

            c = 0

            for data in loader:

                BS = data[0].shape[0] # batch size
                
                imgs_all, audio_all, cam_all, _, rem_count = create_samples(data, augment=False, contrast_vid=contrast_vid, device=device)


                if conf.dnn_arch['AVOL'] or heatmap:

                    labels_all = create_labels(BS, rem_count, device=device, heatmap=True)
                    out, _ = net(imgs_all, audio_all, cam_all, BS, inference=inference)

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

                c += 1
                if count != None and c == count:
                    break

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


def Eval_contrast(net, loader, loss_fn,
            epochs=1, 
            contrast_vid=False,
            heatmap=conf.dnn_arch['heatmap'],
            verbose=True,
            count=None,
            inference=conf.training_param['inference'],
            binary_samples = conf.training_param['train_binary']
            ):

    device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))

    
    net.to(device=device)
    net.eval()

    acc_avg = 0.0
    acc_pos_avg = 0.0
    acc_neg_avg = 0.0

    for _ in range(1, epochs+1):

        e_total = 0.0
        err_pos = 0.0
        err_neg = 0.0

        n_total = 0.0
        total_pos = 0.0
        total_neg = 0.0

        with torch.no_grad():

            c = 0

            for data in loader:

                BS = data[0].shape[0] # batch size
                
                imgs_all, audio_all, cam_all, _, neg_mask = create_samples(data, augment=True, device=device, create_contrast=binary_samples)

                scores_raw = net(imgs_all, audio_all, cam_all)

                e_batch, n_batch = loss_fn(scores_raw, neg_mask, return_err=True)

                #     _, idx1 = torch.max(out, dim=-1)
                #     _, idx2 = torch.max(labels_all, dim=-1)

                # idx1_pos, idx2_pos = idx1[0:BS], idx2[0:BS]
                # idx1_neg, idx2_neg = idx1[BS:], idx2[BS:]

                # n_pos = len(idx1_pos)
                # n_neg = len(idx1_neg)

                e_total += e_batch
                n_total += n_batch

                # err += torch.abs(idx1-idx2).sum()
                # err_pos += torch.abs(idx1_pos-idx2_pos).sum()
                # err_neg += torch.abs(idx1_neg-idx2_neg).sum()

                # total_n += n_pos + n_neg
                # total_pos += n_pos
                # total_neg += n_neg

                c += 1
                if count != None and c == count:
                    break


        acc = 100*((n_total - e_total) / n_total)
        # acc_pos = 100*((total_pos - err_pos) / total_pos)
        # acc_neg = 100*((total_neg - err_neg) / total_neg)


        if verbose:
            verbose_str = 'acc {}%'.format(acc)
            print(verbose_str)

        acc_avg += acc
        # acc_pos_avg += acc_pos
        # acc_neg_avg += acc_neg

    acc_avg = round(float(acc_avg / epochs) , 2)
    # acc_pos_avg = round(float(acc_pos_avg / epochs) , 2)
    # acc_neg_avg = round(float(acc_neg_avg / epochs) , 2)

    return acc_avg
    # acc_pos_avg, acc_neg_avg


