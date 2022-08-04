import datetime
import os

import torch
from tqdm import tqdm

import core.config as conf
from core.dataset import create_labels, create_samples


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
        
        for data in tqdm(train_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):

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
        e_pos = 0.0
        e_neg = 0.0

        n_total = 0.0
        n_pos = 0.0
        n_neg = 0.0

        with torch.no_grad():

            c = 0

            for data in tqdm(loader, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
                
                imgs_all, audio_all, cam_all, _, neg_mask = create_samples(data, augment=True, device=device, create_contrast=binary_samples)

                scores_raw = net(imgs_all, audio_all, cam_all)

                errors = loss_fn(scores_raw, neg_mask, return_err=True)

                e_total += errors['e_total']
                n_total += errors['n_total']

                e_pos += errors['e_pos']
                n_pos += errors['n_pos']

                e_neg += errors['e_neg']
                n_neg += errors['n_neg']

                c += 1
                if count != None and c == count:
                    break


        acc = 100*((n_total - e_total) / n_total)
        acc_pos = 100*((n_pos - e_pos) / n_pos)
        acc_neg = 100*((n_neg - e_neg) / n_neg)


        if verbose:
            verbose_str = 'acc: {}%, acc_pos: {}%, acc_neg: {}%'.format(acc, acc_pos, acc_neg)
            print(verbose_str)

        acc_avg += acc
        acc_pos_avg += acc_pos
        acc_neg_avg += acc_neg

    acc_avg = round(float(acc_avg / epochs) , 2)
    acc_pos_avg = round(float(acc_pos_avg / epochs) , 2)
    acc_neg_avg = round(float(acc_neg_avg / epochs) , 2)

    return acc_avg, acc_pos_avg, acc_neg_avg


