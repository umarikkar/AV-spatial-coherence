import datetime
import os
import math

import torch
from tqdm import tqdm

import core.config as conf
from fn.dataset import create_samples


def Trainer(net,
            epochs, 
            loss_fn,
            optimiser, 
            train_loader, 
            val_loader):

    def save_model(epoch, net, optimiser, net_path):

        torch.save({ 
            'epoch': epoch,
            'model': net.state_dict(),
            'optimizer': optimiser.state_dict()}, net_path)
        return

    device = conf.training_param['device']

    for epoch in range(epochs[0], epochs[1] + 1):

        loss_train = 0.0
        loss_perf = 0.0
        loss_l2 = 0.0
        net.train()

        min_val = math.inf
        
        for data in tqdm(train_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', leave=None):
                
            samples = create_samples(data, augment=True, device=device, 
                            hard_negatives=conf.training_param['hard_negatives'])

            if samples is not None:

                scores_raw = net(samples['imgs_all'], samples['audio_all'], samples['cam_all'] )
                loss, loss_perfect = loss_fn(scores_raw, samples['neg_mask'])
                # loss function --->
                l2_lambda = 0.00001
                l2_reg = l2_lambda*sum(p.pow(2.0).sum() for p in net.parameters())**0.5 # L2 reg for all the weights
                loss += l2_reg
                

                optimiser.zero_grad() 
                loss.backward()
                optimiser.step()

                loss_train += loss.item()
                loss_perf += (loss_perfect + l2_reg).item()
                loss_l2 += l2_reg

            else:
                pass
            

        if val_loader is not None:

            # validations -----------------------------------------------------------------------------
            loss_val = 0.0
            net.eval()

            with torch.no_grad():

                for data in val_loader:

                    samples = create_samples(data, augment=False, device=device)

                    if samples is not None:

                        scores_raw = net(samples['imgs_all'], samples['audio_all'], samples['cam_all'] )
                        lossVal, _ = loss_fn(scores_raw, samples['neg_mask'])
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

        l_val = round(loss_val / len_val, 4)

        if l_val < min_val:
            min_val = l_val
            update = True
        else:
            update = False


        verbose_str = '{} Epoch {}, Train loss {}, Val loss {}, Perfect loss {}'.format(
            dt, 
            epoch,
            round(loss_train / len(train_loader), 4),
            l_val,
            round(loss_perf / len(train_loader), 4)
        )

        print(verbose_str, 
        file=open(os.path.join(file_path, 'train_logs.txt'), "a") # results/checkpoints/MultiChannel_sz/train_logs.txt
        )

        print(verbose_str)

        if update:
            save_model(epoch, net, optimiser, net_path)

        elif epoch == 1 or epoch % 4 == 0:
            save_model(epoch, net, optimiser, net_path)


    return


def Evaluator(net, loader, loss_fn,
            epochs=1,
            verbose=True,
            count=None,
            ):

    device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))

    
    net.to(device=device)
    net.eval()

    acc_avg, acc_pos_avg, acc_neg_avg = 0.0, 0.0, 0.0

    for _ in range(1, epochs+1):

        e_total, e_pos, e_neg = 0.0, 0.0, 0.0
        n_total, n_pos, n_neg = 0.0, 0.0, 0.0

        with torch.no_grad():

            c = 0

            for data in tqdm(loader, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', disable=True):
                
                imgs_all, audio_all, cam_all, _, neg_mask = create_samples(data, augment=True, device=device)

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


