import datetime
import os
import math

import torch
from tqdm import tqdm

import core.config as conf
from fn.dataset import create_samples, get_train_val

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from core.utils import n_fold_generator

def Trainer(net,
            epochs, 
            loss_fn,
            optimiser,
            dataset,
            bs = conf.training_param['batch_size']):

    def save_model(epoch, net, optimiser, net_path):

        torch.save({ 
            'epoch': epoch,
            'model': net.state_dict(),
            'optimizer': optimiser.state_dict()}, net_path)
        return

    device = conf.training_param['device']

    k,ls = 6, []
    for _ in range(k):
        ls.append(int(len(dataset)/k))
    ls[-1] = len(dataset) - sum(ls[:-1])

    data_all = torch.utils.data.random_split(dataset, ls)

    for epoch in range(epochs[0], epochs[1] + 1):

        val_idx = epoch%k

        data_val = data_all[val_idx]
        data_train = [x for i,x in enumerate(data_all) if i!=val_idx]

        data_train = torch.utils.data.ConcatDataset(data_train)

        train_loader = DataLoader(data_train, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(data_val, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True)

        if conf.contrast_param['flip_img'] or conf.contrast_param['flip_mic']:
            alpha = conf.contrast_param['alpha']

        loss_train = 0.0
        loss_ez = 0.0
        loss_hd = 0.0
        loss_perf = 0.0
        loss_l2 = 0.0
        net.train()

        min_val = math.inf
        
        for data in tqdm(train_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', leave=None):
            
            # plt.figure()
            # for id, im in enumerate(data[0][0]):
            #     # print()
            #     plt.subplot(4,4,id+1)
            #     plt.imshow(im, aspect='auto')
            #     plt.colorbar()
            #     plt.title(id)
            # plt.show()

            samples = create_samples(data, augment=True, device=device)

            if samples is not None:

                scores_raw = net(samples['imgs_all'], samples['audio_all'], samples['cam_all'] )
                loss_dict, loss_perfect = loss_fn(scores_raw, samples['neg_mask'])

                # a = (scores_raw*samples['neg_mask']).cpu().detach().numpy()
                # import matplotlib.pyplot as plt
                # plt.figure()
                # plt.imshow(a)
                # plt.colorbar()
                # plt.show()

                loss = loss_dict['total']
                # loss function --->
                l2_lambda = 0.00001
                l2_reg = l2_lambda*sum(p.pow(2.0).sum() for p in net.parameters())**0.5 # L2 reg for all the weights
                loss += l2_reg
                

                optimiser.zero_grad() 
                loss.backward()
                optimiser.step()

                loss_train += loss.item()
                loss_perf += (loss_perfect['total'] + l2_reg).item()
                loss_l2 += l2_reg

                if conf.contrast_param['flip_img'] or conf.contrast_param['flip_mic']:
                    loss_ez = loss_ez + loss_dict['easy'].item() if loss_dict['easy'] is not None else 0.0
                    loss_hd = loss_hd + loss_dict['hard'].item() if loss_dict['hard'] is not None else 0.0

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

                        lossVal = lossVal['total']

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


        verbose_str = '{} Epoch {}, Train loss {}, '.format(
            dt, 
            epoch,
            round(loss_train / len(train_loader), 4),
        )

        if conf.contrast_param['flip_img'] or conf.contrast_param['flip_mic']:
            verbose_str += ', loss_ez: {}, loss_hd {}'.format(round(loss_ez / len(train_loader), 4), round(loss_hd / len(train_loader), 4),)

        verbose_str += ', Val loss {}, Perfect loss {}'.format(
            l_val,
            round(loss_perf / len(train_loader), 4))

        print(verbose_str, 
        file=open(os.path.join(file_path, 'train_logs.txt'), "a") # results/checkpoints/MultiChannel_sz/train_logs.txt
        )

        print(verbose_str)

        # if update:
        #     save_model(epoch, net, optimiser, net_path)

        if epoch == 1 or epoch % 4 == 0:
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


def plot_results(net, ep, loader=None, seq='all', num_plots=5, device=conf.training_param['device']
):

    net.to(device=device)
    net.eval()

    if loader is None:
        _ , _ , data_all = get_train_val(train_or_test='test', sequences=seq)
        loader = DataLoader(data_all, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    
    plt.figure(figsize=(3*num_plots, 6))
    count=0

    for _, data in enumerate(loader):

        if count < num_plots:

            if data[6] == [b'SPEAKING']:

                samples = create_samples(data, augment=False, device=device, return_mask=False, train_or_test='test')
                imgs_all = samples['imgs_all']
                _, heatmap = net(samples['imgs_all'], samples['audio_all'], samples['cam_all'], flip_img=False)


                heatmap = heatmap.cpu().squeeze(0).detach().numpy()
                imgs_all = imgs_all.cpu().squeeze(0).permute(1,2,0).detach().numpy()

                romeo = samples['meta_male'][0]
                juliet = samples['meta_female'][0]

                R = True if romeo['activity']=='SPEAKING' else False
                J = True if juliet['activity']=='SPEAKING' else False

                R_coord = np.array([romeo['x'], romeo['y']])*heatmap.shape[-1]/224
                J_coord = np.array([juliet['x'], juliet['y']])*heatmap.shape[-1]/224

                plt.subplot(2, num_plots, count+1)
                plt.imshow(imgs_all, aspect='equal')
                plt.axis('off')

                plt.subplot(2, num_plots, num_plots+count+1)
                plt.imshow(heatmap, vmin=0, vmax=1, aspect='equal')

                if R:
                    plt.scatter(R_coord[0], R_coord[1], s=100, color="r")
                if J:
                    plt.scatter(J_coord[0], J_coord[1], s=100, color="r")

                if R and J:
                    speaker = 'both'
                    c = None
                elif R:
                    speaker = 'Romeo'
                    c = R_coord
                elif J:
                    speaker = 'Juliet'
                    c = J_coord
                else:
                    speaker = 'none'
                    c = None

                c_max = np.flip(np.array(np.unravel_index(heatmap.argmax(), heatmap.shape)))                 # i=y, j=x

                acc = 100*(heatmap.shape[0] - np.abs(c-c_max)) / heatmap.shape[0]

                acc_x, acc_y = acc[0], acc[1]

                plt.title('%s, GT:(%.1f,%.1f)'%(speaker, c[0], c[1]))
                plt.axis('off')

                txt = "acc_x:%.1f, acc_y:%.1f"%(acc_x, acc_y)

                plt.text(0, heatmap.shape[0]+0.5, txt, fontsize=12)

                count += 1

        else:

            plt.suptitle('epochs: '+str(ep))

            fig_path = os.path.join(conf.filenames['net_folder_path'], conf.filenames['train_val'])
            img_name = 'ep_%s_%s_1'%(ep,seq)+'.png'

            while img_name in os.listdir(fig_path):

                num = int(img_name[img_name.find('.png')-1]) + 1

                img_name = img_name[:img_name.find('.png')-1] + str(num) + '.png'

            plt.savefig(os.path.join(fig_path, img_name))
            plt.close()
            plt.show()

            break

    return


def eval_loc(net, ep, loader=None, seq='all', device=conf.training_param['device'], tolerance=2.0, verbose=True):
    net.to(device=device)
    net.eval()

    eval_name = 'eval_results_%s.txt'%seq
    save_file = os.path.join(conf.filenames['net_folder_path'], eval_name)

    if loader is None:
        _ , _ , data_all = get_train_val(train_or_test='test', sequences=seq)
        loader = DataLoader(data_all, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)

    n_sum = 0
    err_x = 0.0
    err_y = 0.0


    for n, data in enumerate(loader):

        samples = create_samples(data, augment=False, device=device, return_mat=False, train_or_test='test')

        if samples is not None:
            scores, heatmap = net(samples['imgs_all'], samples['audio_all'], samples['cam_all'], flip_img=False)


            heatmap = heatmap.cpu().detach().numpy()

            romeo = samples['meta_male']
            juliet = samples['meta_female']

            c_GT = np.zeros((len(romeo), 2))
            c_pred = np.zeros((len(romeo), 2))

            spk = []

            sz = heatmap.shape[-1]

            for idx, tup in enumerate(zip(romeo, juliet)):
                if romeo[idx]['activity']=='SPEAKING':
                    spk.append('Romeo')
                    tup_ = tup[0]  
                elif juliet[idx]['activity']=='SPEAKING':
                    spk.append('Juliet')
                    tup_ = tup[1]
                    
                c_GT[idx,0], c_GT[idx,1] = tup_['x']*sz/224, tup_['y']*sz/224

                c_pred[idx,:] = np.flip(np.array(np.unravel_index(heatmap[idx].argmax(), heatmap[idx].shape)))

                e_x = c_GT[:,0] - c_pred[:,0]
                e_y = c_GT[:,1] - c_pred[:,1]


            if n==0:
                e2_x, e2_y = e_x, e_y
            else:
                e2_x = np.concatenate((e2_x, e_x))
                e2_y = np.concatenate((e2_y, e_y))

    ex_bias, ey_bias = round(e2_x.mean(), 2), round(e2_y.mean(), 2)
    ex_corr = e2_x - ex_bias
    ey_corr = e2_y - ey_bias
    x_corrected = round(abs(ex_corr).mean(), 2)
    y_corrected = round(abs(ey_corr).mean(), 2)

    acc_dist = {
        'x' : round(abs(e2_x).mean(), 2),
        'y' : round(abs(e2_y).mean(), 2),
        'x_corrected': x_corrected,
        'y_corrected': y_corrected,
    }

    verbose_str = "epochs: %d, err_dist: x:%.2f, y:%.2f, bias corrected: x:%.2f, y:%.2f"%(ep, acc_dist['x'], 
                acc_dist['y'], acc_dist['x_corrected'], acc_dist['y_corrected'])
    print(verbose_str, file=open(save_file, "a"))

    if verbose:
        print(verbose_str)

    acc_box = []
    for tol in tolerance:
        t = tol * heatmap.shape[0] / 100

        a_x = round(100*sum(abs(e2_x) <= t) / len(e2_x), 2)
        a_y = round(100*sum(abs(e2_y) <= t) / len(e2_y), 2)

        a_x_corr = round(100*sum(abs(ex_corr) <= t) / len(ex_corr), 2)
        a_y_corr = round(100*sum(abs(ey_corr) <= t) / len(ey_corr), 2)

        acc_box.append({
            'tolerance':t,
            'x' : a_x,
            'y' : a_y,
            'x_corrected' : a_x_corr,
            'y_corrected' : a_y_corr
                    })
  
        verbose_str = "tolerance: %d, acc_box: x:%.2f, y:%.2f, x_corr:%.2f, y_corr:%.2f"%(tol, a_x, a_y, a_x_corr, a_y_corr)
        print(verbose_str, file=open(save_file, "a"))

        if verbose:
            print(verbose_str)

    print('\n\n', file=open(save_file, "a"))
    print('\n')

    return acc_dist, acc_box

    
