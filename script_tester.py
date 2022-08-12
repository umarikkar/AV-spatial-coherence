
import os
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import core.config as conf
from fn.dataset import get_train_val, create_samples
from fn.nets import set_network
from fn.train_eval import Evaluator

import matplotlib.pyplot as plt

from tqdm import tqdm


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

                samples = create_samples(data, augment=False, device=device, return_mat=False, hard_negatives=False, train_or_test='test')
                imgs_all = samples['imgs_all']
                _, heatmap = net(samples['imgs_all'], samples['audio_all'], samples['cam_all'], hard_negatives=False)
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
            img_name = '%s_ep_%s_1'%(seq,ep)+'.png'

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

        samples = create_samples(data, augment=False, device=device, return_mat=False, hard_negatives=False, train_or_test='test')

        if samples is not None:
            _, heatmap = net(samples['imgs_all'], samples['audio_all'], samples['cam_all'], hard_negatives=False)

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
    for t in tolerance:
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
  
        verbose_str = "tolerance: %d, acc_box: x:%.2f, y:%.2f, x_corr:%.2f, y_corr:%.2f"%(t, a_x, a_y, a_x_corr, a_y_corr)
        print(verbose_str, file=open(save_file, "a"))

        if verbose:
            print(verbose_str)

    print('\n\n', file=open(save_file, "a"))
    print('\n')

    return acc_dist, acc_box

    

def main():

    random.seed(5)

    # loading network ----------------------------------------

    net, _ = set_network(set_train=False)

    # seq_list = ['all']
    seq_list = ['all', 'conversation1_t3', 'femalemonologue2_t3', 'interactive1_t2', 'interactive4_t3', 'malemonologue2_t3']
    # seq_list = ['femalemonologue2_t3']

    plot_res= False

    bs = 1 if plot_res else 16

    for seq in seq_list:

        _ , _ , data_all = get_train_val(train_or_test='test', sequences=seq)

        loader = DataLoader(data_all, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True)

        fol_name = conf.filenames['net_folder_path']

        # epoch_settings = np.linspace(4,16,4).astype('int')

        epoch_settings = np.linspace(4, 40, 10).astype('int')

        # epoch_settings = [20]

        tolerance_list = [1,2,3,4,5,6]

        for ep in epoch_settings:

            net_name = 'net_ep_%s.pt'%ep
            net_path = os.path.join(fol_name, net_name)

            checkpoint = torch.load(net_path)

            net.load_state_dict(checkpoint['model'])

            if plot_res:
                plot_results(net, ep, loader=loader, seq=seq)
            else:
                eval_loc(net, ep, loader=loader, seq=seq, tolerance=tolerance_list, verbose=True)
            




    # loading dataset (QUANTITATIVE RESULTS)--------------------------------------

    """
    
    # # seq_list = ['all']

    # seq_compare = True if len(seq_list) > 1 else False

    # if seq_list[0] == 'all':
    #     seq_type = 'all'
    # else:
    #     seq_type = 'seq'
    
    # contrast_vid_setting = [False]

    # for contrast_vid in contrast_vid_setting:

    #     if contrast_vid:
    #         eval_name = 'eval_results_%s_vid.txt'%seq_type
    #     else:
    #         eval_name = 'eval_results_%s.txt'%seq_type

    #     save_file = os.path.join(conf.filenames['net_folder_path'], eval_name)

    #     print('save file:', save_file)

    #     acc_mat = np.zeros((len(seq_list), len(seq_list)))
    #     acc_pos_mat = np.zeros((len(seq_list), len(seq_list)))
    #     acc_neg_mat = np.zeros((len(seq_list), len(seq_list)))

    #     for i, seq1 in enumerate(seq_list):

    #         for j, seq2 in enumerate(seq_list):

    #             # if i==j:

    #             if j >= i:   
    #                 if seq1 != seq2:
    #                     seq = [seq1, seq2]
    #                 else:
    #                     seq = seq1

    #                 _ , _ , data_all = get_train_val(train_or_test='test', sequences=seq)

    #                 loader = DataLoader(data_all, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)

    #                 acc, acc_pos, acc_neg = Evaluator(net, loader=loader, loss_fn=loss_fn,  epochs=1, contrast_vid=contrast_vid, verbose=False)

    #                 acc_mat[i,j] = acc
    #                 acc_pos_mat[i,j] = acc_pos
    #                 acc_neg_mat[i,j] = acc_neg

    #                 print_txt = 'for sequence: {}, size: {}, acc: {}%, acc_pos: {}%, acc_neg: {}%'.format(seq, len(data_all), acc, acc_pos, acc_neg)

    #                 print(print_txt, file=open(save_file, "a"))
    #                 print(print_txt)
                

    #     print_txt = '\ninter-video accuracy matrix:\nepochs: {} \n\nacc_total: \n{} \n\nacc_pos: \n{} \n\nacc_neg: \n{}\n\n.'.format(ep, acc_mat, acc_pos_mat, acc_neg_mat)
    #     print(print_txt, file=open(save_file, "a"))
    #     print(print_txt)
    
    """






if __name__=='__main__':

    main()
