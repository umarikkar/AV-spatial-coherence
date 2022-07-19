
import os
import random
from math import ceil, floor
from multiprocessing.dummy import freeze_support
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import core.config as conf
import utils.utils as utils
from core.dataset import get_train_val
from fn_networks import MergeNet, AVOL_Net, AVE_Net
from fn_trainer import Trainer, Evaluator


def main():

    random.seed(5)
    inference=False

    # loading network ----------------------------------------

    if conf.dnn_arch['AVOL']:
        net = AVOL_Net()
        loss_fn = nn.BCELoss()
    elif conf.dnn_arch['AVE']:
        net = AVE_Net()
        loss_fn = nn.CrossEntropyLoss()
    else:
        net = MergeNet()
        loss_fn = nn.BCELoss() if conf.dnn_arch['heatmap'] else nn.CrossEntropyLoss()

    optimiser = optim.Adam(net.parameters(), lr=1e-4)

    fol_name = conf.filenames['net_folder_path']

    print(fol_name)
    ep = 16

    net_name = 'net_ep_%s.pt'%ep
    net_path = os.path.join(fol_name, net_name)

    checkpoint = torch.load(net_path)

    net.load_state_dict(checkpoint['model'])
    optimiser.load_state_dict(checkpoint['optimizer'])

    # %% 
    # loading dataset (QUANTITATIVE RESULTS)--------------------------------------


    seq_list = ['conversation1_t3', 'femalemonologue2_t3', 'interactive1_t2', 'interactive4_t3', 'malemonologue2_t3']
    # seq_list = ['all']

    seq_compare = True if len(seq_list) > 1 else False

    seq_type = 'all' if seq_list[0]=='all' else 'seq'
    contrast_vid_setting = [False, True]
    # contrast_vid_setting = [False]

    for contrast_vid in contrast_vid_setting:

        if contrast_vid:
            eval_name = 'eval_results_%s_vid.txt'%seq_type
        else:
            eval_name = 'eval_results_%s.txt'%seq_type

        save_file = os.path.join(conf.filenames['net_folder_path'], eval_name)

        print(save_file)

        # if seq_compare:
        acc_mat = np.zeros((len(seq_list), len(seq_list)))
        acc_pos_mat = np.zeros((len(seq_list), len(seq_list)))
        acc_neg_mat = np.zeros((len(seq_list), len(seq_list)))

        for i, seq1 in enumerate(seq_list):

            for j, seq2 in enumerate(seq_list):

                # if i==j:

                if j >= i:   
                    if seq1 != seq2:
                        seq = [seq1, seq2]
                    else:
                        seq = seq1

                    _ , _ , data_all = get_train_val(train_or_test='test', sequences=seq)

                    loader = DataLoader(data_all, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)

                    acc , acc_pos, acc_neg = Evaluator(net, loader=loader, epochs=10, contrast_vid=contrast_vid, verbose=False)

                    acc_mat[i,j] = acc
                    acc_pos_mat[i,j] = acc_pos
                    acc_neg_mat[i,j] = acc_neg

                    print('for sequence: {}, size: {}, acc: {}%, acc_pos: {}%, acc_neg: {}%'.format(seq, len(data_all), acc, acc_pos, acc_neg),
                            file=open(save_file, "a")
                    )

        print('\ninter-video accuracy matrix:\n', file=open(save_file, "a"))
        print('acc_total: \n{} \n\nacc_pos: \n{} \n\nacc_neg: \n{}\n\n.'.format(acc_mat, acc_pos_mat, acc_neg_mat), file=open(save_file, "a"))
    

    # LOADING DATASET -- QUALITATIVE RESULTS -------------->

# %%
    # # # evaluating  ------------------------------------------

    # seq_list = ['conversation1_t3', 'femalemonologue2_t3', 'interactive1_t2', 'interactive4_t3', 'malemonologue2_t3']

    # for seq in seq_list:

    #     _ , _ , data_all = get_train_val(train_or_test='test', sequences=seq)

    #     loader = DataLoader(data_all, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)

    #     device = (torch.device('cuda') if torch.cuda.is_available()
    #         else torch.device('cpu'))

    #     net.to(device=device)
    #     net.eval()

    #     acc , acc_pos, acc_neg = Evaluator(net, loader=loader, epochs=1, contrast_vid=False, verbose=True, count=2)


    # for idx, data in enumerate(loader):

    #     aud = data[0].to(device=device)
    #     cam = data[1].to(device=device)
    #     img = data[2].to(device=device).squeeze(1)

    #     _, heatmap = net(img, aud, cam)

    #     print(idx)
    #     if idx==10:
    #         break

        # pass

    #     # acc , acc_pos, acc_neg = Evaluator(net, loader=loader, epochs=10, contrast_vid=False, verbose=False)




if __name__=='__main__':

    main()
