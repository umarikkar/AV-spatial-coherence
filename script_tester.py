
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
from fn_networks import MergeNet, AVOL_Net
from fn_trainer import Trainer, Evaluator


def main():

    random.seed(5)

    # loading network ----------------------------------------

    if conf.dnn_arch['AVOL']:
        net = AVOL_Net()
    else:
        net = MergeNet()

    optimiser = optim.Adam(net.parameters(), lr=1e-4)

    fol_name = conf.filenames['net_folder_path']

    print(fol_name)
    ep = 16

    net_name = 'net_ep_%s.pt'%ep
    net_path = os.path.join(fol_name, net_name)

    checkpoint = torch.load(net_path)

    net.load_state_dict(checkpoint['model'])
    optimiser.load_state_dict(checkpoint['optimizer'])

    # loading dataset --------------------------------------


    seq_list = ['conversation1_t3', 'femalemonologue2_t3', 'interactive1_t2', 'interactive4_t3', 'malemonologue2_t3']
    # seq_list = ['all']

    seq_compare = True if len(seq_list) > 1 else False

    seq_type = 'all' if seq_list[0]=='all' else 'seq'
    # contrast_vid_setting = [False, True]
    contrast_vid_setting = [True]

    for contrast_vid in contrast_vid_setting:

        if contrast_vid:
            eval_name = 'eval_results_%s_vid.txt'%seq_type
        else:
            eval_name = 'eval_results_%s.txt'%seq_type

        save_file = os.path.join(conf.filenames['net_folder_path'], eval_name)

        print(save_file)

        if seq_compare:
            acc_mat = np.zeros((len(seq_list), len(seq_list)))
            acc_pos_mat = np.zeros((len(seq_list), len(seq_list)))
            acc_neg_mat = np.zeros((len(seq_list), len(seq_list)))

        for i, seq1 in enumerate(seq_list):

            for j, seq2 in enumerate(seq_list):

                if j > i:

                    seq = [seq1, seq2]

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





    # evaluating  ------------------------------------------

    


if __name__=='__main__':

    main()
