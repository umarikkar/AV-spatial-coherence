
import os
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import core.config as conf
from fn.dataset import get_train_val
from fn.sampling import create_samples
from fn.nets import set_network
from fn.train_eval import Evaluator, plot_results, eval_loc

import matplotlib.pyplot as plt

from tqdm import tqdm


def main():

    random.seed(5)

    # loading network ----------------------------------------

    net, loss_fn = set_network(set_train=False)

    fol_name = conf.filenames['net_folder_path']

    bs = conf.training_param['batch_size']

    # seq_list = ['all']
    seq_list = ['conversation1_t3', 'femalemonologue2_t3', 'interactive1_t2', 'interactive4_t3', 'malemonologue2_t3']
    # seq_list = ['femalemonologue2_t3']

    epoch_settings = [80]

    for ep in epoch_settings:

        net_name = 'net_ep_%s.pt'%ep
        net_path = os.path.join(fol_name, net_name)

        checkpoint = torch.load(net_path)

        net.load_state_dict(checkpoint['model'])

        if seq_list[0] == 'all':
            seq_type = 'all'
        else:
            seq_type = 'seq'
        

        eval_name = 'eval_results_%s.txt'%seq_type

        save_file = os.path.join(conf.filenames['net_folder_path'], eval_name)

        print('save file:', save_file)

        acc_mat = np.zeros((len(seq_list), len(seq_list)))
        acc_pos_mat = np.zeros((len(seq_list), len(seq_list)))
        acc_neg_mat = np.zeros((len(seq_list), len(seq_list)))

        for i, seq1 in enumerate(seq_list):
            for j, seq2 in enumerate(seq_list):

                if j > i:   
                    if seq1 != seq2:
                        seq = [seq1, seq2]
                        vid_contrast = True
                    else:
                        seq = seq1
                        vid_contrast = False

                    data_all = get_train_val(train_or_test='test', sequences=seq)

                    loader = DataLoader(data_all, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True)

                    acc, acc_pos, acc_neg = Evaluator(net, loader=loader, loss_fn=loss_fn,  epochs=1, verbose=False, vid_contrast=vid_contrast)

                    acc_mat[i,j] = acc
                    acc_pos_mat[i,j] = acc_pos
                    acc_neg_mat[i,j] = acc_neg

                    print_txt = 'for sequence: {}, size: {}, acc: {}%, acc_pos: {}%, acc_neg: {}%'.format(seq, len(data_all), acc, acc_pos, acc_neg)

                    print(print_txt, file=open(save_file, "a"))
                    print(print_txt)
                

        print_txt = '\ninter-video accuracy matrix:\nepochs: {} \n\nacc_total: \n{} \n\nacc_pos: \n{} \n\nacc_neg: \n{}\n\n.'.format(ep, acc_mat, acc_pos_mat, acc_neg_mat)
        print(print_txt, file=open(save_file, "a"))
        print(print_txt)

    


    # # loading dataset (QUANTITATIVE RESULTS)--------------------------------------

    # ep = 100
    # net, loss_fn = set_network(set_train=False)
    
    # seq_list = ['all']

    # net_name = 'net_ep_%s.pt'%ep
    # fol_name = conf.filenames['net_folder_path']
    # net_path = os.path.join(fol_name, net_name)

    # checkpoint = torch.load(net_path)

    # net.load_state_dict(checkpoint['model'])





    # if seq_list[0] == 'all':
    #     seq_type = 'all'
    # else:
    #     seq_type = 'seq'
    

    # eval_name = 'eval_results_%s.txt'%seq_type

    # save_file = os.path.join(conf.filenames['net_folder_path'], eval_name)

    # print('save file:', save_file)

    # acc_mat = np.zeros((len(seq_list), len(seq_list)))
    # acc_pos_mat = np.zeros((len(seq_list), len(seq_list)))
    # acc_neg_mat = np.zeros((len(seq_list), len(seq_list)))

    # for i, seq1 in enumerate(seq_list):
    #     for j, seq2 in enumerate(seq_list):

    #         if j >= i:   
    #             if seq1 != seq2:
    #                 seq = [seq1, seq2]
    #             else:
    #                 seq = seq1

    #             data_all = get_train_val(train_or_test='test', sequences=seq)

    #             loader = DataLoader(data_all, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)

    #             acc, acc_pos, acc_neg = Evaluator(net, loader=loader, loss_fn=loss_fn,  epochs=1, verbose=False)

    #             acc_mat[i,j] = acc
    #             acc_pos_mat[i,j] = acc_pos
    #             acc_neg_mat[i,j] = acc_neg

    #             print_txt = 'for sequence: {}, size: {}, acc: {}%, acc_pos: {}%, acc_neg: {}%'.format(seq, len(data_all), acc, acc_pos, acc_neg)

    #             print(print_txt, file=open(save_file, "a"))
    #             print(print_txt)
            

    # print_txt = '\ninter-video accuracy matrix:\nepochs: {} \n\nacc_total: \n{} \n\nacc_pos: \n{} \n\nacc_neg: \n{}\n\n.'.format(ep, acc_mat, acc_pos_mat, acc_neg_mat)
    # print(print_txt, file=open(save_file, "a"))
    # print(print_txt)








if __name__=='__main__':

    main()
