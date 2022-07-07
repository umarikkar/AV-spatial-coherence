
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

    seq_type = 'all' if seq_list[0]=='all' else 'seq'
    eval_name = 'eval_results_%s.txt'%seq_type 

    save_file = os.path.join(conf.filenames['net_folder_path'], eval_name)

    if eval_name in os.listdir(conf.filenames['net_folder_path']):
        os.remove(save_file)

    for seq in seq_list:

        _ , _ , data_all = get_train_val(train_or_test='test', sequences=seq)

        loader = DataLoader(data_all, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)

        acc = Evaluator(net, loader=loader, epochs=10)

        print('for sequence: {}, size: {}, acc: {}%'.format(seq, len(data_all), acc),
                file=open(save_file, "a")
        )





    # evaluating  ------------------------------------------

    


if __name__=='__main__':

    main()
