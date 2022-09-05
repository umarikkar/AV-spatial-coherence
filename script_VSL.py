
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

    seq_list = ['all']
    # seq_list = ['all', 'conversation1_t3', 'femalemonologue2_t3', 'interactive1_t2', 'interactive4_t3', 'malemonologue2_t3']
    # seq_list = ['femalemonologue2_t3']

    plot_res= False

    bs = 1 if plot_res else 16

    for seq in seq_list:

        data_all = get_train_val(train_or_test='test', sequences=seq)
        loader = DataLoader(data_all, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True)
        fol_name = conf.filenames['net_folder_path']

        # epoch_settings = np.linspace(4,16,4).astype('int')

        # epoch_settings = np.linspace(44, 60, 5).astype('int')

        epoch_settings = [
            # 1,
                10, 
                20, 
                30, 
                # 40,
                40, 50, 60
                ]
        # epoch_settings = [10]

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



if __name__=='__main__':

    main()
