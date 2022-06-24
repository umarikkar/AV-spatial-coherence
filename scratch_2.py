
import os
from multiprocessing.dummy import freeze_support

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split
from pathlib import Path
from math import floor, ceil

import random


import core.config as conf
import utils.utils as utils
from AVOL import MergeNet, SubNet_main, Trainer
from core.dataset import dataset_from_hdf5, dataset_from_scratch

def get_dataset(from_h5=True, multi_mic=True, train_or_test='train', toy=False):

    base_path = conf.input['project_path']

    if from_h5:
        mic_info = 'MultiChannel' if multi_mic else 'SingleChannel'
        h5py_dir_str = os.path.join(base_path, 'data', 'h5py_%s' %mic_info,'')
        h5py_name = '%s_%s.h5' % (train_or_test, mic_info)

        h5py_path_str = os.path.join(h5py_dir_str, h5py_name)
        h5py_path = Path(h5py_path_str)
        d_dataset = dataset_from_hdf5(h5py_path, augment=True)

    else:
        csv_file_path = os.path.join(base_path, 'data', 'train.csv')
        d_dataset = dataset_from_scratch(csv_file_path, train_or_test='train', normalize=False, augment=False)

    if toy:
        sz = 32
        rand_toy = list(range(sz))
        random.shuffle(rand_toy)
        d_dataset = Subset(d_dataset, rand_toy)

    # DATA LOADER INITIALISATION -----------------------------------------------------------------------------
    rand_idxs = list(range(len(d_dataset)))
    random.shuffle(rand_idxs)

    train_size = floor(0.8*len(d_dataset))
    train_idxs = rand_idxs[0:train_size]
    val_idxs = rand_idxs[train_size:]

    data_train = Subset(d_dataset, train_idxs)
    data_val = Subset(d_dataset, val_idxs)

    train_loader = DataLoader(data_train, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(data_val, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    # print(len(data_train))

    return train_loader, val_loader


def main():
    # LOSS and OPTIMISERS
    multi_mic = conf.logmelspectro['get_gcc']

    net = MergeNet(multi_mic=multi_mic)

    epochs = 40
    optimiser = optim.Adam(net.parameters(), lr=1e-4)

    loss_fn = nn.BCELoss() if conf.dnn_arch['heatmap'] else nn.CrossEntropyLoss()

    train_loader, val_loader = get_dataset(from_h5=True, multi_mic=multi_mic, train_or_test='train')

    device = (torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu'))

    net = net.to(device=device)

    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

    # net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    net.apply(init_weights)

    Trainer(net, epochs, loss_fn, optimiser, train_loader, val_loader, multi_mic=multi_mic)



if __name__=='__main__':
    random.seed(5)

    main()




