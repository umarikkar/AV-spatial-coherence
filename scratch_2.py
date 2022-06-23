
import os
from multiprocessing.dummy import freeze_support

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from pathlib import Path

import core.config as conf
import utils.utils as utils
from AVOL import MergeNet, SubNet_main, Trainer
from core.dataset import dataset_from_hdf5, dataset_from_scratch

def get_dataset(from_h5=True, multi_mic=True, train_or_test='train'):


    base_path = conf.input['project_path']

    if from_h5:
        mic_info = 'MultiChannel' if multi_mic else 'SingleChannel'
        h5py_dir_str = os.path.join(base_path, 'data', 'h5py_%s' %mic_info,'')
        h5py_name = '%s_%s.h5' % (train_or_test, mic_info)

        h5py_path_str = os.path.join(h5py_dir_str, h5py_name)
        h5py_path = Path(h5py_path_str)
        d_dataset = dataset_from_hdf5(h5py_path)

    else:
        csv_file_path = os.path.join(base_path, 'data', 'train.csv')
        d_dataset = dataset_from_scratch(csv_file_path, train_or_test='train', normalize=False, augment=False)

    # DATA LOADER INITIALISATION -----------------------------------------------------------------------------

    data_loader = DataLoader(d_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    sz = 1024
    sub_idxs = torch.randint(low=0, high=len(d_dataset), size=(sz,))

    d_subset = Subset(d_dataset, sub_idxs)
    toy_loader = DataLoader(d_subset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)

    return data_loader, toy_loader


def main():
    # LOSS and OPTIMISERS
    multi_mic = conf.logmelspectro['get_gcc']

    net = MergeNet(multi_mic=multi_mic)

    epochs = 40
    optimiser = optim.Adam(net.parameters(), lr=1e-4)

    loss_fn = nn.BCELoss() if conf.dnn_arch['heatmap'] else nn.CrossEntropyLoss()

    data_loader, toy_loader = get_dataset(from_h5=True, multi_mic=multi_mic, train_or_test='train')

    device = (torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu'))

    net = net.to(device=device)

    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

    # net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    net.apply(init_weights)

    Trainer(net, epochs, loss_fn, optimiser, toy_loader, val_loader=None, multi_mic=multi_mic)



if __name__=='__main__':
    main()




