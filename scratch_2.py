
import os
import random
from math import ceil, floor
from multiprocessing.dummy import freeze_support
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split

import core.config as conf
import utils.utils as utils
from nets.merge import MergeNet, SubNet_main, Trainer
from core.dataset import dataset_from_hdf5, dataset_from_scratch, get_train_val


def main():

    data_train, data_val = get_train_val(from_h5=True, multi_mic=multi_mic, train_or_test='train', toy=True, toy_size=16)

    train_loader = DataLoader(data_train, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(data_val, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    device = (torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu'))

    net = net.to(device=device)

    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)

    Trainer(net, epochs, loss_fn, optimiser, train_loader, val_loader, multi_mic=multi_mic)



if __name__=='__main__':
    random.seed(5)

    multi_mic = conf.logmelspectro['get_gcc']

    net = MergeNet(multi_mic=multi_mic)

    epochs = 40
    optimiser = optim.Adam(net.parameters(), lr=1e-4)

    loss_fn = nn.BCELoss() if conf.dnn_arch['heatmap'] else nn.CrossEntropyLoss()

    main()




