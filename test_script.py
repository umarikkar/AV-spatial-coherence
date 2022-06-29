
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
from fn_networks import MergeNet, MergeNet_eval
from fn_trainer import Trainer


def main():

    random.seed(5)
    multi_mic = conf.logmelspectro['multi_mic']
    net = MergeNet_eval()
    epochs = 40
    optimiser = optim.Adam(net.parameters(), lr=1e-4)
    loss_fn = nn.BCELoss() if conf.dnn_arch['heatmap'] else nn.CrossEntropyLoss()

    data_train, data_val = get_train_val(multi_mic=multi_mic, train_or_test='train')

    train_loader = DataLoader(data_train, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(data_val, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    device = (torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu'))

    # loading network ----------------------------------------

    fol_name = conf.filenames['net_folder_path']

    print(fol_name)
    ep = 36

    net_name = 'net_ep_%s.pt'%ep
    net_path = os.path.join(fol_name, net_name)

    checkpoint = torch.load(net_path)

    net.load_state_dict(checkpoint['model'])
    optimiser.load_state_dict(checkpoint['optimizer'])

    net.eval()

    # evaluating some stuff ------------------------------------

    # load a single data point

    if conf.filenames['train_val']=='train':
        loader = train_loader
    else:
        loader = val_loader

    net = net.to(device=device)

    count = 0
    for data in loader:
        count +=1
        print('count: ', count)
        aud = data[0].to(device=device)
        cam = data[1].to(device=device)
        img = data[2].squeeze(1).to(device=device)
        
        out = net(img, aud, cam)

        if count==10:
            break
    pass


if __name__=='__main__':

    main()
