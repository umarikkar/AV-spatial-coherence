
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
from fn_trainer import Trainer, Trainer_AVOL



def main():

    random.seed(5)
    multi_mic = conf.logmelspectro['multi_mic']
    
    if conf.dnn_arch['AVOL']:
        net = AVOL_Net()
        loss_fn = nn.BCELoss()
    elif conf.dnn_arch['AVE']:
        net = AVE_Net()
        loss_fn = nn.CrossEntropyLoss()
    else:
        net = MergeNet()
        loss_fn = nn.BCELoss() if conf.dnn_arch['heatmap'] else nn.CrossEntropyLoss()

    epochs = 16
    optimiser = optim.Adam(net.parameters(), lr=1e-4)

    # # loading network ------------------------------------------

    # fol_name = conf.filenames['net_folder_path']

    # print(fol_name)
    # ep = 8

    # net_name = 'net_ep_%s.pt'%ep
    # net_path = os.path.join(fol_name, net_name)

    # checkpoint = torch.load(net_path)

    # net.load_state_dict(checkpoint['model'])
    # optimiser.load_state_dict(checkpoint['optimizer'])

    # # ---------------------------------------------------------

    data_train, data_val, _ = get_train_val(multi_mic=multi_mic, train_or_test='train')

    train_loader = DataLoader(data_train, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(data_val, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)

    device = (torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu'))

    net = net.to(device=device)

    # if conf.dnn_arch['AVOL']:
    #     Trainer_AVOL(net,[1, epochs], loss_fn, optimiser, train_loader, val_loader=val_loader)
    # else:
    #     Trainer(net,[1, epochs], loss_fn, optimiser, train_loader, val_loader=val_loader)

 
    Trainer(net,[1, epochs], loss_fn, optimiser, train_loader, val_loader=val_loader)


if __name__=='__main__':

    main()




