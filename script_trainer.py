
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
from fn_trainer import Trainer_contrast, Trainer_binary
from core.helper_fns import set_network



def main():

    random.seed(5)
    multi_mic = conf.logmelspectro['multi_mic']

    net, loss_fn = set_network()

    
    optimiser = optim.Adam(net.parameters(), lr=1e-4)
    # optimiser = optim.SGD(net.parameters(), lr=1e-2)

    epochs = conf.training_param['epochs']
    bs = conf.training_param['batch_size']

    # loading network ------------------------------------------

    load_net = False

    if load_net:

        fol_name = conf.filenames['net_folder_path']

        print(fol_name)
        ep = 28

        net_name = 'net_ep_%s.pt'%ep
        net_path = os.path.join(fol_name, net_name)

        checkpoint = torch.load(net_path)

        net.load_state_dict(checkpoint['model'])
        # optimiser.load_state_dict(checkpoint['optimizer'])

    else:
        ep = 0

    # ---------------------------------------------------------

    data_train, data_val, _ = get_train_val(multi_mic=multi_mic, train_or_test='train')

    train_loader = DataLoader(data_train, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(data_val, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True)

    device = conf.training_param['device']

    net = net.to(device=device)

    print('starting epochs: {}\ntotal epochs: {}\n'.format(ep, epochs))

    if conf.training_param['train_binary']:
        Trainer_binary(net,[ep+1, epochs], loss_fn, optimiser, train_loader, val_loader=val_loader)
    else:
        Trainer_contrast(net,[ep+1, epochs], loss_fn, optimiser, train_loader, val_loader=val_loader)


if __name__=='__main__':

    main()




