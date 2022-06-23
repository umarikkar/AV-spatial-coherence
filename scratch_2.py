
import os
from multiprocessing.dummy import freeze_support

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset

import core.config as conf
import utils.utils as utils
from AVOL import MergeNet, SubNet_main, Trainer
from core.dataset import dataset_from_hdf5, dataset_from_scratch

current_dir = os.getcwd()
csv_file_path = os.path.join(conf.input['project_path'], 'data', 'train.csv')

# DATA LOADER INITIALISATION -----------------------------------------------------------------------------

d_dataset = dataset_from_scratch(csv_file_path, train_or_test='train', normalize=False, augment=False)
data_loader = DataLoader(d_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

sz = 64

grr = torch.randint(low=0, high=len(d_dataset), size=(sz,))
print(grr)

d_subset = Subset(d_dataset, grr)
d_sub = DataLoader(d_subset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)

# LOSS and OPTIMISERS ------------------------------------------------------------------------------------------

multi_mic = False

net = MergeNet(multi_mic=multi_mic)

epochs = 40
optimiser = optim.Adam(net.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

device = (torch.device('cuda') if torch.cuda.is_available()
        else torch.device('cpu'))

net = net.to(device=device)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

# net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
net.apply(init_weights)

if __name__=='__main__':

    # print(len(d_subset))

    # freeze_support()

    # for val in d_sub:
    #     img = val[-1][0]
    #     aud = val[0]
    #     print(img.shape, aud.shape)
        
    #     # im, au = net(img, aud)
        
    #     # print(im.shape, au.shape)

    # pass

    Trainer(net, epochs, loss_fn, optimiser, d_sub, val_loader=None, multi_mic=multi_mic)




