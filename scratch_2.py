
from multiprocessing.dummy import freeze_support
import os
from torch.utils.data import DataLoader, Subset

from core.dataset import dataset_from_scratch, dataset_from_hdf5
import core.config as conf

import utils.utils as utils

import numpy as np

from AVOL import SubNet_main, MergeNet

current_dir = os.getcwd()
csv_file_path = os.path.join(conf.input['project_path'], 'data', 'train.csv')


d_dataset = dataset_from_scratch(csv_file_path, train_or_test='train', normalize=False, augment=False)
data_loader = DataLoader(d_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

d_subset = Subset(d_dataset, list(range(4)))
d_sub = DataLoader(d_subset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

net = MergeNet()


if __name__=='__main__':

    # freeze_support()

    for val in d_sub:
        img = val[-1][0]
        aud = val[0]
        print(img.shape, aud.shape)
        
        # im, au = net(img, aud)
        
        # print(im.shape, au.shape)

    pass




