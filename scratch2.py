import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchaudio.models as models_audio
import torchvision.models as models_vision
from torch.utils.data import DataLoader
from torchvision import transforms

import core.config as conf
from core.dataset import dataset_from_scratch, get_train_val

from fn_networks import AVOL_Net

# base_path = os.getcwd()

# train_or_test = 'train'
# train_csv = os.path.join(base_path, 'data', 'csv', '%s.csv'%(train_or_test))

# train_or_test = 'test'
# test_csv = os.path.join(base_path, 'data', 'csv', '%s.csv'%(train_or_test))


# d_dataset = dataset_from_scratch(test_csv, train_or_test=train_or_test)

# data = d_dataset[50]

# for d in data:   
#     print(type(d))

# pass

d_train, d_val, d_dataset = get_train_val(train_or_test='train')

loader = DataLoader(d_train, batch_size=4, shuffle=True)

randint = np.random.randint(len(d_train))

# data = d_train[randint]

data = next(iter(loader))

aud = data[0]
cam = data[1]
imgs = data[2]

# plt.figure()
# plt.imshow(imgs[0,:,:,:].permute(1,2,0).detach().numpy(), aspect='auto')

# plt.show()

net = AVOL_Net()

# count = 0
# for idx, layer in enumerate(net):
#     if isinstance(layer, nn.Conv2d):
#         count+=1
#         print(idx, count)

x_class, x_map = net(imgs=imgs, audio=aud, cam=cam)

print('out:', x_class.shape)
