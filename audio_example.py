from fn.dataset import get_train_val
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from fn.sampling import gcc_collapse, flip_mics


data_all = get_train_val(multi_mic=True, train_or_test='train')

loader = DataLoader(data_all, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

aud = iter(loader).next()[0]

aud_flip = flip_mics(aud, return_all=False)

aud_ref, aud_gcc1 = gcc_collapse(aud, collapse_type='max')

aud_ref, aud_gcc2 = gcc_collapse(aud_flip, collapse_type='max')

plt.figure(figsize=(3,3))
plt.imshow(aud_gcc1.squeeze(0).T, aspect='auto')
plt.axis('off')
plt.show()

plt.figure(figsize=(3,3))
plt.imshow(aud_gcc2.squeeze(0).T, aspect='auto')
plt.axis('off')
plt.show()

# plt.figure()
# plt.subplot(121)
# plt.imshow(aud_gcc1.squeeze(0).T, aspect='auto')
# plt.title('(a)')
# plt.ylabel('time lag')
# plt.xlabel('microphone index')
# plt.yticks(ticks=list(range(0,64,16)), labels = list(range(-32,32,16)))
# plt.subplot(122)
# plt.imshow(aud_gcc2.squeeze(0).T, aspect='auto')
# plt.xlabel('microphone index')
# plt.yticks(ticks=list(range(0,64,16)), labels = list(range(-32,32,16)))
# plt.title('(b)')
# plt.show()


print('hi')