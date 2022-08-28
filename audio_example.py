from fn.dataset import get_train_val
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn


data_all = get_train_val(multi_mic=True, train_or_test='train')


aud = data_all[0][0]

c_bot = [0,1,2,3,4,6,7,8,9,10]
c_top = [11,12,13,14,15]

aud_ref = aud[5]

aud_bot = aud[c_bot]
aud_top = aud[c_top]

aud_bot_mean = aud_bot.mean(dim=-2)
aud_top_mean = aud_top.mean(dim=-2)

aud_mean = torch.cat((aud_bot_mean, aud_top_mean), dim=0).unsqueeze(0)
x = torch.max_pool2d(aud_mean, (1,2), (1,2))

sz_FC1 = x.shape[-1]*x.shape[-2]
sz_FC2 = 64
sz_ex = 14*14

FCLayer1 = nn.Linear(sz_FC1, sz_FC2)
bn = nn.BatchNorm1d(num_features=sz_FC2)
FCLayer2 = nn.Linear(sz_FC2, sz_ex)

x_in = x.reshape((-1, sz_FC1))

x_in = FCLayer1(x_in)
x_in = FCLayer2(x_in)

x_out = x_in.reshape((-1, 14, 14))

print(x_in.shape)




# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(aud_mean)

# aud_mean = aud_mean.unsqueeze(0).unsqueeze(0)



# plt.subplot(1,2,2)
# plt.imshow(out.squeeze())

# plt.show()

# bn1 = nn.BatchNorm2d(num_features=16)
# conv1 = nn.Conv2d(1,1,3,stride=(1,2))
# conv2 = nn.Conv2d(1,1,3,stride=(1,2))

# out = conv2(bn1(conv1(aud_mean)))


# plt.figure()
# plt.subplot(1,3,1)
# plt.imshow(aud_bot_mean.T, aspect='equal')
# plt.title('bottom mics')
# plt.ylabel('lag')
# plt.xlabel('channel')
# plt.subplot(1,3,2)
# plt.imshow(aud_top_mean.T, aspect='equal')
# plt.title('top mics')
# plt.ylabel('lag')
# plt.xlabel('channel')
# plt.subplot(1,3,3)
# plt.imshow(aud_mean.T, aspect='equal')
# plt.title('all mics')
# plt.ylabel('lag')
# plt.xlabel('channel')
# plt.show()

print('hi')