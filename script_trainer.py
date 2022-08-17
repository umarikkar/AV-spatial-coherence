
import os
import random

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import core.config as conf
from fn.dataset import get_train_val
from fn.train_eval import Trainer
from fn.nets import set_network



def main():

    random.seed(5)
    multi_mic = conf.logmelspectro['multi_mic']

    net, loss_fn = set_network()

    optimiser = optim.Adam(net.parameters(), lr=1e-4)
    # optimiser = optim.SGD(net.parameters(), lr=1e-2)

    epochs = conf.training_param['epochs']
    bs = conf.training_param['batch_size']

    # loading network ------------------------------------------

    load_net = True

    if load_net:

        fol_name = conf.filenames['net_folder_path']

        print(fol_name)
        ep =  16

        net_name = 'net_ep_%s.pt'%ep
        net_path = os.path.join(fol_name, net_name)

        checkpoint = torch.load(net_path)

        net.load_state_dict(checkpoint['model'])
        # optimiser.load_state_dict(checkpoint['optimizer'])

    else:
        ep = 0

    # ---------------------------------------------------------

    data_all = get_train_val(multi_mic=multi_mic, train_or_test='train')

    # _, _, data_train = get_train_val(multi_mic=multi_mic, train_or_test='test')

    # train_loader = DataLoader(data_train, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True)
    # val_loader = DataLoader(data_val, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True)

    device = conf.training_param['device']

    net = net.to(device=device)

    print('starting epochs: {}\ntotal epochs: {}\n'.format(ep, epochs))

    Trainer(net,[ep+1, epochs], loss_fn, optimiser, dataset=data_all)


if __name__=='__main__':

    main()




