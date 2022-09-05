import torch
import matplotlib.pyplot as plt
import os


import core.config as conf

fol_name = conf.filenames['net_folder_path']

# fol_name = fol_name[:fol_name.find('FC')]

net_names = ['FC128_MF_S', 'FC128_MF_S_skip', 'FC128_MF_S_res']
lbs = ['GRU only', '+ skip connection', '+ difference input']

plt.figure()


log_file = os.path.join(fol_name, 'train_logs.txt')

strs = list(open(log_file))

ls_ez = []
ls_hd = []

for line in strs:

    loss_str = line[line.find('Train loss ')+11:line.find('Train loss ')+17]

    loss_ez = line[line.find('ez: ')+4:line.find('ez: ')+10]
    loss_hd = line[line.find('hd ')+3:line.find('hd ')+9]

    losses = [loss_ez, loss_hd]

    cond=True
    while cond:
        if ',' in loss_ez:
            loss_ez = loss_ez[:-1]
        else:
            cond=False

    cond=True
    while cond:
        if ',' in loss_hd:
            loss_hd = loss_hd[:-1]
        else:
            cond=False

    loss_ez = float(loss_ez)
    loss_hd = float(loss_hd)

    ls_ez.append(loss_ez)
    ls_hd.append(loss_hd)

plt.plot(ls_ez, label='L_(original)')
plt.plot(ls_hd, label='L_(spatial)')
plt.legend()
plt.grid()
plt.xlabel('epochs')
plt.ylabel('training loss')
plt.show()





