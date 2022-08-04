
import os
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import core.config as conf
from fn.dataset import get_train_val, create_samples
from fn.nets import set_network
from fn.train_eval import Evaluator

import matplotlib.pyplot as plt

def main():

    random.seed(5)

    # loading network ----------------------------------------

    net, loss_fn = set_network(set_train=False)


    optimiser = optim.Adam(net.parameters(), lr=1e-4)

    fol_name = conf.filenames['net_folder_path']

    # epoch_settings = np.linspace(4, 40, 10).astype('int')
    epoch_settings = [52]

    for ep in epoch_settings:

        # ep= 40


        net_name = 'net_ep_%s.pt'%ep
        net_path = os.path.join(fol_name, net_name)

        checkpoint = torch.load(net_path)

        net.load_state_dict(checkpoint['model'])
        optimiser.load_state_dict(checkpoint['optimizer'])

    
        # %% 
        # loading dataset (QUANTITATIVE RESULTS)--------------------------------------


        seq_list = ['conversation1_t3', 'femalemonologue2_t3', 'interactive1_t2', 'interactive4_t3', 'malemonologue2_t3']
        # # seq_list = ['all']

        # seq_compare = True if len(seq_list) > 1 else False

        # if seq_list[0] == 'all':
        #     seq_type = 'all'
        # else:
        #     seq_type = 'seq'
        
        # contrast_vid_setting = [False]

        # for contrast_vid in contrast_vid_setting:

        #     if contrast_vid:
        #         eval_name = 'eval_results_%s_vid.txt'%seq_type
        #     else:
        #         eval_name = 'eval_results_%s.txt'%seq_type

        #     save_file = os.path.join(conf.filenames['net_folder_path'], eval_name)

        #     print('save file:', save_file)

        #     acc_mat = np.zeros((len(seq_list), len(seq_list)))
        #     acc_pos_mat = np.zeros((len(seq_list), len(seq_list)))
        #     acc_neg_mat = np.zeros((len(seq_list), len(seq_list)))

        #     for i, seq1 in enumerate(seq_list):

        #         for j, seq2 in enumerate(seq_list):

        #             # if i==j:

        #             if j >= i:   
        #                 if seq1 != seq2:
        #                     seq = [seq1, seq2]
        #                 else:
        #                     seq = seq1

        #                 _ , _ , data_all = get_train_val(train_or_test='test', sequences=seq)

        #                 loader = DataLoader(data_all, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)

        #                 acc, acc_pos, acc_neg = Evaluator(net, loader=loader, loss_fn=loss_fn,  epochs=1, contrast_vid=contrast_vid, verbose=False)

        #                 acc_mat[i,j] = acc
        #                 acc_pos_mat[i,j] = acc_pos
        #                 acc_neg_mat[i,j] = acc_neg

        #                 print_txt = 'for sequence: {}, size: {}, acc: {}%, acc_pos: {}%, acc_neg: {}%'.format(seq, len(data_all), acc, acc_pos, acc_neg)

        #                 print(print_txt, file=open(save_file, "a"))
        #                 print(print_txt)
                    

        #     print_txt = '\ninter-video accuracy matrix:\nepochs: {} \n\nacc_total: \n{} \n\nacc_pos: \n{} \n\nacc_neg: \n{}\n\n.'.format(ep, acc_mat, acc_pos_mat, acc_neg_mat)
        #     print(print_txt, file=open(save_file, "a"))
        #     print(print_txt)
        

        # # LOADING DATASET -- QUALITATIVE RESULTS -------------->

    # %%
        # evaluating  ------------------------------------------

        # seq_list = ['conversation1_t3', 'femalemonologue2_t3', 'interactive1_t2', 'interactive4_t3', 'malemonologue2_t3']
        seq_list = ['all']

        for i, seq in enumerate(seq_list):

            _ , _ , data_all = get_train_val(train_or_test='test', sequences=seq)

            loader = DataLoader(data_all, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

            device = (torch.device('cuda') if torch.cuda.is_available()
                else torch.device('cpu'))

            net.to(device=device)
            net.eval()

            num_plots = 5
            plt.figure(i+1, figsize=(15, 6))

            count=0

            for _, data in enumerate(loader):

                if count < num_plots:

                    if data[6] == [b'SPEAKING']:

                        samples = create_samples(data, augment=False, device=device, return_mat=False)
                        imgs_all = samples['imgs_all']
                        score, heatmap = net(samples['imgs_all'], samples['audio_all'], samples['cam_all'] )

                        heatmap = heatmap.cpu().squeeze(0).detach().numpy()

                        imgs_all = imgs_all.cpu().squeeze(0).permute(1,2,0).detach().numpy()

                        plt.subplot(2, num_plots, count+1)
                        plt.imshow(imgs_all, aspect='equal')
                        plt.axis('off')

                        plt.subplot(2, num_plots, num_plots+count+1)
                        plt.imshow(heatmap, vmin=0, vmax=1, aspect='equal')
                        # plt.colorbar()
                        plt.title('score:%s'%round(score.item(), 2))
                        plt.axis('off')

                        count += 1

                else:

                    plt.suptitle('epochs: '+str(ep))

                    fig_path = os.path.join(conf.filenames['net_folder_path'], conf.filenames['train_val'])
                    num = len(os.listdir(fig_path)) + 1
                    plt.savefig(os.path.join(conf.filenames['net_folder_path'], conf.filenames['train_val'], 'img_%s'%num + '.png'))
                    plt.close()
                    plt.show()

                    break








if __name__=='__main__':

    main()
