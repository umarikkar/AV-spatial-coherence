#!/usr/bin/python

import numpy as np
import torch
import torch.nn as nn
import argparse

import os

from torchvision import transforms

#mean_vector = torch.load(input['project_path'] + 'output/test/mean_vector.pt')
#std_vector = torch.load(input['project_path'] + 'output/test/std_vector.pt')


# ------ setting the file name ---------------------------------------------------------------------
def set_filename(name='AVE', 
                toy_params=(False, 128),
                print_path=False
                ):

    mic_name = 'MC' if logmelspectro['multi_mic'] else 'SC'

    if toy_params[0]: 
        net_size = str(toy_params[1])
    else:
        net_size = ''

    fol_name = mic_name + net_size + '_' + name

    """
    # if name=='AVOL':
    #     fol_name = mic_name + net_size + '_AVOL'
    # elif name=='AVE':
    #     fol_name = mic_name + net_size + '_AVE'
    # elif name=='EZ_VSL':
    #     fol_name = mic_name +net_size + '_EZ_VSL'
    # else:
    #     if not dnn_arch['vid_custom']:
    #         if dnn_arch['vid_pretrained']:
    #             if not dnn_arch['vid_freeze']:
    #                 fol_name = mic_name + '_' + net_size + '_vgg_flex' # MultiChannel_sz_vgg_flex
    #             else:
    #                 fol_name = mic_name + '_' + net_size + '_vgg_freeze' # MultiChannel_sz_vgg_freeze
    #         else:
    #             fol_name = mic_name + '_' + net_size + '_vgg' # MultiChannel_sz_vgg_flex
    #     else:
    #         fol_name = mic_name + '_' + net_size # MultiChannel_sz

    #     if dnn_arch['heatmap']:
    #         fol_name = fol_name + '_BCE'
    """

    if training_param['vid_contrast']:
        fol_name = fol_name + '_vid'

    if training_param['frame_seq']:
        fol_name = fol_name + '_MF'

    if not name=='AVE' and dnn_arch['ave_backbone']:
        fol_name = fol_name + '_pretrained'

    if data_param['filter_silent']:
        fol_name = fol_name + '_filtered'

    file_path = os.path.join(os.getcwd(), 'results', 'checkpoints', fol_name)  # results/checkpoints/MultiChannel_sz

    if not os.path.exists(file_path):
        os.mkdir(file_path)

    imgs = ['train', 'val', 'test']
    for img in imgs:
        img_path = os.path.join(file_path, img)
        if not os.path.exists(img_path):
            os.mkdir(img_path)

    if print_path:
        print('network path', file_path)

    return file_path


proj_path = os.getcwd()

input = {
    'project_path': proj_path,
    'fps': 30,
    'sr': 48000, # 48 kHz
    'frame_len_sec': 2, # seconds
    'frame_step_train': 1, #seconds, has overlaps
    'frame_step_test': 2, #seconds, no overlaps
}

dnn_arch = {
    'heatmap':False,

    'vid_custom':False,
    'vid_pretrained': True,
    'vid_freeze':True,

    'ave_backbone': True,

    'aud_custom': True,      
    'aud_pretrained': False,
    'aud_freeze': False,

    'net_name':'AVOL',
    # AVE, EZ_VSL

    'ave_backbone':False

}

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

training_param = {

    'optimizer': torch.optim.Adam,
    #'criterion': nn.CrossEntropyLoss,
    'learning_rate': 0.001, # this is used if user does not provide another lr with the parser
    'epochs': 60, # this is used if user does not provide the epoch number with the parser
    'batch_size': 32,
    'frame_len_samples': input['frame_len_sec'] * input['sr'], # number of audio samples in 2 sec

    'frame_seq': False,
    'frame_vid_samples': 5,

    'toy_params': (False, 128),

    'inference': True,
    'vid_contrast': False,

    'device': device,
    'train_binary': False,
    #'input_norm': 'freq_wise', # choose between: 'freq_wise', 'global', or None
    #'step_size':,
    #'gamma': ,
}

data_param = {

    't_image' : transforms.Compose([
        # transforms.Resize((112,112)),
        transforms.ColorJitter((0.6, 1.4), (0.6, 1.4), (0.6, 1.4), (-0.2, 0.2)),
        transforms.RandomGrayscale(0.2),
        transforms.RandomInvert(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        ]) ,

    't_audio' : transforms.Compose([
        transforms.RandomAutocontrast(p=0.5),
        transforms.RandomAdjustSharpness(2),
        ]) ,

    'filter_silent':True,
}

logmelspectro = {
    'get_gcc':  True,
    'multi_mic': True,
    'mfcc_azimuth_only': True, # False for using all the 89 look dir, True only the 15 central-horizontal look dir
    'winlen': 512, # samples
    'hoplen': 100, # samples
    'numcep': 64, # number of cepstrum bins to return
    'n_fft': 512, #fft lenght
    'fmin': 40, #Hz
    'fmax': 24000 #Hz
}


filenames = {
    'net_folder_path': set_filename(name=dnn_arch['net_name'], 
                            toy_params=training_param['toy_params'],
                            print_path=True),
    'train_val': 'test',
}

