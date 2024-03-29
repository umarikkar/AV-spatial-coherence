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
                print_path=False,
                hard_negatives = None,
                ):

    if hard_negatives is None:
        hard_negatives = contrast_param['hard_negatives']

    mic_name = 'MC' if logmelspectro['multi_mic'] else 'SC'

    if toy_params[0]: 
        net_size = 'TOY'+str(toy_params[1])
    else:
        net_size = ''

    if data_param['filter_silent']:
        sil = 'silent_filtered'
    else:
        sil = 'silent_unfiltered'

    fol_name = os.path.join(input['project_path'], 'results',  'checkpoints', 'sec_%d'%input['frame_len_sec'], sil, name + '_' + mic_name)

    net_name = net_size+'FC%d'%dnn_arch['FC_size'] 

    if training_param['frame_seq']:
        net_name += '_MF'
    else:
        net_name += '_SF'

    # net_name += '_L2'

    if dnn_arch['small_img']:
        net_name += '_S'
    else:
        net_name += '_L'

    if training_param['frame_seq'] and name=='AVE':
        if dnn_arch['seq_res']:
            net_name += '_res'
        elif dnn_arch['seq_skip']:
            net_name += '_skip'

    if not dnn_arch['small_features']:
        net_name += '_large'

    if not name=='AVE' and dnn_arch['ave_backbone']:
        net_name += '_pre'

    if name=='AVE':
        if not dnn_arch['ave_L2']:
            net_name += '_CS'

    if name=='FactorNet':
        net_name += '_' + dnn_arch['vsl_merge_type']

    else:
        if hard_negatives:
            net_name += '_hrd'
        else:
            if contrast_param['flip_mic'] and contrast_param['flip_img']:
                net_name = net_name + '_mix'

        if contrast_param['flip_mic'] and contrast_param['flip_img']:
            net_name = net_name + '_flip'
        elif contrast_param['flip_img']:
            net_name = net_name + '_img_' + str(contrast_param['alpha'])
        elif contrast_param['flip_mic']:
            net_name += '_mic'

    if training_param['curriculum_learning']:
        net_name += '_CL'





    # if training_param['rig_neg']:
    # net_name += '_rig'

    # if contrast_param['flip_img']:
    #     net_name = net_name + '_img_' + str(contrast_param['alpha'])

    # if contrast_param['flip_mic']:
    #     net_name += '_mic'

    # net_name += '_SM'

    file_path = os.path.join(fol_name, net_name)

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    imgs = ['train', 'val', 'test']
    for img in imgs:
        img_path = os.path.join(file_path, img)
        if not os.path.exists(img_path):
            os.mkdir(img_path)

    if print_path:
        print('network path', file_path)
        print('batch_size', training_param['batch_size'])

    return file_path, name +'_'+ net_name


proj_path = os.getcwd()

input = {
    'project_path': proj_path,
    'fps': 30,
    'sr': 48000, # 48 kHz
    'frame_len_sec': 1, # seconds
    'frame_step_train': 1, #seconds, has overlaps
    'frame_step_test': 2, #seconds, no overlaps
}

dnn_arch = {


    'net_name':'SpaceNet_V',
    # AVE, VSL, AVOL, SL2, SpaceNet, SpaceNet_split, SpaceNet_V, FactorNet, MixNet are the options

    # SL2 - AVOL with L2 distance metric
    # VSL - AVOL with cosine similarity
    # AVOL - AVOL

    #FactorNet
    #FAV_sim

    'heatmap':False, # only applies to the heatmap

    'ave_L2':True,


    'ave_backbone':False, # use pretrained backbone for AVENet

    'vsl_merge_type':'L2',

    'small_features':True, # small feature map (14x14) otherwise (28x28)
    'small_img':False,
    'FC_size':128,

    'seq_skip':True, # related to the GRU block
    'seq_res':True,

}

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

training_param = {

    'optimizer': torch.optim.Adam,
    #'criterion': nn.CrossEntropyLoss,
    'learning_rate': 0.0001, # this is used if user does not provide another lr with the parser
    'epochs': 60, # this is used if user does not provide the epoch number with the parser
    'batch_size': 16,
    'frame_len_samples': input['frame_len_sec'] * input['sr'], # number of audio samples in 2 sec

    'frame_seq': False,
    'frame_vid_samples': 5,

    'toy_params': (False, 512),

    'inference': True,
    'vid_contrast': False,

    'device': device,
    'train_binary': False,

    'curriculum_learning':True # FOR CURRICULUM LEARNING

    #'input_norm': 'freq_wise', # choose between: 'freq_wise', 'global', or None
    #'step_size':,
    #'gamma': ,
}

contrast_param = {

    'hard_negatives':False, # only consider the flip
    'flip_mic':False,
    'flip_img':False,
    'alpha':0.25,
    'rig_neg':False,
}

data_param = {

    't_image' : transforms.Compose([
        transforms.ColorJitter((0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (-0.05, 0.05)),
        # transforms.ColorJitter((0.6, 1.4), (0.6, 1.4), (0.6, 1.4), (-0.2, 0.2)),

        transforms.RandomGrayscale(0.2),
        # transforms.RandomInvert(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        ]) ,

    't_flip' : transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
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
    'fmax': 24000, #Hz
    'ref_mic_id': 5 # reference microphone (ID is 6, but we need the 5th index.)

}


filenames = {
    'net_folder_path': set_filename(name=dnn_arch['net_name'], 
                            toy_params=training_param['toy_params'],
                            print_path=True)[0],
    'train_val': 'test',
    'net_name': set_filename(name=dnn_arch['net_name'], 
                            toy_params=training_param['toy_params'],
                            print_path=True)[1]
}

