#!/usr/bin/python

import numpy as np
import torch
import torch.nn as nn

import os

#mean_vector = torch.load(input['project_path'] + 'output/test/mean_vector.pt')
#std_vector = torch.load(input['project_path'] + 'output/test/std_vector.pt')

proj_path = os.getcwd()

input = {
    'project_path': proj_path,
    'fps': 30,
    'sr': 48000, # 48 kHz
    'frame_len_sec': 2, # seconds
    'frame_step_train': 1, #seconds
    'frame_step_test': 2, #seconds

}

beamformer = {
    'num_look_dir': 89,
    'mat_file': 'SD_BeamformingWeights.mat', # mat file containing BF weights
}

training_param = {
    'optimizer': torch.optim.Adam,
    #'criterion': nn.CrossEntropyLoss,
    'learning_rate': 0.0001, # this is used if user does not provide another lr with the parser
    'epochs': 50, # this is used if user does not provide the epoch number with the parser
    'batch_size': 32,
    'frame_len_samples': input['frame_len_sec'] * input['sr'], # number of audio samples in 2 sec
    'frame_vid_samples': 1,

    #'input_norm': 'freq_wise', # choose between: 'freq_wise', 'global', or None
    #'step_size':,
    #'gamma': ,
}

logmelspectro = {
    'get_gcc':  True,
    'mfcc_azimuth_only': True, # False for using all the 89 look dir, True only the 15 central-horizontal look dir
    'winlen': 512, # samples
    # 'winlen': 1024,
    'hoplen': 100, # samples
    'numcep': 64, # number of cepstrum bins to return
    'n_fft': 512, #fft lenght
    'fmin': 40, #Hz
    'fmax': 24000 #Hz
}

