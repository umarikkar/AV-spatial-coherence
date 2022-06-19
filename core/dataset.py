#!/usr/bin/python
from email.mime import base
import glob
import os
import random

import h5py
import numpy as np
import soundfile as sf
import torch
import utils.utils as utils
from torch.utils.data import Dataset

import core.config as conf

from torchvision.transforms import transforms
from PIL import Image

# from utils import find_audio_frame_idx


#import utils.andres_salsa as andres_salsa_extraction

# from utils import generate_mel_spectrograms
# from utils import generate_gcc_spectrograms


base_path = conf.input['project_path']
fps = conf.input['fps']


def read_audio_file(sequence, train_or_test, rig, initial_time, base_path):
    # ==================== read audio file ===============================
    #sequence_path = base_path + 'data/' + train_or_test + '/' + sequence + '/bf/' + rig + '/' # beamformer case
    # sequence_path = base_path + 'data/' + train_or_test + '/' + sequence + '/' + rig + '/'

    # sequence_path = os.path.join(base_path, sequence, rig, '')

    sequence_path = os.path.join(base_path, 'data', 'RJDataset', 'audio', sequence, rig, '')


    #dir_idx = np.array(['68', '02', '07', '12', '17', '22', '27', '32', '37', '42', '47', '52', '57', '62',
    #                    '80'])  # central 15 look dir indices

    if rig == '01':
        dir_idx = np.array(['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16'])
        """
        # dir_idx = np.array(['01','02','03','04','05','06','07','08','09','10','11','12','13','15','16']) # 15 mics case
        # dir_idx = np.array(['01','02','03','04','05','06','07','08','09','10','11','12','14','16']) # 14 mics case
        # dir_idx = np.array(['01','02','03','04','05','06','07','08','09','10','11','12','16']) # 13 mics case
        # dir_idx = np.array(['01','02','03','04','05','06','07','08','09','10','11','14']) # 12 mics case
        # dir_idx = np.array(['01','02','03','04','05','06','07','08','09','10','11']) # 11 mics case
        # dir_idx = np.array(['02','03','04','05','06','07','08','09','10','14']) # 10 mics case
        # dir_idx = np.array(['02','03','04','05','06','07','08','09','10']) # 9 mics case
        # dir_idx = np.array(['02','03','04','05','07','08','09','10']) # 8 mics case
        # dir_idx = np.array(['02','03','05','06','07','09','10']) # 7 mics case
        # dir_idx = np.array(['02','03','05','07','09','10']) # 6 mics case
        # dir_idx = np.array(['03','05','06','07','09']) # 5 mics case
        #dir_idx = np.array(['03', '05', '07', '09'])  # 4 mics case
        # dir_idx = np.array(['03','06','09']) # 3 mics case
        # dir_idx = np.array(['03','09']) # 2 mics case
        """
    else:  # rig == '02'
        dir_idx = np.array(['23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38'])
        """
        # dir_idx = np.array(['23','24','25','26','27','28','29','30','31','32','33','34','35','37','38']) # 15 mics case
        # dir_idx = np.array(['23','24','25','26','27','28','29','30','31','32','33','34','36','38']) # 14 mics case
        # dir_idx = np.array(['23','24','25','26','27','28','29','30','31','32','33','34','38']) # 13 mics case
        # dir_idx = np.array(['23','24','25','26','27','28','29','30','31','32','33','36']) # 12 mics case
        # dir_idx = np.array(['23','24','25','26','27','28','29','30','31','32','33']) # 11 mics case
        # dir_idx = np.array(['24','25','26','27','28','29','30','31','32','36']) # 10 mics case
        # dir_idx = np.array(['24','25','26','27','28','29','30','31','32']) # 9 mics case
        # dir_idx = np.array(['24','25','26','27','29','30','31','32']) # 8 mics case
        # dir_idx = np.array(['24','25','27','28','29','31','32']) # 7 mics case
        # dir_idx = np.array(['24','25','27','29','31','32']) # 6 mics case
        # dir_idx = np.array(['25','27','28','29','31']) # 5 mics case
        #dir_idx = np.array(['25', '27', '29', '31'])  # 4 mics case
        # dir_idx = np.array(['25','28','31']) # 3 mics case
        # dir_idx = np.array(['25','31']) # 2 mics case
        """

    num_samples = conf.training_param['frame_len_samples']  # number of samples to be kept to produce an audio frame
    # initial audio sample
    first_audio_sample = np.int((np.round(initial_time * fps) / fps) * conf.input['sr'])

    # audio_file_list = sorted(glob.glob(sequence_path + '/*.wav'))
    audio = []
    for i in range(len(dir_idx)):

        seq = sorted(glob.glob(sequence_path + dir_idx[i] + '-*.wav'))[0]
        aud, sr = sf.read(seq)
        #aud, sr = sf.read(sequence_path + dir_idx[i] + '.wav')
        aud = aud[first_audio_sample:first_audio_sample + num_samples]
        aud = utils.pad_audio_clip(aud, num_samples) # pad in the case the extracted segment is too short
        audio.append(aud)

    audio = np.transpose(np.array(audio))
    return audio, sr


def generate_audio_tensor(audio, sr):
    ## ======================= compute log mel features ===================

    winlen = conf.logmelspectro['winlen']
    hoplen = conf.logmelspectro['hoplen']
    numcep = conf.logmelspectro['numcep']
    n_fft = conf.logmelspectro['n_fft']
    fmin = conf.logmelspectro['fmin']
    fmax = conf.logmelspectro['fmax']
    azimuth_only = conf.logmelspectro['mfcc_azimuth_only']
    # start = time.time()

    ## ----------- LOG MEL SPECTROGRAMS TENSOR -------------
    '''
    #logmel_sp = utils.generate_mel_spectrograms(audio[:, 0], sr, winlen, hoplen, numcep, n_fft, fmin, fmax)
    #tensor = np.expand_dims(logmel_sp, axis=0)
    tensor = [] # log mel tensor
    channel_num = audio.shape[1]
    for idx in range(channel_num):
        logmel_sp = utils.generate_mel_spectrograms(audio[:, idx], sr, winlen, hoplen, numcep, n_fft, fmin, fmax)
        tensor.append(logmel_sp)
    tensor = np.concatenate(tensor, axis=0) # (n_channels, timebins, freqbins)
    '''
    ## ----------- GCC SPECTROGRAMS TENSOR -------------

    tensor = [] # gcc_tensor
    channel_num = audio.shape[1]

    # use all possible pairs
    #for n in range(channel_num):
    #    for m in range(n + 1, channel_num):
    #        tensor.append(utils.generate_gcc_spectrograms(audio[:,m], audio[:,n], winlen, hoplen, numcep, n_fft))
    
    # use reference mic
    ref_mic_id = 0 #np.int(np.floor(channel_num/2))
    for n in range(channel_num):
        if not n == ref_mic_id:
            tensor.append(utils.generate_gcc_spectrograms(audio[:, n], audio[:, ref_mic_id], winlen, hoplen, numcep, n_fft))

    ## ---------- ADD mono log mel spect (1st channel only) ------------------------
    # logmel = np.expand_dims(utils.generate_mel_spectrograms(audio[:, 0], sr, winlen, hoplen, numcep, n_fft, fmin, fmax), axis =0)
    logmel = utils.generate_mel_spectrograms(audio[:, 0], sr, winlen, hoplen, numcep, n_fft, fmin, fmax)

    """
    BUG - REPORT TO DAVIDE
    """
    tensor.append(logmel)
    
    # for l in tensor:
    #     print(l.shape)

    tensor = np.concatenate(tensor, axis=0) # (n_channels, timebins, freqbins)

    ## -------------- SALSA FEATURE EXTRACTION --------------

    #tensor = salsa_ipd_extraction.extract_features(audio, conf=conf, ref_mic=1)

    #tensor = salsa_extraction.extract_features(audio, conf=conf, ref_mic=1)


    return tensor


def read_video_file(sequence, train_or_test, rig, cam_vid, initial_time, base_path):

    cam_vid = cam_vid.zfill(2)

    data_path = os.path.join(base_path, 'data', 'RJDataset')
    frame_data_path = os.path.join(data_path, 'frames')

    frame_path = os.path.join(frame_data_path, sequence, rig, 'cam-'+cam_vid)

    fps = 30
    start_time = 0.3
    end_time = 2.3
    num_frames = 10

    frame_idxs = np.linspace(round(start_time*fps), round(end_time*fps), num_frames)

    imgs = []
    for idx in frame_idxs:
        frame_name = os.path.join(frame_path, sequence +'-cam'+ cam_vid + '-frame' + str(int(idx)) + '.jpg')
        im = Image.open(frame_name)

        im = transforms.Resize((224, 224))(im)

        # augmenting the images
        if train_or_test == 'train':
            im = transforms.ColorJitter((0.6, 1.4), (0.6, 1.4), (0.6, 1.4), (-0.1, 0.1))(im)
            im = transforms.RandomGrayscale(0.2)(im)

        im = transforms.ToTensor()(im)

        imgs.append(im)

    return imgs


    # print("video: :", sequence_path_video)

    

    return None

class dataset_from_scratch(Dataset):
    def __init__(self, csv_file_path, train_or_test, normalize=False, augment=False, mean=None, std=None):

        self.csv_list = utils.csv_to_list(csv_file_path)[1:]  # [1:] is to remove first row ('name','time',etc)
        self.train_or_test = train_or_test
        self.normalize = normalize
        self.augment = augment
        self.mean = mean
        self.std = std
        if train_or_test == 'train':
            self.frame_idx_list = utils.find_audio_frame_idx(self.csv_list, conf.input['frame_step_train'])
        else:
            self.frame_idx_list = utils.find_audio_frame_idx(self.csv_list, conf.input['frame_step_test'])

    def __len__(self):
        return int(len(self.frame_idx_list) / 1)
    
    def __getitem__(self, audio_seg):


        # print(audio_seg)
        full_name = self.csv_list[self.frame_idx_list[audio_seg]][0]
        # print(full_name)
        sequence = full_name[:-6]
        # print(sequence)
        cam = np.int(full_name[-2:])
        initial_time = np.float(self.csv_list[self.frame_idx_list[audio_seg]][1])

        cam_vid = str(cam)


        if cam < 12:
            rig = '01'
        else:
            rig = '02'
            cam = cam - 11
        cam = utils.cam_one_hot(cam)
        cam = np.expand_dims(cam, axis=0)

        # read audio files
        audio, sr = read_audio_file(sequence, self.train_or_test, rig, initial_time, base_path)
        # compute log mel features and generate 15x960x64 image tensor
        tensor = generate_audio_tensor(audio, sr)

        # ## ------------------- VIDEO FILE READING HERE -----------------------------

        img_tensor = read_video_file(sequence, self.train_or_test, rig, cam_vid, initial_time, base_path)

        # ## -------------------------------------------------------------------------

        if self.normalize:
            # Normalize feature
            n_scaler_chan = self.mean.shape[0]
            # for SALSA feature, only normalize the spectrogram channels
            if n_scaler_chan < tensor.shape[0]:
                tensor[:n_scaler_chan] = (tensor[:n_scaler_chan] - self.mean) / self.std
            else:
                tensor = (tensor - self.mean) / self.std

        tensor = tensor.astype('float32')
        input_features = torch.from_numpy(tensor)


        if self.train_or_test == 'train':
            return input_features, cam, sequence, img_tensor
        else:  # == 'test
            return input_features, cam, full_name, initial_time

        

    """

    def __getitem__(self, audio_seg):


        print(audio_seg)
        full_name = self.csv_list[self.frame_idx_list[audio_seg]][0]
        print(full_name)
        sequence = full_name[:-6]
        # print(sequence)
        cam = np.int(full_name[-2:])
        initial_time = np.float(self.csv_list[self.frame_idx_list[audio_seg]][1])
        target_coords = []
        pseudo_labels = []
        sp_activity = []

        # for idx in range(self.frame_idx_list[audio_seg] - (fps * conf.input['frame_len_sec']//2),
        #                 self.frame_idx_list[audio_seg] + (fps * conf.input['frame_len_sec']//2)):
        for idx in range(self.frame_idx_list[audio_seg],
                         self.frame_idx_list[audio_seg] + (fps * conf.input['frame_len_sec'])):
            if idx >= len(self.csv_list):  # out of range i.e. the very end of malemonologue2_t2-cam22
                target_coords.append(1) # default
                pseudo_labels.append(0)  # NOT_SPEAKING
                sp_activity.append(0)  # NOT_SPEAKING
            elif self.csv_list[idx][
                0] != full_name:  # end of a seq e.g. conv1_t1-cam22 finished and conv1_t2-cam01 starts
                target_coords.append(1)
                pseudo_labels.append(0)  # NOT_SPEAKING
                # pseudo_labels.append('NOT_SPEAKING')
                sp_activity.append(0)  # NOT_SPEAKING
                # sp_activity.append('NOT_SPEAKING')
            else:
                target_coords.append(self.csv_list[idx][2])
                if self.csv_list[idx][3] == 'SPEAKING':
                    pseudo_labels.append(1)
                else:  # NOT_SPEAKING
                    pseudo_labels.append(0)
                if self.csv_list[idx][4] == 'SPEAKING':
                    sp_activity.append(1)
                else:  # NOT_SPEAKING
                    sp_activity.append(0)

        if cam < 12:
            rig = '01'
        else:
            rig = '02'
            cam = cam - 11
        cam = utils.cam_one_hot(cam)
        cam = np.expand_dims(cam, axis=0)

        # read audio files

        # print(sequence, self.train_or_test, rig, initial_time)

        audio, sr = read_audio_file(sequence, self.train_or_test, rig, initial_time, base_path)
        # compute log mel features and generate 15x960x64 image tensor
        tensor = generate_audio_tensor(audio, sr)

        if self.normalize:
            # Normalize feature
            n_scaler_chan = self.mean.shape[0]
            # for SALSA feature, only normalize the spectrogram channels
            if n_scaler_chan < tensor.shape[0]:
                tensor[:n_scaler_chan] = (tensor[:n_scaler_chan] - self.mean) / self.std
            else:
                tensor = (tensor - self.mean) / self.std

        tensor = tensor.astype('float32')
        input_features = torch.from_numpy(tensor)
        target_coords = np.asarray(target_coords).astype('float32')
        pseudo_labels = np.asarray(pseudo_labels).astype('float32')
        sp_activity = np.asarray(sp_activity).astype('float32')
        # sequence = np.asarray(list(sequence))

        # if self.train_or_test == 'train':
        #     return input_features, cam, target_coords, pseudo_labels, sp_activity, sequence
        # else:  # == 'test
        #     return input_features, cam, full_name, initial_time
        if self.train_or_test == 'train':
            return input_features, cam, sequence
        else:  # == 'test
            return input_features, cam, full_name, initial_time
        
    """


class dataset_from_hdf5(Dataset):
    def __init__(self, h5py_dir, normalize=False, augment=False, mean=None, std=None):

        self.h5_file = h5py.File(h5py_dir, 'r')
        self.normalize = normalize
        self.augment = augment
        self.mean = mean
        self.std = std

    def __len__(self):
        return int((self.h5_file['features'].shape[0]) / 1)

    def __getitem__(self, audio_seg):

        features = self.h5_file['features'][audio_seg]
        cams = self.h5_file['cams'][audio_seg]
        target_coords = self.h5_file['target_coords'][audio_seg]
        pseudo_labels = self.h5_file['pseudo_labels'][audio_seg]
        speech_activity = self.h5_file['speech_activity'][audio_seg]
        sequence = self.h5_file['sequence'][audio_seg]


        if self.normalize:
            # Normalize feature
            n_scaler_chan = self.mean.shape[0]
            # for SALSA feature, only normalize the spectrogram channels
            if n_scaler_chan < features.shape[0]:
                features[:n_scaler_chan] = (features[:n_scaler_chan] - self.mean) / self.std
            else:
                features = (features - self.mean) / self.std

        features = features.astype('float32')
        input_features = torch.from_numpy(features)

        # return input_features, cams, target_coords, pseudo_labels, speech_activity, sequence
        return input_features, cams, sequence
    