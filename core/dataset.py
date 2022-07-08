#!/usr/bin/python
import glob
import os
import random
from math import floor
from pathlib import Path

import h5py
import numpy as np
import soundfile as sf
import torch
import utils.utils as utils
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision.transforms import transforms

import pickle

import core.config as conf

# from utils import find_audio_frame_idx


#import utils.andres_salsa as andres_salsa_extraction

# from utils import generate_mel_spectrograms
# from utils import generate_gcc_spectrograms


base_path = conf.input['project_path']
fps = conf.input['fps']

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def read_audio_file(sequence, train_or_test, rig, initial_time, base_path):

    sequence_path = os.path.join(base_path, 'data', 'RJDataset', 'audio', sequence, rig, '')

    if conf.logmelspectro['multi_mic']:
        if rig == '01':
            dir_idx = np.array(['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16'])

        else:  # rig == '02'
            dir_idx = np.array(['23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38'])

    else:
        if rig == '01':
            dir_idx = np.array(['01'])
            
        else:  # rig == '02'
            dir_idx = np.array(['23'])

    num_samples = conf.training_param['frame_len_samples']  # number of samples to be kept to produce an audio frame
    # initial audio sample
    first_audio_sample = np.int((np.round(initial_time * fps) / fps) * conf.input['sr'])

    audio = []
    for i in range(len(dir_idx)):

        seq = sorted(glob.glob(sequence_path + dir_idx[i] + '-*.wav'))[0]
        aud, sr = sf.read(seq)
        aud = aud[first_audio_sample:first_audio_sample + num_samples]
        aud = utils.pad_audio_clip(aud, num_samples) # pad in the case the extracted segment is too short
        audio.append(aud)

    audio = np.transpose(np.array(audio))
    return audio, sr

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def generate_audio_tensor(audio, sr, multi_mic = conf.logmelspectro['multi_mic'], get_gcc=conf.logmelspectro['get_gcc']):

    winlen = conf.logmelspectro['winlen']
    hoplen = conf.logmelspectro['hoplen']
    numcep = conf.logmelspectro['numcep']
    n_fft = conf.logmelspectro['n_fft']
    fmin = conf.logmelspectro['fmin']
    fmax = conf.logmelspectro['fmax']

    tensor = [] # log mel tensor
    channel_num = audio.shape[1]
    
    ref_mic_id = 0

    if multi_mic:
        for n in range(channel_num):
            if not n==ref_mic_id:
                if get_gcc:
                    tensor.append(utils.generate_gcc_spectrograms(audio[:, n], audio[:, ref_mic_id], winlen, hoplen, numcep, n_fft))
                else:
                    tensor.append(utils.generate_mel_spectrograms(audio[:, n], sr, winlen, hoplen, numcep, n_fft, fmin, fmax))
    else:
        pass

    logmel = utils.generate_mel_spectrograms(audio[:, ref_mic_id], sr, winlen, hoplen, numcep, n_fft, fmin, fmax)
    tensor.append(logmel)
        
    tensor = np.concatenate(tensor, axis=0) # (n_channels, timebins, freqbins)

    return tensor

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def read_video_file(sequence, train_or_test, rig, cam_vid, initial_time, base_path):

    cam_vid = cam_vid.zfill(2)

    data_path = os.path.join(base_path, 'data', 'RJDataset')
    frame_data_path = os.path.join(data_path, 'frames')

    frame_path = os.path.join(frame_data_path, sequence, rig, 'cam-'+cam_vid)

    fps = conf.input['fps']

    period = conf.input['frame_len_sec']
    end_time = initial_time + period
    num_frames = conf.training_param['frame_vid_samples']

    if num_frames !=1:
        frame_idxs = np.linspace(round(initial_time*fps), round(end_time*fps), num_frames)
    else:
        frame_idxs = [round(initial_time*fps)]

    for idx, frame_idx in enumerate(frame_idxs):
        
        img_name = sequence +'-cam'+ cam_vid + '-frame' + str(int(frame_idx)) + '.jpg'
        frame_name = os.path.join(frame_path, img_name)

        if img_name in os.listdir(frame_path):
            im = Image.open(frame_name)
            im = transforms.Resize((224, 224))(im)
            im = transforms.ToTensor()(im)

        if idx == 0:
            im2 = im.unsqueeze(0)
        else:
            im2 = torch.concat((im2, im.unsqueeze(0)), dim=0)

    return im2


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

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

        tensor = tensor.astype('float32')
        input_features = torch.from_numpy(tensor)

        return input_features, cam, img_tensor, full_name, initial_time

        
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class dataset_from_hdf5(Dataset):

    def __init__(self, h5py_path, 
            normalize=True, 
            mean=None, 
            std=None, 
            seq=conf.training_param['frame_seq'], 
            t_image = transforms.Compose([]),
            t_audio = transforms.Compose([]),
        ):

        self.h5_file = h5py.File(h5py_path, 'r')
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.seq = seq 
        self.t_image = t_image
        self.t_audio = t_audio

    def __len__(self):
        return int((self.h5_file['features'].shape[0]) / 1)

    def __getitem__(self, audio_seg):

        # actual data
        features = self.h5_file['features'][audio_seg]
        cam = torch.from_numpy(self.h5_file['cams'][audio_seg])
        imgs = torch.from_numpy(self.h5_file['img_frames'][audio_seg])

        # metadata
        full_name = self.h5_file['sequence'][audio_seg]
        init_time = self.h5_file['initial_time'][audio_seg]

        # getting sequences only or full samples
        if not self.seq:
            rand_num = np.random.randint(imgs.shape[0])
            imgs = imgs[rand_num,:,:,:].unsqueeze(0)

        if self.normalize:
            # 0-1 norm
            f_max = np.expand_dims(features.max(axis=(1,2)), (-2,-1))
            f_min = np.expand_dims(features.min(axis=(1,2)), (-2,-1))

            features = (features - f_min) / (f_max - f_min)

        features = features.astype('float32')
        input_features = torch.from_numpy(features)

        # input_features[-1,:,:] = self.t_audio(input_features[-1,:,:].unsqueeze(0)).squeeze(0)
        # imgs = self.t_image(imgs)

        return input_features, cam, imgs, full_name, init_time
    

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def get_train_val(multi_mic=conf.logmelspectro['multi_mic'], train_or_test='train', toy_params=conf.training_param['toy_params'], sequences='all'):
    """
    Function to get train and validation sets. 
    Precomputed indexes are available for the full dataset. 
    Else, we need to compute the indexes randomly.
    """
    base_path = conf.input['project_path']


    mic_info = 'MC_seq' if multi_mic else 'SC_seq'

    h5py_dir_str = os.path.join(base_path, 'data', 'h5py_%s' %mic_info,'')
    h5py_name = '%s_%s.h5' % (train_or_test, mic_info)

    h5py_path_str = os.path.join(h5py_dir_str, h5py_name)
    h5py_path = Path(h5py_path_str)
    
    if train_or_test=='train':
        d_dataset = dataset_from_hdf5(h5py_path, 
                                    normalize=True, 
                                    # t_audio=conf.data_param['t_audio'], 
                                    # t_image=conf.data_param['t_image'],
                                    )
    else:
        d_dataset = dataset_from_hdf5(h5py_path, 
                            normalize=True, 
                            )
        if sequences != 'all':
            idx_list = []
            for (idx, data) in enumerate(d_dataset):
                # print(data[3])
                word = data[3].decode("utf-8")
                word = word[:word.find('-cam')]
                if word in sequences:
                    idx_list.append(idx)
                
            d_dataset = Subset(d_dataset, idx_list)

    # DATA LOADER INITIALISATION -----------------------------------------------------------------------------
    load_idxs= False

    if toy_params[0]:
        sz_train = toy_params[1]
        sz_val = int(0.2 * (toy_params[1] / 0.8))

        rand_toy = list(range(sz_train + sz_val))
        random.shuffle(rand_toy)
        d_dataset = Subset(d_dataset, rand_toy)
    
    elif sequences == 'all':
        file_train = os.path.join(os.getcwd(), 'results','indexes', 'idxs_train.pkl')
        file_val = os.path.join(os.getcwd(), 'results','indexes', 'idxs_val.pkl')

        files = [file_train, file_val]

        load_idxs = True
        
        for i,f in enumerate(files):
            if not os.path.exists(f):
                print('making new indexes')
                load_idxs = False
                break
            else:
                print('getting pre-computed indexes')
                open_file = open(f, "rb")
                if i==0:
                    train_idxs = pickle.load(open_file)
                else:
                    val_idxs = pickle.load(open_file)
                open_file.close()
    else:
        pass


    if not load_idxs:

        rand_idxs = list(range(len(d_dataset)))
        random.shuffle(rand_idxs)

        train_size = floor(0.8*len(d_dataset))
        train_idxs = rand_idxs[0:train_size]
        val_idxs = rand_idxs[train_size:]

        file_train = os.path.join(conf.filenames['net_folder_path'], 'idxs_train.pkl')
        file_val = os.path.join(conf.filenames['net_folder_path'], 'idxs_val.pkl')

        open_file = open(file_train, "wb")
        pickle.dump(train_idxs, open_file)
        open_file.close()

        open_file = open(file_val, "wb")
        pickle.dump(val_idxs, open_file)
        open_file.close()

    data_train = Subset(d_dataset, train_idxs)
    data_val = Subset(d_dataset, val_idxs)

    return data_train, data_val, d_dataset
