#!/usr/bin/python
import glob
import os
import pickle
import random
import shutil
from math import floor
from pathlib import Path
import ast

import cv2

import core.config as conf
import core.utils as utils
import h5py
import numpy as np
import soundfile as sf
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision.transforms import transforms

# MAKE THE DATASET FOLDERS ---------------------------------------

def MakeFolders(sequence_path, video=False):



    if not video:
        dir_1 = np.array(['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16'])
        dir_2 = np.array(['23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38'])
    else:
        dir_1 = np.array(['01','02','03','04','05','06','07','08','09','10','11'])
        dir_2 = np.array(['12','13','14','15','16','17','18','19','20','21','22'])

    all_takes = os.listdir(sequence_path)
    all_takes.pop(all_takes.index('README.txt'))

    for folder in all_takes:

        seq_dir = os.path.join(sequence_path, folder)

        # make the dirs for each rig
        os.mkdir(os.path.join(seq_dir, '01'))
        os.mkdir(os.path.join(seq_dir, '02'))
        os.mkdir(os.path.join(seq_dir, 'other'))

        for file_name in os.listdir(seq_dir):

            (end_token, idx_range) = ('.wav', range(0,2)) if not video else ('.mp4', range(-6,-4))

            if not file_name.endswith(end_token):
                pass
            else:
                if file_name[idx_range] in dir_1:
                    fol_name = '01'
                elif file_name[idx_range] in dir_2: 
                    fol_name = '02'
                else:
                    fol_name = 'other'

                file_path = os.path.join(seq_dir, file_name)
                shutil.move(file_path, os.path.join(seq_dir, fol_name, file_name))

    return


def SaveVideoFrames(data_path):

    vid_data_path = os.path.join(data_path, 'videos')
    frame_data_path = os.path.join(data_path, 'frames')

    rig_folders = ['01', '02']

    dir_1 = np.array(['01','02','03','04','05','06','07','08','09','10','11'])
    dir_2 = np.array(['12','13','14','15','16','17','18','19','20','21','22'])

    count2 = 0

    for sequence_name in os.listdir(vid_data_path):
        if not sequence_name=="README.txt":
            if sequence_name not in os.listdir(frame_data_path):
                os.mkdir(os.path.join(frame_data_path, sequence_name))
                
            for rig in rig_folders:
                
                if rig not in os.listdir(os.path.join(frame_data_path, sequence_name)):
                    os.mkdir(os.path.join(frame_data_path, sequence_name, rig))

                vid_path = os.path.join(vid_data_path, sequence_name, rig)
                    
                for cam_idx in range(11):
                
                    if rig == '01':
                        cam = dir_1[cam_idx]
                        if 'cam-'+cam not in os.listdir(os.path.join(frame_data_path, sequence_name, rig)):
                            # print(os.listdir(os.path.join(frame_data_path, sequence_name, rig)))
                            os.mkdir(os.path.join(frame_data_path, sequence_name, rig, 'cam-'+cam))
                        cam_save = cam

                    elif rig == '02':
                        cam = dir_2[cam_idx]
                        if 'cam-'+cam not in os.listdir(os.path.join(frame_data_path, sequence_name, rig)):
                            os.mkdir(os.path.join(frame_data_path, sequence_name, rig, 'cam-'+cam))
                        cam_save = cam


                    frame_path = os.path.join(frame_data_path, sequence_name, rig, 'cam-'+cam_save)
                    vid_name = sequence_name + '-cam' + cam + '.mp4'

                    vid_path_cam = os.path.join(vid_path, vid_name)
                    
                    if len(os.listdir(frame_path)) == 0:
                        print("copying and saving frames for sequence: {} rig: {} cam: {}  ".format(sequence_name, rig, cam))
                        vidcap = cv2.VideoCapture(vid_path_cam)
                        success,image = vidcap.read()
                        count = 0
                        count2 += 1

                        # print(image.shape)

                        while success:
                            image = cv2.resize(image, (612,512))
                            cv2.imwrite(os.path.join(frame_path, vid_name[:-4] + "-frame"+str(count)+".jpg"), image)     # save frame as JPEG file      
                            success,image = vidcap.read()
                            # print('Read a new frame: ', success)
                            count += 1

                        vidcap.release()

                    else:
                        pass
                        # print("already saved frames for sequence: {} rig: {} cam: {}  ".format(sequence_name, rig, cam))

    print(count2)
    return
            

# READ DATA AND GENERATE FEATURES -----------------------------------------------------------------------------------------------------------------------------------

def read_audio_file(sequence, train_or_test, rig, initial_time, base_path = conf.input['project_path'], fps=30):

    """
    returns an 16-dimensional audio vector dim=i corresponds to microphone i+1. 
    """

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


def generate_audio_tensor(audio, sr, multi_mic = conf.logmelspectro['multi_mic'], get_gcc=conf.logmelspectro['get_gcc']):

    winlen = conf.logmelspectro['winlen']
    hoplen = conf.logmelspectro['hoplen']
    numcep = conf.logmelspectro['numcep']
    n_fft = conf.logmelspectro['n_fft']
    fmin = conf.logmelspectro['fmin']
    fmax = conf.logmelspectro['fmax']
    ref_mic_id = conf.logmelspectro['ref_mic_id']

    tensor = [] # log mel tensor
    channel_num = audio.shape[1]
    
    if multi_mic:
        for n in range(channel_num):
            if n==ref_mic_id:
                logmel = utils.generate_mel_spectrograms(audio[:, n], sr, winlen, hoplen, numcep, n_fft, fmin, fmax)
                tensor.append(logmel)
            else:
                if get_gcc:
                    tensor.append(utils.generate_gcc_spectrograms(audio[:, n], audio[:, ref_mic_id], winlen, hoplen, numcep, n_fft))
                else:
                    tensor.append(utils.generate_mel_spectrograms(audio[:, n], sr, winlen, hoplen, numcep, n_fft, fmin, fmax))
       
    else:
        logmel = utils.generate_mel_spectrograms(audio[:, 0], sr, winlen, hoplen, numcep, n_fft, fmin, fmax)
        tensor.append(logmel)

    tensor = np.concatenate(tensor, axis=0) # (n_channels, timebins, freqbins)

    return tensor


def read_video_file(sequence, train_or_test, rig, cam_vid, initial_time, base_path = conf.input['project_path']):

    cam_vid = cam_vid.zfill(2)

    data_path = os.path.join(base_path, 'data', 'RJDataset')
    frame_data_path = os.path.join(data_path, 'frames')

    frame_path = os.path.join(frame_data_path, sequence, rig, 'cam-'+cam_vid)

    fps = conf.input['fps']

    period = conf.input['frame_len_sec']
    end_time = initial_time + period
    num_frames = conf.training_param['frame_vid_samples']
    frame_seq = conf.training_param['frame_seq']

    if num_frames !=1 and frame_seq:
        frame_idxs = np.linspace(round(initial_time*fps), round(end_time*fps), num_frames).astype('int')
    else:
        frame_idxs = [round(initial_time*fps)]

    f_id = []

    for idx, frame_idx in enumerate(frame_idxs):
        
        img_name = sequence +'-cam'+ cam_vid + '-frame' + str(frame_idx) + '.jpg'
        frame_name = os.path.join(frame_path, img_name)

        if img_name in os.listdir(frame_path):
            im = Image.open(frame_name)
            im = transforms.Resize((224, 224))(im)
            im = transforms.ToTensor()(im)
            f_id.append(frame_idx)

        if idx == 0:
            im2 = im.unsqueeze(0)
        else:
            im2 = torch.concat((im2, im.unsqueeze(0)), dim=0)

    return im2, f_id


# GETTING DATASET FROM SCRATCH AND HDF5 ---------------------------------------------------------------------------------------------------------------------------

class dataset_from_scratch(Dataset):
    def __init__(self, csv_file_path, train_or_test, normalize=False, augment=False, mean=None, std=None, base_path = conf.input['project_path']):

        self.csv_list = utils.csv_to_list(csv_file_path)[1:]  # [1:] is to remove first row ('name','time',etc)
        self.train_or_test = train_or_test
        self.normalize = normalize
        self.augment = augment
        self.mean = mean
        self.std = std
        self.base_path = base_path

        if train_or_test == 'train':
            self.frame_idx_list = utils.find_audio_frame_idx(self.csv_list, conf.input['frame_step_train'])
        else:
            self.frame_idx_list = utils.find_audio_frame_idx(self.csv_list, conf.input['frame_step_test'])

    def __len__(self):
        return int(len(self.frame_idx_list) / 1)
    
    def __getitem__(self, audio_seg):

        """
        CSV row looks like this:
        0:seq_name | 1:time | 2:x | 3:pseudo_label | 4:SA_total | 5:SA_male | 6:SA_female | 7:x_male | 8:y_male | 9:x_female | 10:y_female | 11:rig index
        """

        full_name = self.csv_list[self.frame_idx_list[audio_seg]][0]
        sequence = full_name[:-6]
        cam = np.int(full_name[-2:])
        initial_time = np.float(self.csv_list[self.frame_idx_list[audio_seg]][1])

        pseudo_label = self.csv_list[self.frame_idx_list[audio_seg]][3]
        speech_activity = self.csv_list[self.frame_idx_list[audio_seg]][4]

        cam_vid = str(cam)

        if cam < 12:
            rig = '01'
        else:
            rig = '02'
            cam = cam - 11

        # if cam > 2:
        #     print('hi')

        cam = utils.cam_one_hot(cam)

        

        cam = np.expand_dims(cam, axis=0)

        # read audio files
        audio, sr = read_audio_file(sequence, self.train_or_test, rig, initial_time, self.base_path)
        # compute log mel features and generate 15x960x64 image tensor
        tensor = generate_audio_tensor(audio, sr)

        # ## ------------------- VIDEO FILE READING HERE -----------------------------

        img_tensor, _ = read_video_file(sequence, self.train_or_test, rig, cam_vid, initial_time, self.base_path)

        # ## -------------------------------------------------------------------------

        # if self.train_or_test == 'test':

        #     ls_list = []

        #     frame_idxs = self.frame_idx_list[audio_seg] + frame_idxs 
        #     frame_idxs[np.where(frame_idxs >= len(self.csv_list))] = len(self.csv_list) - 1

        #     for idx in frame_idxs:

        #         ls_list.append(self.csv_list[idx])

        #     # ls = self.csv_list[frame_idxs][:]
        #     meta_male = str({
        #         'ID':'Romeo',
        #         'activity':[ls[5] for ls in ls_list],
        #         'x':[ls[7] for ls in ls_list],
        #         'y':[ls[8] for ls in ls_list],
        #     })

        #     meta_female = str({
        #         'ID':'Juliet',
        #         'activity':[ls[6] for ls in ls_list],
        #         'x':[ls[9] for ls in ls_list],
        #         'y':[ls[10] for ls in ls_list],
        #     })

        if self.train_or_test == 'test':
            ls = self.csv_list[self.frame_idx_list[audio_seg]]
            meta_male = str({
                'ID':'Romeo',
                'activity':ls[5],
                'x':ls[7],
                'y':ls[8]
            })

            meta_female = str({
                'ID':'Juliet',
                'activity':ls[6],
                'x':ls[9],
                'y':ls[10]
            })
            
        else:
            meta_male, meta_female = '', ''

        tensor = tensor.astype('float32')
        input_features = torch.from_numpy(tensor)

        return input_features, cam, img_tensor, full_name, initial_time, pseudo_label, speech_activity, meta_male, meta_female, rig


class dataset_from_hdf5(Dataset):

    def __init__(self, h5py_path, 
            normalize=True, 
            mean=None, 
            std=None, 
            seq=conf.training_param['frame_seq'], 
            t_image = transforms.Compose([]),
            t_audio = transforms.Compose([]),
            train_or_test = 'train',
        ):

        self.h5_file = h5py.File(h5py_path, 'r')
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.seq = seq 
        self.t_image = t_image
        self.t_audio = t_audio
        self.train_or_test = train_or_test

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
        pseudo_labels = self.h5_file['pseudo_labels'][audio_seg]
        speech_activity = self.h5_file['speech_activity'][audio_seg]


        rig = self.h5_file['rig'][audio_seg]

        # getting sequences only or full samples
        if not self.seq and conf.dnn_arch['net_name']=='AVE':
            rand_num = np.random.randint(imgs.shape[0])
            imgs = imgs[rand_num,:,:,:].unsqueeze(0)
        elif not self.seq:
            imgs = imgs[0,:,:,:].unsqueeze(0)


        if self.train_or_test=='test':
            # meta_male = ast.literal_eval(self.h5_file['meta_male'][audio_seg].decode("utf-8"))
            # meta_female = ast.literal_eval(self.h5_file['meta_female'][audio_seg].decode("utf-8"))

            meta_male = self.h5_file['meta_male'][audio_seg]
            meta_female = self.h5_file['meta_female'][audio_seg]

            # if not self.seq:
            #     for mp in [meta_male, meta_female]:
            #         mp['x'] = mp['x'][rand_num]
            #         mp['y'] = mp['y'][rand_num]
            #         # mp['activity'] = mp['activity'][rand_num]
                    
        else:
            meta_male, meta_female = 'not_computed', 'not_computed'

        if self.normalize:
            # 0-1 norm
            f_max = np.expand_dims(features.max(axis=(1,2)), (-2,-1))
            f_min = np.expand_dims(features.min(axis=(1,2)), (-2,-1))

            features = (features - f_min) / (f_max - f_min)

        features = features.astype('float32')
        input_features = torch.from_numpy(features)

        # if conf.training_param['frame_seq']:
        # imgs = transforms.Resize((112, 224))(imgs)

        # input_features[-1,:,:] = self.t_audio(input_features[-1,:,:].unsqueeze(0)).squeeze(0)
        # imgs = self.t_image(imgs)

        return input_features, cam, imgs, full_name, init_time, pseudo_labels, speech_activity, meta_male, meta_female, rig


# EXTRACT TRAIN/VAL/TEST DATA --------------------------------------------------------------------------------------------------------------------------------------

def get_train_val(multi_mic=conf.logmelspectro['multi_mic'], train_or_test='train', toy_params=conf.training_param['toy_params'], sequences='all', remove_silent=conf.data_param['filter_silent'],
load_idx=False
):



    """
    Function to get train and validation sets.
    """

    if type(sequences) is not list:
        sequences = (sequences).split()

    base_path = conf.input['project_path']

    frame_len = conf.input['frame_len_sec']

    mic_info = 'MC' if multi_mic else 'SC'

    if conf.training_param['frame_seq']:
        mic_info += '_seq'

    h5py_dir_str = os.path.join(base_path, 'data', 'h5py_%s_sec%d' %(mic_info, frame_len),'')

    # h5py_dir_str = os.path.join(base_path, 'data', 'h5py_%s' %mic_info,'')

    if not remove_silent:
        h5py_name = '%s_%s.h5' % (train_or_test, mic_info)
    else: 
        h5py_name = '%s_%s_sil.h5' % (train_or_test, mic_info)

    # h5py_name = '%s_%s.h5' % (train_or_test, mic_info)

    h5py_path_str = os.path.join(h5py_dir_str, h5py_name)
    h5py_path = Path(h5py_path_str)
    
    if train_or_test=='train':
        d_dataset = dataset_from_hdf5(h5py_path, 
                                    normalize=True, 
                                    train_or_test='train'
                                    # t_audio=conf.data_param['t_audio'], 
                                    # t_image=conf.data_param['t_image'],
                                    )

        if toy_params[0]:
            sz_train = toy_params[1]
            sz_val = int(0.2 * (toy_params[1] / 0.8))

            rand_toy = list(range(sz_train + sz_val))
            random.shuffle(rand_toy)
            d_dataset = Subset(d_dataset, rand_toy)

    else:
        d_dataset = dataset_from_hdf5(h5py_path, 
                            normalize=True, 
                            train_or_test='test'
                            )
        if sequences != ['all']:
            idx_list = []
            for (idx, data) in enumerate(d_dataset):
                word = data[3].decode("utf-8")
                word = word[:word.find('-cam')]
                if word in sequences:
                    idx_list.append(idx)
                
            d_dataset = Subset(d_dataset, idx_list)

    return d_dataset



