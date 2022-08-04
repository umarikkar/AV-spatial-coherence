#!/usr/bin/python
import glob
import os
import pickle
import random
from math import floor
from pathlib import Path

import core.config as conf
import core.utils as utils
import h5py
import numpy as np
import soundfile as sf
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision.transforms import transforms

# MAKE THE DATASET FOLDERS

def MakeFolders(sequence_path, video=False):

    import shutil

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

def read_audio_file(sequence, train_or_test, rig, initial_time, base_path = conf.input['project_path']):

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


def read_video_file(sequence, train_or_test, rig, cam_vid, initial_time, base_path = conf.input['project_path']):

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

        full_name = self.csv_list[self.frame_idx_list[audio_seg]][0]
        sequence = full_name[:-6]
        cam = np.int(full_name[-2:])
        initial_time = np.float(self.csv_list[self.frame_idx_list[audio_seg]][1])

        pseudo_label = self.csv_list[self.frame_idx_list[audio_seg]][-2]
        speech_activity = self.csv_list[self.frame_idx_list[audio_seg]][-1]

        cam_vid = str(cam)

        if cam < 12:
            rig = '01'
        else:
            rig = '02'
            cam = cam - 11
        cam = utils.cam_one_hot(cam)
        cam = np.expand_dims(cam, axis=0)

        # read audio files
        audio, sr = read_audio_file(sequence, self.train_or_test, rig, initial_time, self.base_path)
        # compute log mel features and generate 15x960x64 image tensor
        tensor = generate_audio_tensor(audio, sr)

        # ## ------------------- VIDEO FILE READING HERE -----------------------------

        img_tensor = read_video_file(sequence, self.train_or_test, rig, cam_vid, initial_time, self.base_path)

        # ## -------------------------------------------------------------------------

        tensor = tensor.astype('float32')
        input_features = torch.from_numpy(tensor)

        return input_features, cam, img_tensor, full_name, initial_time, pseudo_label, speech_activity


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
        pseudo_labels = self.h5_file['pseudo_labels'][audio_seg]
        speech_activity = self.h5_file['speech_activity'][audio_seg]

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

        # if conf.training_param['frame_seq']:
        # imgs = transforms.Resize((112, 224))(imgs)

        # input_features[-1,:,:] = self.t_audio(input_features[-1,:,:].unsqueeze(0)).squeeze(0)
        # imgs = self.t_image(imgs)

        return input_features, cam, imgs, full_name, init_time, pseudo_labels, speech_activity


# EXTRACT TRAIN/VAL/TEST DATA --------------------------------------------------------------------------------------------------------------------------------------

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


# CREATE SAMPLES AND LABELS

def create_samples(data, 
    device=conf.training_param['device'], 
    augment=False, 
    contrast_vid=conf.training_param['vid_contrast'], 
    t_image=conf.data_param['t_image'], 
    create_contrast=True,
    filter_silent=conf.data_param['filter_silent']):

    def rotate(l, n): # rotate sequence list
        return l[n:] + l[:n]

    batch_size = data[0].shape[0]

    audio = data[0]
    cam = data[1]
    all_frames = data[2]
    seq_pos = data[3]
    speaking = data[6]

    if conf.training_param['frame_seq']:
        imgs_pos = all_frames
    else:
        imgs_pos = all_frames.squeeze(1)

    if augment:
        for i in range(batch_size):
            imgs_pos[i] = t_image(imgs_pos[i])
    
    
    if create_contrast:
        rem_count = 0
        # create contrastive batch (shift by some n)
        if batch_size > 1:
            roll_idx = random.randint(1, batch_size-1)
        else:
            roll_idx = 1

        imgs_neg = torch.roll(imgs_pos, roll_idx, dims=0)
        cam_neg = torch.roll(cam, roll_idx, dims=0)
        seq_neg = rotate(seq_pos, roll_idx)

        imgs_all = torch.concat((imgs_pos, imgs_neg), dim=0)
        audio_all = torch.concat((audio, audio), dim=0)
        cam_all = torch.concat((cam, cam_neg), dim=0)
        seq_all = seq_pos

        # choosing which indexes to drop for contrastive learning.
        for idx, seq in enumerate(seq_neg):

            seq1 = seq
            seq2 = seq_pos[idx]

            if filter_silent:

                if speaking[idx] == b'NOT_SPEAKING':

                    idx_rem = idx - rem_count

                    imgs_all = imgs_all[torch.arange(imgs_all.size(0))!=idx_rem] 
                    audio_all = audio_all[torch.arange(audio_all.size(0))!=idx_rem] 
                    cam_all = cam_all[torch.arange(cam_all.size(0))!=idx_rem]

                    rem_count +=1

                else:
                    if contrast_vid:
                        seq1 = seq1.decode("utf-8")
                        seq1 = seq1[:seq1.find('_t')]

                        seq2 = seq2.decode("utf-8")
                        seq2 = seq2[:seq2.find('_t')]

                    if seq1 == seq2:
                        idx_rem = batch_size + idx - rem_count

                        imgs_all = imgs_all[torch.arange(imgs_all.size(0))!=idx_rem] 
                        audio_all = audio_all[torch.arange(audio_all.size(0))!=idx_rem] 
                        cam_all = cam_all[torch.arange(cam_all.size(0))!=idx_rem] 

                        rem_count +=1
                    else:
                        seq_all.append(seq)

            else:
                if contrast_vid:
                    seq1 = seq1.decode("utf-8")
                    seq1 = seq1[:seq1.find('_t')]

                    seq2 = seq2.decode("utf-8")
                    seq2 = seq2[:seq2.find('_t')]


                if seq1 == seq2:
                    idx_rem = batch_size + idx - rem_count

                    imgs_all = imgs_all[torch.arange(imgs_all.size(0))!=idx_rem] 
                    audio_all = audio_all[torch.arange(audio_all.size(0))!=idx_rem] 
                    cam_all = cam_all[torch.arange(cam_all.size(0))!=idx_rem] 

                    rem_count +=1
                else:
                    seq_all.append(seq)

    else:
        cam_all = cam
        imgs_all = imgs_pos
        audio_all = audio
        seq_all = data[3]

        rem_count = torch.ones((batch_size, batch_size)).to(device=device)

        for idx1, seq1 in enumerate(seq_all):

            if filter_silent:

                if speaking[idx1] == b'NOT_SPEAKING':

                    rem_count[idx1, :] = 0
                    rem_count[:, idx1] = 0

                else:
                    for idx2, seq2 in enumerate(seq_all):
                        if idx1 != idx2:
                            if seq1 == seq2:
                                rem_count[idx1, idx2] = 0 

            else:
                for idx2, seq2 in enumerate(seq_all):
                    if idx1 != idx2:
                        if seq1 == seq2:
                            rem_count[idx1, idx2] = 0


    imgs_all = imgs_all.to(device=device)
    audio_all = audio_all.to(device=device)
    cam_all = cam_all.to(device=device)

    return imgs_all, audio_all, cam_all, seq_all, rem_count


def create_labels(BS, rem_count, device='cpu', heatmap=conf.dnn_arch['heatmap']):

    neg_BS = BS - rem_count

    if heatmap or conf.dnn_arch['AVOL']:
        labels_pos = torch.ones(BS).to(device=device)
        labels_neg = torch.zeros(neg_BS).to(device=device)
        labels_all = torch.concat((labels_pos, labels_neg), dim=0).to(device=device)
        
    else:
        one = torch.ones((BS,1))
        zer = torch.zeros((neg_BS,1))

        labels_pos = torch.concat((one, zer), dim=0)
        labels_neg = (-1.0)*labels_pos + 1.0

        labels_neg = torch.concat((zer, one), dim=0)
        labels_all = torch.concat((labels_pos, labels_neg), dim=-1).to(device=device)

    return labels_all
