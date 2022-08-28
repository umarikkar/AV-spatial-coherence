#!/usr/bin/python
import glob
import os
import pickle
import random
import shutil
from math import floor
from pathlib import Path
import ast

import matplotlib.pyplot as plt

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

# WORKING WITH SAMPLES ---------------------------------------------------------------------------------------------------------------------------------------------

def filter_silent(data):

    """
    filters out the silent samples and returns the reduced data batch.
    """

    speaking = data[6]

    speak_idxs = [i for i, x in enumerate(speaking) if x == b'SPEAKING']

    if not len(speak_idxs)==0:
        data_new = []
        for dat in data:
            if type(dat) is list:
                dat_new = []
                for idx in speak_idxs:
                    dat_new.append(dat[idx])
            else:
                dat_new = dat[speak_idxs]

            data_new.append(dat_new)

    else:
        data_new = None

    return data_new

def filter_sequences(data, seq_filter=['conversation', 'interactive'], bs_max=16):

    """
    filters out the unnecessary sequences.
    """

    seq_all = data[3]

    seq_all = [x.decode('utf-8')[:-10] for x in seq_all]

    seq_idxs_all = [i for i, x in enumerate(seq_all) if x not in seq_filter]
    
    seq_idxs = seq_idxs_all[:bs_max] if len(seq_idxs_all) > bs_max else seq_idxs_all

    if not len(seq_idxs)==0:
        data_new = []
        for dat in data:
            if type(dat) is list:
                dat_new = []
                for idx in seq_idxs:
                    dat_new.append(dat[idx])
            else:
                dat_new = dat[seq_idxs]

            data_new.append(dat_new)

    else:
        data_new = None

    return data_new




def create_neg_mask(data,
            remove_cam = True,
            create_double = False,
            contrast_param = conf.contrast_param,
            curriculum_neg=False,
            device=conf.training_param['device'],
            vid_contrast=False,
            ):

    """
    removes from the batch, contrastive pairs which contain same camera indexes.
    creates the negative mask, with or without the hard negatives.
    """

    seq_all = data[3]
    rig = data[-1]

    def remove_string(neg_mask, seq_all, rig, remove_rig=True):
        for idx1, seq1 in enumerate(seq_all):
            for idx2, seq2 in enumerate(seq_all):
                if idx1 != idx2:
                    if seq1==seq2:
                        neg_mask[idx1, idx2] = 0
                    if remove_rig and rig[idx1]==rig[idx2]:
                        neg_mask[idx1, idx2] = 0
             
        return neg_mask

    bs = len(seq_all)

    if contrast_param['hard_negatives']:
        neg_mask = torch.eye(bs)
    else:
        neg_mask = torch.ones((bs, bs))

    rig_neg = conf.contrast_param['rig_neg']

    zer = torch.zeros((bs, bs))
    ide = torch.eye(bs)

    if not curriculum_neg and remove_cam:
        neg_mask = remove_string(neg_mask, seq_all, rig, remove_rig=rig_neg)
    elif curriculum_neg:
        seq_all = [x.decode('utf-8')[:-10] for x in seq_all]
        neg_mask = remove_string(neg_mask, seq_all, rig, remove_rig=rig_neg)

    if vid_contrast:
        seq_all = [x.decode('utf-8')[:-6] for x in seq_all]
        neg_mask = remove_string(neg_mask, seq_all, rig, remove_rig=rig_neg)

    if create_double:
        if contrast_param['hard_negatives']: # the weird negatives only
            if contrast_param['flip_mic'] and contrast_param['flip_img']:
                neg_mask = torch.cat((torch.cat((neg_mask, ide), dim=1), torch.cat((ide, ide), dim=1)), dim=0)
            elif contrast_param['flip_img']:
                neg_mask = torch.cat((torch.cat((neg_mask, zer), dim=1), torch.cat((ide, ide), dim=1)), dim=0)
            elif contrast_param['flip_mic']:
                neg_mask = torch.cat((torch.cat((neg_mask, ide), dim=1), torch.cat((zer, ide), dim=1)), dim=0)
        else:
            neg_mask = torch.cat((torch.cat((neg_mask, zer), dim=1), torch.cat((zer, neg_mask), dim=1)), dim=0)

    neg_mask = neg_mask.to(device=device)
    return neg_mask


def create_contrast(data):

    def rotate(l, n): # rotate sequence list
        return l[n:] + l[:n]

    batch_size = data[0].shape[0]

    audio = data[0]
    cam = data[1]
    all_frames  = data[2]
    seq_all = data[3]

    # create contrastive batch (shift by some n)
    if batch_size > 1:
        roll_idx = random.randint(1, batch_size-1)
    else:
        roll_idx = 1

    imgs_pos = all_frames*1

    imgs_neg = torch.roll(imgs_pos, roll_idx, dims=0)
    cam_neg = torch.roll(cam, roll_idx, dims=0)
    seq_neg = rotate(seq_all, batch_size - roll_idx)

    imgs_all = torch.concat((imgs_pos, imgs_neg), dim=0)
    audio_all = torch.concat((audio, audio), dim=0)
    cam_all = torch.concat((cam, cam_neg), dim=0)
    seq_all.extend(seq_neg)

    return audio_all, cam_all, imgs_all, seq_all



def flip_imgs(imgs, cams, t_image=conf.data_param['t_image'], t_flip=conf.data_param['t_flip']):

    """
    augment and flips
    Flips the image horizontally as to remove the spatial alignment. Note that the studio is also flipped.
    Therefore the network might learn the studio flip only.
    """

    imgs_flipped = imgs*1.0
    imgs_flipped = t_flip(imgs_flipped)

    imgs = torch.concat((imgs, imgs_flipped), dim=0)

    for i, _ in enumerate(imgs):
        imgs[i] = t_image(imgs[i])
        

    cams_flipped = torch.concat((torch.flip(cams[:,:,:3], dims=(-1,)), torch.flip(cams[:,:,3:8], 
                                dims=(-1,)), torch.flip(cams[:,:,8:], dims=(-1,))), dim=-1)
    cams = torch.concat((cams, cams_flipped), dim=0)

    return imgs, cams

def gcc_collapse(aud, c_bot=None, c_top=None):

    if c_bot is None:
        c_bot = [0,1,2,3,4,6,7,8,9,10]

    if c_top is None:
        c_top = [11,12,13,14,15]

    aud_ref = aud[:,5].unsqueeze(1)
    aud_bot = aud[:,c_bot]
    aud_top = aud[:,c_top]

    aud_gcc = torch.cat((aud_bot, aud_top), dim=1)
    aud_gcc = torch.amax(aud_gcc, dim=-2)
    aud_gcc = aud_gcc[:,:,15:45]

    """
    # # aud_gcc_mean = aud_gcc.mean(dim=-2)

    # # aud_gcc_max2 = aud_gcc_max - aud_gcc_max.mean(dim=1)
    # a = aud_gcc[0].cpu().detach().numpy()
    # # b = aud_gccnew[0].cpu().detach().numpy()

    # # b = aud_gcc_max[0].cpu().detach().numpy()
    # # c = aud_gcc_max2[0].cpu().detach().numpy()
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(a, aspect='auto')
    # plt.colorbar()
    # plt.subplot(122)
    # a[a < 0.8] = 0
    # plt.imshow(a, aspect='auto')
    # # plt.subplot(133)
    # # plt.imshow(c, aspect='auto')
    # plt.colorbar()
    # plt.show()
    """

    return aud_ref, aud_gcc


def flip_mics(audio,
                return_all=True):

    """
    shift the microphone indexes by a random amount, making it a 'hard' negative. 

    The initial microphones are physcially placed as:
                    '12','13','14','15','16'
    '01','02','03','04','05','06-ref','07','08','09','10','11'.

    if ref_mic is 'k' we disregard the k'th channel index and shift the others. (For now, k=5 (index))
    so idx of bottom channels: 0-9, and top channels: 10-14

    As the default case, we shift the bottom mic indexes by 2 times the top mic indexes, but in the same direction.
    Random horizontal flip of mics is also applied.

    """
    mic_top = torch.arange(11, 16)
    mic_bot = torch.arange(0, 11)

    aud_top = audio[:,mic_top]
    aud_bot = audio[:,mic_bot]

    aud_top, aud_bot = torch.flip(aud_top, dims=(1,)), torch.flip(aud_bot, dims=(1,))

    audio_aug = torch.cat((aud_bot, aud_top), dim=1)

    if return_all:
        return torch.cat((audio, audio_aug), dim=0)
    else:
        return audio_aug


def create_samples(data, 
    device=conf.training_param['device'], 
    augment=False,

    vid_contrast=False,

    data_param=conf.data_param, 
    remove_silent=conf.data_param['filter_silent'],
    contrast_samples=False,

    remove_cam = True,
    return_mask = True,

    contrast_param = conf.contrast_param,
    curriculum_setting = None,

    train_or_test = 'train'):

    if train_or_test == 'train':
        hard_negatives = contrast_param['hard_negatives']
        flip_img = contrast_param['flip_img']
        flip_mic = contrast_param['flip_mic']
    else:
        hard_negatives = flip_mic = flip_img = False

    t_image = data_param['t_image']
    
    no_batch_flag = False

    # # let us remove all the non-speaking samples.
    # if remove_silent:
    #     data = filter_silent(data)
    #     no_batch_flag = True if data is None else False

    if curriculum_setting is not None:
        seq_remove = curriculum_setting['seq_remove']
        seq_neg = curriculum_setting['neg']
        data = filter_sequences(data, seq_remove)
        no_batch_flag = True if data is None else False
    else:
        seq_neg = False

    if no_batch_flag:
        return None

    else:
        # remove the same cam negative samples and do the other nice stuff.

        if contrast_samples:
            audio, cam, all_frames, seq_all = create_contrast(data)
        else:
            audio = data[0]
            cam = data[1]
            all_frames = data[2]
            seq_all = data[3]

        bs = audio.shape[0]
    
        if conf.training_param['frame_seq']:
            imgs_all = all_frames
        else:
            imgs_all = all_frames.squeeze(1)

        imgs_all = imgs_all.to(device=device)
        audio_all = audio.to(device=device)
        cam_all = cam.to(device=device)

        if flip_img:
            imgs_all, cam_all = flip_imgs(imgs_all, cam_all)
        else:
            if augment:
                for i,_ in enumerate(imgs_all):
                    imgs_all[i] = t_image(imgs_all[i])

        if conf.dnn_arch['small_img']:
            imgs_all = torch.cat([transforms.Resize((112, 112))(im).unsqueeze(0) for im in imgs_all])

        if flip_mic:
            audio_all = flip_mics(audio_all)

        if return_mask:
            create_double = True if hard_negatives or flip_img or flip_mic else False
            neg_mask = create_neg_mask(data, 
                            remove_cam=remove_cam, 
                            create_double=create_double,
                            curriculum_neg=seq_neg,
                            vid_contrast=vid_contrast)
        else:
            neg_mask = None

        # let us now convert the string arrays for male and female, as dict and append to output dictionary
        if train_or_test == 'test':

            meta_all = [data[-3], data[-2]]
            meta_all = [[ast.literal_eval(i.decode("utf-8")) for i in d] for d in meta_all]
            original_res = np.array([2048, 2448])
            final_res = np.array([all_frames[0].shape[-2], all_frames[0].shape[-1]])

            r = final_res / original_res
            
            for meta_person in meta_all:
                for d in meta_person:
                    d['x'] = int(float(d['x'])*r[0])
                    d['y'] = int(float(d['y'])*r[1])

            meta_male, meta_female = meta_all[0], meta_all[1]

        else:
            meta_male, meta_female = data[-3], data[-2]

        rig = [int(r) for r in data[-1]]


        samples = {
            'imgs_all':imgs_all,
            'audio_all':audio_all,
            'cam_all':cam_all,
            'seq_all':seq_all,
            'neg_mask':neg_mask,
            'raw_batch':bs,
            'meta_male':meta_male,
            'meta_female':meta_female,
            'rig':rig,
        }
            
        return samples


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
