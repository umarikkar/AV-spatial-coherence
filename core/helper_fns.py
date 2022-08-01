
import numpy as np
import os
import cv2

import torch.nn as nn
from fn_networks import MergeNet, AVOL_Net, AVE_Net, AVE_Net_temporal, AVOL_Net_temporal ,AVE_Net2, AVOL_Net_v2

from fn_nets_v2 import AVOL, EZ_VSL, AVE
from fn_trainer import loss_AVE, loss_AVOL, loss_VSL

import core.config as conf


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
            

# ------ setting the network -----------------------------------------------------------------------
def set_network(set_train=True):
    if conf.dnn_arch['AVOL']:
        net = AVOL(set_train=set_train)
        loss_fn = loss_AVOL()
    elif conf.dnn_arch['AVE']:
        net = AVE(set_train=True)
        loss_fn = loss_AVE()
    elif conf.dnn_arch['EZ_VSL']:
        net = EZ_VSL(set_train=set_train)
        loss_fn = loss_VSL()
    else:
        net = MergeNet()
        loss_fn = nn.BCELoss() if conf.dnn_arch['heatmap'] else nn.CrossEntropyLoss()

    print('net: ', type(net), '\nloss: ', loss_fn)

    return net, loss_fn