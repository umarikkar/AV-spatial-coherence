#!/usr/bin/python
import torch
import os
import h5py, argparse
from pathlib import Path
import core.config as conf
from fn.dataset import dataset_from_hdf5, dataset_from_scratch
from torch.utils.data import DataLoader
import core.utils as utils

from tqdm import tqdm
import matplotlib.pyplot as plt


base_path = conf.input['project_path']

train_or_test = 'test'

if train_or_test == 'train':
    csv_file = os.path.join(base_path, 'data', 'csv', train_or_test + '.csv')
elif train_or_test=='test':
    csv_file = os.path.join(base_path, 'data', 'csv', train_or_test + '_new.csv')


def main():

    frame_len = conf.input['frame_len_sec']

    h5py_dir_str = os.path.join(base_path, 'data', 'h5py_%s_sec%d' %(args.info, frame_len),'')
    h5py_dir = Path(h5py_dir_str)
    h5py_dir.mkdir(exist_ok=True, parents=True)

    h5py_name = '%s_%s.h5' % (train_or_test, (args.info))

    print(h5py_name)

    h5py_path_str = os.path.join(h5py_dir_str, h5py_name)
    h5py_path = Path(h5py_path_str)

    if create_hdf5:
        ## ---------- Data loaders -----------------

        if h5py_name in os.listdir(h5py_dir_str):
            os.remove(h5py_path_str)

        f = h5py.File(h5py_path, 'a')

        print(h5py_path)

        d_dataset = dataset_from_scratch(csv_file, train_or_test)

        total_size = len(d_dataset)

        data_loader = DataLoader(d_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        # data_loader = DataLoader(d_dataset, batch_size=1, shuffle=False)

        count = 0
        for data in  tqdm(data_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', leave=None):

            # print('count: {}/{}'.format('%s'%count, '%s'%total_size))

            audio = data[0]
            cam = data[1]
            all_frames = data[2]
            sequence = data[3]
            initial_time = data[4]
            pseudo_labels = data[5]
            speech_activity = data[6]
            meta_male = data[7]
            meta_female = data[8]

            #print(count)
            if count == 0:
                # Create the dataset at first
                f.create_dataset('features', data=audio, chunks=True, maxshape=(None, None, None, 64))
                f.create_dataset('cams', data=cam, chunks=True, maxshape=(None, 1, 11))
                f.create_dataset('img_frames', data=all_frames, chunks=True, maxshape=(None, None, 3, 224, 224))
                              
                f.create_dataset('initial_time', data=initial_time, maxshape=(None,), chunks=True)

                f.create_dataset('sequence', data=sequence, maxshape=(None,), chunks=True)  
                f.create_dataset('pseudo_labels', data=pseudo_labels, maxshape=(None,), chunks=True)
                f.create_dataset('speech_activity', data=speech_activity, maxshape=(None,), chunks=True)

                f.create_dataset('meta_male', data=meta_male, maxshape=(None,), chunks=True)
                f.create_dataset('meta_female', data=meta_female, maxshape=(None,), chunks=True)
            else:
                # Append new data to it
                f['features'].resize((f['features'].shape[0] + audio.shape[0]), axis=0)
                f['features'][-audio.shape[0]:] = audio

                f['cams'].resize((f['cams'].shape[0] + cam.shape[0]), axis=0)
                f['cams'][-cam.shape[0]:] = cam

                f['img_frames'].resize((f['img_frames'].shape[0] + all_frames.shape[0]), axis=0)
                f['img_frames'][-all_frames.shape[0]:] = all_frames

                f['sequence'].resize((f['sequence'].shape[0] + len(sequence)), axis=0)
                f['sequence'][-len(sequence):] = sequence

                f['initial_time'].resize((f['initial_time'].shape[0] + len(initial_time)), axis=0)
                f['initial_time'][-len(initial_time):] = initial_time

                f['pseudo_labels'].resize((f['pseudo_labels'].shape[0] + len(pseudo_labels)), axis=0)
                f['pseudo_labels'][-len(pseudo_labels):] = pseudo_labels

                f['speech_activity'].resize((f['speech_activity'].shape[0] + len(speech_activity)), axis=0)
                f['speech_activity'][-len(speech_activity):] = speech_activity

                f['meta_male'].resize((f['meta_male'].shape[0] + len(meta_male)), axis=0)
                f['meta_male'][-len(meta_male):] = meta_male

                f['meta_female'].resize((f['meta_female'].shape[0] + len(meta_female)), axis=0)
                f['meta_female'][-len(meta_female):] = meta_female

                # f['all_frames'].resize((f['all_frames'].shape[0] + audio.shape[0]), axis=0)
                # f['all_frames'][-audio.shape[0]:] = audio

            # print("'features' chunk has shape:{}".format(f['features'].shape))
            print("'features' chunk has shape:{}".format(f['features'].shape), file=open('%s/make_h5py_log.txt' % h5py_dir, "w"))
            #print("'cams' chunk has shape:{}".format(f['cams'].shape))
            #print("'target_coords' chunk has shape:{}".format(f['target_coords'].shape))
            #print("'pseudo_labels' chunk has shape:{}".format(f['pseudo_labels'].shape))
            #print("'speech_activity' chunk has shape:{}".format(f['speech_activity'].shape))
            #print('--------------------------------------------------------------------')

            count = count + 1
            # if count == 10:
            #     break

        # print('Computing feature scaler...')
        print('Computing feature scaler...', file=open('%s/make_h5py_log.txt' % h5py_dir, "a"))
        utils.compute_scaler(f, h5py_dir, is_salsa=False)
        f.close()

    return h5py_path



if __name__ == "__main__":

    data_str = 'MC' if conf.logmelspectro['multi_mic'] else 'SC'

    data_str = data_str + '_seq'
        
    parser = argparse.ArgumentParser(description='Extract and save input features')
    parser.add_argument('--info', type=str, default=data_str, metavar='S',
                        help='Add additional info for storing (default: ours)')

    args = parser.parse_args()

    create_hdf5 = True

    h5py_path = main()

    # data_all = dataset_from_hdf5(h5py_path)
    # data_loader = DataLoader(data_all, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)

    # rand_num = int(torch.randint(1, high=len(data_all), size=(1,1)))

    # data = data_all[rand_num]

    # aud = data[0]
    # cam = data[1]
    # img = data[2]

    # num_imgs = aud.shape[0]

    # for d in data:
    #     print(d)

    # plt.figure()
    # for i in range(num_imgs):
    #     plt.subplot(int(num_imgs/4),4,i+1)
    #     plt.imshow(aud[i,:,:], aspect='auto')
    # plt.show()




