#!/usr/bin/python
import os
import h5py, argparse
from pathlib import Path
import core.config as conf
from core.dataset import dataset_from_scratch
from torch.utils.data import DataLoader
import utils.utils as utils

from tqdm import tqdm


base_path = conf.input['project_path']

train_or_test = 'train'
csv_file = os.path.join(base_path, 'data', 'csv', train_or_test + '.csv')
# csv_file = base_path + 'data/csv/' + train_or_test + '.csv'


def main():

    h5py_dir_str = os.path.join(base_path, 'data', 'h5py_%s' %(args.info),'')
    h5py_dir = Path(h5py_dir_str)
    h5py_dir.mkdir(exist_ok=True, parents=True)

    h5py_name = '%s_%s.h5' % (train_or_test, (args.info))

    h5py_path_str = os.path.join(h5py_dir_str, h5py_name)
    h5py_path = Path(h5py_path_str)

    if create_hdf5:
        ## ---------- Data loaders -----------------

        if h5py_name in os.listdir(h5py_dir_str):
            os.remove(h5py_path_str)

        f = h5py.File(h5py_path, 'a')

        d_dataset = dataset_from_scratch(csv_file, train_or_test, normalize=False, augment=False)

        total_size = len(d_dataset)

        data_loader = DataLoader(d_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        count = 0
        for data in tqdm(data_loader):

            print('count: {}/{}'.format('%s'%count, '%s'%total_size))

            audio = data[0]
            cam = data[1]
            all_frames = data[-1]

            #print(count)
            if count == 0:
                # Create the dataset at first
                f.create_dataset('features', data=audio, chunks=True, maxshape=(None, None, None, 64))
                f.create_dataset('cams', data=cam, chunks=True, maxshape=(None, 1, 11))
                f.create_dataset('img_frames', data=all_frames, chunks=True, maxshape=(None, None, 3, 224, 224))

            else:
                # Append new data to it
                f['features'].resize((f['features'].shape[0] + audio.shape[0]), axis=0)
                f['features'][-audio.shape[0]:] = audio

                f['cams'].resize((f['cams'].shape[0] + cam.shape[0]), axis=0)
                f['cams'][-cam.shape[0]:] = cam

                f['img_frames'].resize((f['img_frames'].shape[0] + all_frames.shape[0]), axis=0)
                f['img_frames'][-all_frames.shape[0]:] = all_frames

                # f['all_frames'].resize((f['all_frames'].shape[0] + audio.shape[0]), axis=0)
                # f['all_frames'][-audio.shape[0]:] = audio

            #print("'features' chunk has shape:{}".format(f['features'].shape))
            # print("'features' chunk has shape:{}".format(f['features'].shape), file=open('%s/make_h5py_log.txt' % h5py_dir, "w"))
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

    parser = argparse.ArgumentParser(description='Extract and save input features')
    parser.add_argument('--info', type=str, default='MultiChannel', metavar='S',
                        help='Add additional info for storing (default: ours)')
    args = parser.parse_args()

    create_hdf5 = True

    main()

    # h5py_path = main()

    # data_all = dataset_from_hdf5(h5py_path)
    # data_loader = DataLoader(data_all, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

    # for idx, data in enumerate(data_loader):
    #     aud = data[0]
    #     cam = data[1]
    #     img = data[-1]
    #     BS = aud.shape[0]

    #     if idx==2:
    #         print(aud.shape)
    #         print(cam.shape)
    #         print(img.shape)
    #         # plt.figure()
    #         # for i in range(16):
    #         #     plt.subplot(4,4,i+1)
    #         #     plt.imshow(aud[0,i,:,:], aspect='auto')
    #         # plt.show()
    #         plt.figure()
    #         for i in range(BS):
    #             plt.subplot(BS,1,i+1)
    #             plt.imshow(img[i,0,:,:,:].permute(1,2,0), aspect='auto')
    #         plt.show()
            
    #         break


