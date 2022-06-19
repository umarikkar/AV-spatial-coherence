#!/usr/bin/python

import h5py, argparse
from pathlib import Path
import core.config as conf
from core.dataset import dataset_from_scratch
from torch.utils.data import DataLoader
import utils.utils as utils

base_path = conf.input['project_path']

train_or_test = 'train'
csv_file = base_path + 'data/csv/' + train_or_test + '.csv'


def main():
    h5py_dir = Path(base_path + 'data/h5py_%s/' %(args.info)) # output dir
    h5py_dir.mkdir(exist_ok=True, parents=True)

    ## ---------- Data loaders -----------------
    d_dataset = dataset_from_scratch(csv_file, train_or_test, normalize=False, augment=False)
    data_loader = DataLoader(d_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    f = h5py.File(base_path + 'data/h5py/%s_dataset.h5' % train_or_test, 'a')

    count = 0

    for image in data_loader:

        # Data to be appended
        features = image[0]  # torch.squeeze(image[0], 0)
        cam = image[1]  # torch.squeeze(image[1], 0)
        target_coords = image[2]  # torch.squeeze(image[2], 0)
        pseudo_labels = image[3]  # torch.squeeze(image[3], 0)
        speech_activity = image[4]
        sequence = image[5]

        #print(count)
        if count == 0:
            # Create the dataset at first
            f.create_dataset('features', data=features, chunks=True, maxshape=(None, 16, 960, 64))
            f.create_dataset('cams', data=cam, chunks=True, maxshape=(None, 1, 11))
            f.create_dataset('target_coords', data=target_coords, maxshape=(None, 60), chunks=True)  # , maxshape=(None, 1))
            f.create_dataset('pseudo_labels', data=pseudo_labels, maxshape=(None, 60), chunks=True)  # , maxshape=(None, 1))
            f.create_dataset('speech_activity', data=speech_activity, maxshape=(None, 60), chunks=True)  # , maxshape=(None, 1))
            f.create_dataset('sequence', data=sequence, maxshape=(None,), chunks=True)

        else:
            # Append new data to it
            f['features'].resize((f['features'].shape[0] + features.shape[0]), axis=0)
            f['features'][-features.shape[0]:] = features

            f['cams'].resize((f['cams'].shape[0] + cam.shape[0]), axis=0)
            f['cams'][-cam.shape[0]:] = cam

            f['target_coords'].resize((f['target_coords'].shape[0] + target_coords.shape[0]), axis=0)
            f['target_coords'][-target_coords.shape[0]:] = target_coords

            f['pseudo_labels'].resize((f['pseudo_labels'].shape[0] + pseudo_labels.shape[0]), axis=0)
            f['pseudo_labels'][-pseudo_labels.shape[0]:] = pseudo_labels

            f['speech_activity'].resize((f['speech_activity'].shape[0] + speech_activity.shape[0]), axis=0)
            f['speech_activity'][-speech_activity.shape[0]:] = speech_activity

            f['sequence'].resize((f['sequence'].shape[0] + len(sequence)), axis=0)
            f['sequence'][-len(sequence):] = sequence

        #print("'features' chunk has shape:{}".format(f['features'].shape))
        print("'features' chunk has shape:{}".format(f['features'].shape), file=open('%s/make_h5py_log.txt' % h5py_dir, "a"))
        #print("'cams' chunk has shape:{}".format(f['cams'].shape))
        #print("'target_coords' chunk has shape:{}".format(f['target_coords'].shape))
        #print("'pseudo_labels' chunk has shape:{}".format(f['pseudo_labels'].shape))
        #print("'speech_activity' chunk has shape:{}".format(f['speech_activity'].shape))
        #print('--------------------------------------------------------------------')

        count = count + 1
        if count == 10:
            break


    print('Computing feature scaler...')
    print('Computing feature scaler...', file=open('%s/make_h5py_log.txt' % h5py_dir, "a"))
    utils.compute_scaler(f, h5py_dir, is_salsa=False)
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract and save input features')
    parser.add_argument('--info', type=str, default='gcc_16mics_firstref', metavar='S',
                        help='Add additional info for storing (default: ours)')
    args = parser.parse_args()
    main()