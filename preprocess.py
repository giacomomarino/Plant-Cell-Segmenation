import h5py
import numpy as np
from tqdm import tqdm

train_files = ['train/Movie1_t00003_crop_gt.h5',
'train/Movie1_t00009_crop_gt.h5',
'train/Movie1_t00010_crop_gt.h5',
'train/Movie1_t00012_crop_gt.h5',
'train/Movie1_t00014_crop_gt.h5',
'train/Movie1_t00016_crop_gt.h5',
'train/Movie1_t00018_crop_gt.h5',
'train/Movie1_t00020_crop_gt.h5',
'train/Movie1_t00035_crop_gt.h5',
'train/Movie1_t00040_crop_gt.h5',
'train/Movie1_t00045_crop_gt.h5',
'train/Movie1_t00049_crop_gt.h5',
'train/Movie2_T00000_crop_gt.h5',
'train/Movie2_T00002_crop_gt.h5',
'train/Movie2_T00006_crop_gt.h5',
'train/Movie2_T00008_crop_gt.h5',
'train/Movie2_T00010_crop_gt.h5',
'train/Movie2_T00012_crop_gt.h5',
'train/Movie2_T00014_crop_gt.h5',
'train/Movie2_T00016_crop_gt.h5',
'train/Movie3_T00002_crop_gt.h5',
'train/Movie3_T00004_crop_gt.h5']

test_files = ['test/Movie1_t00006_crop_gt.h5',
'test/Movie1_t00045_crop_gt.h5',
'test/Movie2_T00010_crop_gt.h5', 
'test/Movie2_T00020_crop_gt.h5']

def get_data(train_files: list, test_files: list):


    first = True
    for train_file in tqdm(train_files):
        if first:
            f = h5py.File(train_file, 'r')
            norm_data = np.divide(np.array(f['raw']), np.max(f['raw']))
            train_labels = np.array(f['label'])
            train_data = norm_data
            first = False
        else:
            f = h5py.File(train_file, 'r')
            norm_data = np.divide(np.array(f['raw']), np.max(f['raw']))
            np.append(train_labels, np.array(f['label']))
            np.append(train_data, norm_data)


    first = True
    for test_file in tqdm(test_files):
        if first:
            g = h5py.File(test_file, 'r')
            test_data = np.divide(np.array(g['raw']), np.max(g['raw']))
            test_labels = np.array(g['label'])
            first = False
        else:
            g = h5py.File(test_file, 'r')
            norm_data = np.divide(np.array(g['raw']), np.max(g['raw']))
            np.append(test_labels, np.array(g['label']))
            np.append(test_data, norm_data)

    print(' train_data shape', np.shape(train_data))
    print(' train_labels shape', np.shape(train_labels))
    print(' test_data shape', np.shape(test_data))
    print(' test_labels shape', np.shape(test_labels))

    return train_data, train_labels, test_data, test_labels


get_data(train_files[0:2], test_files[0:2])
