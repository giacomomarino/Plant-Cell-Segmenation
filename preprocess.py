import h5py
import numpy as np
from tqdm import tqdm
import scipy as sc

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

    train_labels = []
    train_data = []
    for train_file in tqdm(train_files):
        f = h5py.File(train_file, 'r')
        norm_data = np.divide(np.array(f['raw']), np.max(f['raw']))
        train_label = np.array(f['label'])
        train_labels.append(train_label)
        train_data.append(norm_data)


    test_labels = []
    test_data = []
    for test_file in tqdm(test_files):
        g = h5py.File(test_file, 'r')
        norm_data = np.divide(np.array(g['raw']), np.max(g['raw']))
        label = np.array(g['label'])
        test_labels.append(label)
        test_data.append(norm_data)

    #print(' train_data shape', np.shape(train_data))
    #print(' train_labels shape', np.shape(train_labels))
    #print(' test_data shape', np.shape(test_data))
    #print(' test_labels shape', np.shape(test_labels))

    return train_data, train_labels, test_data, test_labels


#train_data, train_labels, test_data, test_labels = get_data(train_files[0:2], test_files[0:2])
