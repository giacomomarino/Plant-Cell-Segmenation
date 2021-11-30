import h5py
import numpy as np

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

    train_data = []
    train_labels = []
    for train_file in train_files:
        f = h5py.File(train_file, 'r')
        train_labels = train_labels + list(f['label'])
        train_data = train_data + list(f['raw'])
        max = np.max(f['raw'])

    test_data = []
    test_labels = []
    for test_file in test_files:
        f = h5py.File(test_file, 'r')
        test_labels = test_labels + list(f['label'])
        test_data = test_data + list(f['raw'])

    train_data = np.array(train_data) / max
    train_labels = np.array(train_labels)

    test_data = np.array(test_data) / max
    test_labels = np.array(test_labels)

    return train_data, train_labels, test_data, test_labels
