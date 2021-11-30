import h5py
import numpy as np

train_files = ['test/Movie1_t00006_crop_gt.h5', 'test/Movie1_t00045_crop_gt.h5', 
'test/Movie2_T00010_crop_gt.h5', 'test/Movie2_T00020_crop_gt.h5']

test_files = ['test/Movie1_t00006_crop_gt.h5', 'test/Movie1_t00045_crop_gt.h5', 
'test/Movie2_T00010_crop_gt.h5', 'test/Movie2_T00020_crop_gt.h5']

def get_data(train_files: list, test_files: list):

    for train_file in train_files:
        f = h5py.File(train_file, 'r')
        train_labels = (f['label'])
        train_data = list(f['raw'])
        max = np.max(f['raw'])

    for test_file in test_files:
        f = h5py.File(train_file, 'r')
        test_labels = list(f['label'])
        test_data = list(f['raw'])

    train_data = np.array(train_data) / max
    train_labels = np.array(train_labels)

    test_data = np.array(test_data) / max
    test_labels = np.array(test_labels)

    return train_data, train_labels, test_data, test_labels

train_data, train_labels, test_data, test_labels = get_data(train_files, train_files)
print(train_data)