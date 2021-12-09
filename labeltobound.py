from skimage import segmentation
import numpy as np
import h5py
from assignment import display
import tensorflow as tf

def _recover_ignore_index(input, orig, ignore_index):
    if ignore_index is not None:
        mask = orig == ignore_index
        input[mask] = ignore_index

    return input

def ltb (m):
    boundaries = segmentation.find_boundaries(m, connectivity=2, mode='thick')
    boundaries = boundaries.astype('int32')
    results = []
    results.append(_recover_ignore_index(boundaries, m, None))
    return np.stack(results, axis=0)


f = h5py.File('test/Movie1_t00006_crop_gt.h5', 'r')
test_labels = f['label']
test = ltb(test_labels[0:4, :, :])
test = test[0, :, :]
display([tf.expand_dims(test[0],2)])

