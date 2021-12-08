import tensorflow as tf
import numpy as np
import h5py
from assignment import display, train

def accuracy_function(label, pred, threshold):

    if label.shape != pred.shape:
        pred = tf.reshape(pred, label.shape)

    thresh = tf.constant(threshold, dtype=tf.dtypes.float32)

    # binarize predictions

    binarized = tf.math.greater(pred, thresh)
    print("pred mat")
    print(binarized)

    print("labels")
    print(label)

    # Measure accuraced
    correct = tf.equal(binarized, tf.cast(label, dtype=tf.dtypes.bool))
    print("correct matrix")
    print(correct)


    accuracy = tf.reduce_mean(tf.cast(correct, tf.dtypes.float32))

        # Measure scores
    #precision = precision_score(tf.reshape(label, [-1]), tf.reshape(binarized, [-1]))
    #recall = recall_score(tf.reshape(label, [-1]), tf.reshape(binarized, [-1]))
    #f1 = 2 * ((precision * recall) / (precision + recall))
    return accuracy

preds = np.array([0.5, 0, .3])
threshold = .4
label = np.array([0, 1, 0])

#print(accuracy_function(label, preds, threshold))



f = h5py.File('test/Movie1_t00006_crop_gt.h5', 'r')
test_labels = np.array(f['label'])
f2 = h5py.File('test/Movie1_t00006_crop_gt.h5', 'r')
test_labels2 = np.array(f['label'])

print(test_labels.shape)
print(test_labels2.shape)

print(np.append(test_labels, test_labels2, axis= 0).shape)

display([tf.reshape(test_labels[200, :, :], [620, 1330, 1])])

