import numpy as np
import tensorflow as tf
import numpy as np
from segmentor import *
from preprocess import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.utils import *
from tensorflow.keras.preprocessing.image import *
from skimage import segmentation
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        #plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

def recover_ignore_index(input, orig, ignore_index):
    if ignore_index is not None:
        mask = orig == ignore_index
        input[mask] = ignore_index

    return input

def ltb (m):
    boundaries = segmentation.find_boundaries(m, connectivity=2, mode='thick')
    boundaries = boundaries.astype('int32')
    results = []
    results.append(recover_ignore_index(boundaries, m, None))
    return np.stack(results, axis=0)


def train(model, train_input, train_labels):

    for j in tqdm(range(len(train_input))):
        train_movie = train_input[j]
        train_label = train_labels[j]
        
        for i in (range(0, len(train_movie), model.batch_size)):
            #print("starting train now in loop")
            inputs = tf.cast(tf.constant(train_movie[i:i + model.batch_size]), dtype=tf.float32)
            labels = tf.cast(tf.constant(train_label[i:i + model.batch_size]), dtype=tf.float32)

            #pre = tf.expand_dims(labels[0],2)
            #post = tf.expand_dims(np.squeeze(ltb(labels[0])),2)
            #print(pre,post)
            #display([pre,post])
            #print(labels)
            #disp = tf.expand_dims(labels[0], 2)
            #display([disp])

            labels = tf.stop_gradient(ltb(labels))

            #disp1 = tf.squeeze(labels)
            #print(disp1)
            #disp1 = tf.expand_dims(disp1[0], 2)
            #print(disp1)
            #display([disp1])

            with tf.GradientTape() as tape:
                #print('running call')
                logits = tf.squeeze(model.call(inputs))
                loss = model.loss_function(logits, tf.squeeze(labels))
                #print('loss worked!', loss)
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            #print("gimme display")
            # print(logits)
            logs = []
            #if j % 100 == 0:
            #    display(tf.expand_dims(logits,3))
            for i in range(logits.shape[0]):
                logs.append(logits[i])
            # print('logs', logs)
            #display(logs)


def test(model, test_input, test_labels):
    losses = 0
    #print("test in", test_input.shape)
    #print("test labels", test_labels.shape)
    logs = []
    accs = []
    for j in range(len(test_input)):
        test_movie = test_input[j]
        test_label = test_labels[j]
        for i in tqdm(range(0, len(test_movie), model.batch_size)):

            #print("i")
            #print(range(0, len(test_input), model.batch_size))

            inputs = tf.cast(tf.constant(test_movie[i:i + model.batch_size]), dtype=tf.float32)
            inps = []

            for i in range(inputs.shape[0]):
                inps.append(tf.expand_dims(inputs[i], axis=2))
        
            #display(inps)
            labels = tf.cast(tf.constant(test_label[i:i + model.batch_size]), dtype=tf.float32) 


            if inputs.shape[0] != labels.shape[0]:
                print("batch misalignment skipping")
                break

            labels = tf.stop_gradient(ltb(labels))


            logits = tf.squeeze(model.call(inputs))
            logits = model.call(inputs)
            logits = tf.squeeze(logits)

            labels = tf.squeeze(labels)
            losses += model.loss_function(logits, labels)

            for i in range(logits.shape[0]):
                #logs.append(logits[i])
                accs.append(model.accuracy_function(labels[i], logits[i], .7))
                #display([labels[i], logits[i]])
            # calculate accuracy
        #print("here da logs", len(logs), logs[0], logs)

        # currently returns total losses
    return losses, tf.reduce_mean(accs), logits


def main():
    print("Running preprocessing...")
    train_data, train_labels, test_data, test_labels = get_data(train_files[0:2], [test_files[1]])
    print("Preprocessing complete.")

    model = Segmentor()
    print("Created SEGMENTOR")

    print("Training")
    train(model, train_data, train_labels)
    print("Testing")
    loss, accs, logits = test(model, test_data, test_labels)
    #display([tf.expand_dims(logits[0], 2)])
    print("SEGMENTOR HAS SEGMENTORED: Loss: ", loss)
    print("Accuracy: ", accs)


if __name__ == '__main__':
    main()
