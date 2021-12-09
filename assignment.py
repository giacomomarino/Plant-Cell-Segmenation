import os
import numpy as np
import tensorflow as tf
import numpy as np
from segmentor import *
from preprocess import *
from tqdm import tqdm
import sys
import random
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.utils import *
from tensorflow.keras.preprocessing.image import *
from partitioning import partitioner, display
from labeltobound import ltb


def train(model, train_input, train_labels):

    for j in tqdm(range(len(train_input))):
        train_movie = train_input[j]
        train_label = train_labels[j]
        
        for i in (range(0, len(train_movie), model.batch_size)):
            #print("starting train now in loop")
            #this is shit HD added to test the function to convert boundaries to labels
            inputs = tf.cast(tf.constant(train_movie[i:i + model.batch_size]), dtype=tf.float32)
            labels = tf.cast(tf.constant(train_label[i:i + model.batch_size]), dtype=tf.float32)
            pre = tf.expand_dims(labels[0],2)
            post = tf.expand_dims(np.squeeze(ltb(labels[0])),2)
            #print(pre,post)
            #display([pre,post])
            labels = tf.stop_gradient(ltb(labels))

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
            labels = tf.stop_gradient(ltb(labels))

            if inputs.shape[0] != labels.shape[0]:
                print("batch misalignment skipping")
                break

            logits = model.call(inputs)
            losses += model.loss_function(logits, labels)

            for i in range(logits.shape[0]):
                #logs.append(logits[i])
                accs.append(model.accuracy_function(ltb(labels[i]), logits[i], .40))
                #display([labels[i], logits[i]])
            # calculate accuracy
        #print("here da logs", len(logs), logs[0], logs)

        # currently returns total losses
    return losses, accs, logits


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
    display([logits[0]])
    print("SEGMENTOR HAS SEGMENTORED: Loss: ", loss)
    print("Accuracy: ", tf.reduce_mean(accs))


if __name__ == '__main__':
    main()
