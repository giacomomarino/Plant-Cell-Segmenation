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


def display(display_list):
    #print('starting model proc')
    plt.figure(figsize=(1330, 650))
    plt.show()

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        # plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    
    #print('showing model')


def train(model, train_input, train_labels):


    for j in range(len(train_input)):
        train_movie = train_input[j]
        train_label = train_labels[j]
        
        for i in tqdm(range(0, len(train_movie), model.batch_size)):
            #print("starting train now in loop")
            inputs = tf.cast(tf.constant(train_movie[i:i + model.batch_size]), dtype=tf.float32)
            labels = tf.cast(tf.constant(train_label[i:i + model.batch_size]), dtype=tf.float32)

            with tf.GradientTape() as tape:
                #print('inputs', inputs)
                logits = model.call(inputs)
                # print("logits done for round", i, logits)
                loss = model.loss_function(logits, labels)
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            #print("gimme display")
            # print(logits)
            logs = []
            for i in range(logits.shape[0]):
                logs.append(logits[i])
            # print('logs', logs)
            display(logs)


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

            logits = model.call(inputs)
            losses += model.loss_function(logits, labels)

            for i in range(logits.shape[0]):
                logs.append(logits[i])
                accs.append(model.accuracy_function(labels[i], logits[i], .65))
                display([labels[i], logits[i]])
            # calculate accuracy
        #print("here da logs", len(logs), logs[0], logs)

        # currently returns total losses
        return losses, accs


def main():
    print("Running preprocessing...")
    train_data, train_labels, test_data, test_labels = get_data(train_files[:5], train_files[5:7])
    print("Preprocessing complete.")

    model = Segmentor()
    print("Created SEGMENTOR")

    print("Training")
    train(model, train_data, train_labels)
    print("Testing")
    loss, accs = test(model, test_data, test_labels)
    print("SEGMENTOR HAS SEGMENTORED: Loss: ", loss)
    print("Accuracy: " + tf.reduce_mean(accs))


if __name__ == '__main__':
    main()
