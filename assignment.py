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
    print('starting model proc')
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        # plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
        plt.show()

    print('showing model')
    plt.show(block=True)


def train(model, train_input, train_labels):
    print('starting train')
    # for i in range(0, len(train_input), model.batch_size):
    for i in tqdm(range(0, 10, model.batch_size)):
        print("starting train now in loop")
        inputs = tf.cast(tf.constant(train_input[i:i + model.batch_size]), dtype=tf.float32)
        labels = tf.cast(tf.constant(train_labels[i:i + model.batch_size]), dtype=tf.float32)
        # print("data has been batched", inputs, labels)
        with tf.GradientTape() as tape:
            print('inputs', inputs)
            logits = model.call(inputs)
            # print("logits done for round", i, logits)
            loss = model.loss_function(logits, labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print("gimme display")
        # print(logits)
        logs = []
        for i in range(logits.shape[0]):
            logs.append(logits[i])
        # print('logs', logs)
        # display(logs)


def test(model, test_input, test_labels):
    losses = 0
    print("test in", test_input.shape)
    print("test labels", test_labels.shape)
    logs = []
    for i in tqdm(range(0, len(test_input), model.batch_size)):
        print("i")
        print(range(0, len(test_input), model.batch_size))
        inputs = tf.cast(tf.constant(test_input[i:i + model.batch_size]), dtype=tf.float32)
        inps = []
        for i in range(inputs.shape[0]):
            inps.append(tf.expand_dims(inputs[i], axis=2))
        display(inps)
        labels = tf.cast(tf.constant(test_labels[i:i + model.batch_size]), dtype=tf.float32)
        logits = model.call(inputs)
        losses += model.loss_function(logits, labels)
        for i in range(logits.shape[0]):
            logs.append(logits[i])
        # calculate accuracy
    print("here da logs", len(logs), logs[0], logs)
    display(logs)

    # currently returns total losses
    return losses


def main():
    print("Running preprocessing...")
    train_data, train_labels, test_data, test_labels = get_data(train_files[0:2], test_files[0:2])
    print("Preprocessing complete.")

    print(' train_data shape', train_data.shape)
    print(' train_labels shape', train_labels.shape)
    print(' test_data shape', test_data.shape)
    print(' test_labels shape', test_labels.shape)
    model = Segmentor()
    print("Created SEGMENTOR")

    print("Training")
    train(model, train_data, train_labels)
    print("Testing")
    loss = test(model, test_data, test_labels)
    print("SEGMENTOR HAS SEGMENTORED: Loss: ", loss)


if __name__ == '__main__':
    main()
