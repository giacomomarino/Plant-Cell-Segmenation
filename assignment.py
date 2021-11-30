import os
import numpy as np
import tensorflow as tf
import numpy as np
from model import Segmentor
from preprocess import *
import sys
import random

def train(model, train_input, train_labels):
	for i in range(0, len(train_input), model.batch_size):
		inputs = train_input[i:i+model.batch_size]
		labels = train_labels[i:i+model.batch_size]
		with tf.GradientTape() as tape:
			probs = model.call(inputs)
			loss = model.loss_function(probs, labels)
		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_input, test_labels):
	losses = 0
	for i in range(0, len(test_input), model.batch_size):
		inputs = test_input[i:i+model.batch_size]
		labels = test_labels[i:i+model.batch_size]
		probs = model.call(inputs)
		losses += model.loss_function(probs, label)
	#calculate accuracy

	#currently returns total losses
	return losses

def main():
	print("Running preprocessing...")
	train_data, train_labels, test_data, test_data = get_data(train_files, test_files)
	print("Preprocessing complete.")
	model = Segmentor()

	train(model, train_data)
	loss, accuracy = test(model, test_data)

if __name__ == '__main__':
	main()