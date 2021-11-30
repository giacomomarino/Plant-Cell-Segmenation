import os
import numpy as np
import tensorflow as tf
import numpy as np
from segmentor import *
from preprocess import *
import sys
import random

def train(model, train_input, train_labels):
	for i in range(0, len(train_input), model.batch_size):
		inputs = train_input[i:i+model.batch_size]
		labels = train_labels[i:i+model.batch_size]
		with tf.GradientTape() as tape:
			logits = model.call(inputs)
			loss = model.loss_function(logits, labels)
		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_input, test_labels):
	losses = 0
	for i in range(0, len(test_input), model.batch_size):
		inputs = test_input[i:i+model.batch_size]
		labels = test_labels[i:i+model.batch_size]
		logits = model.call(inputs)
		losses += model.loss_function(logits, labels)
	#calculate accuracy

	#currently returns total losses
	return losses

def main():
	print("Running preprocessing...")
	train_data, train_labels, test_data, test_labels = get_data(train_files[0:2], test_files[0:2])
	print("Preprocessing complete.")

	
	model = Segmentor()
	print("Created SEGMENTOR")	

	print("Training")
	train(model, train_data, train_labels)
	print("Testing")
	loss = test(model, test_data, test_labels)
	print("Loss: " + loss)

if __name__ == '__main__':
	main()