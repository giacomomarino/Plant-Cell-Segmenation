import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from transformer_model import Transformer_Seq2Seq
from rnn_model import RNN_Seq2Seq
import sys
import random

def train(model):

	pass

def main():
	print("Running preprocessing...")
	train_data, train_labels, test_data, test_data = get_data(train_files, test_files)
	print("Preprocessing complete.")
	train(model, train_data)
	loss, accuracy = test(model, test_data)

if __name__ == '__main__':
	main()