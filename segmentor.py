import tensorflow as tf
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

class Segmentor(tf.keras.Model):
	def __init__(self):

		super(Segmentor, self).__init__()
		inputs = Input(input_size)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        drop2 = Dropout(0.5)(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        drop3 = Dropout(0.5)(conv3)

        up1 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop3))
        merge1 = concatenate([drop2, up1], axis=3)
        conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge1)
        conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)

        up2 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv4))
        merge2 = concatenate([conv1, up2], axis=3)
        conv5 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge2)
        conv5 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

        conv6 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        conv7 = Conv2D(1, 1, activation='sigmoid')(conv6)


	@tf.function
	def call(self):

		pass

	def accuracy_function(self):

		pass


	def loss_function(self):

		return pass
