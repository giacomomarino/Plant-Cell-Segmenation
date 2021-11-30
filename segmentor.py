import tensorflow as tf
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

class Segmentor(tf.keras.Model):
    def __init__(self):
        super(Segmentor, self).__init__()
        self.conv_down1 = Sequential([Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
                                      Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
                                      MaxPooling2D(pool_size=(2, 2))])
        self.conv_down2 = Sequential([Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
                                      Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
                                      Dropout(0.5),
                                      MaxPooling2D(pool_size=(2, 2))])
        self.bottom = Sequential([Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
                                      Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
                                      UpSampling2D(size=(2, 2))])
        #this net needs to concat w/ bottom output and down2 output
        self.conv_up1 = Sequential([Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
                                    Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
                                    UpSampling2D(size=(2, 2))])
        self.conv_up2 = Sequential([Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
                                    Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
                                    Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
                                    Conv2D(1, 1, activation='sigmoid')])

    @tf.function
    def call(self, inputs):
        down1 = self.conv_down1(inputs)
        down2 = self.conv_down2(down1)
        bottom = self.bottom(down2)
        bottom_concat = concatenate([down2, bottom], axis=3)
        up1 = self.conv_up1(bottom_concat)
        up1_concat = concatenate([down1, up1], axis=3)
        up2 = self.conv_up2(up1_concat)
        return up2

    def accuracy_function(self):

        pass


    def loss_function(self):


        return pass

