import tensorflow as tf
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

class Segmentor(tf.keras.Model):
    def __init__(self):
        super(Segmentor, self).__init__()
        # hyperparams
        self.batch_size = 128
        self.alpha = 0.001
        self.optimizer = Adam(learning_rate=self.alpha)

        self.conv_down1 = Sequential([Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
                                      Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')])
                                      #MaxPooling2D(pool_size=(2, 2))])
        self.mp1 = MaxPooling2D(pool_size=(2, 2))
        self.conv_down2 = Sequential([Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
                                      Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
                                      Dropout(0.5)])
                                      #MaxPooling2D(pool_size=(2, 2))])
        self.mp2 = MaxPooling2D(pool_size=(5, 5))
        self.bottom = Sequential([Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
                                  Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
                                  UpSampling2D(size=(5, 5))])
        # this net needs to concat w/ bottom output and down2 output
        self.conv_up1 = Sequential([Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
                                    Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
                                    UpSampling2D(size=(2, 2))])
        self.conv_up2 = Sequential([Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
                                    Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
                                    Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
                                    Conv2D(1, 1, activation='sigmoid')])

    @tf.function
    def call(self, inputs):
        inputs = tf.expand_dims(inputs, 3)
        print("inputs shape be", inputs.shape)
        down1 = self.conv_down1(inputs)
        print("down1 layer done!", down1.shape)
        downpool1 = self.mp1(down1)
        down2 = self.conv_down2(downpool1)
        print("down2 layer done!", down2.shape)
        downpool2 = self.mp2(down2)
        bottom = self.bottom(downpool2)
        print("bottom layer done!", bottom.shape)
        bottom_concat = concatenate([bottom, down2], axis=3)
        print("bottom concated,", bottom_concat.shape)
        up1 = self.conv_up1(bottom_concat)
        up1_concat = concatenate([down1, up1], axis=3)
        print("up1 concat layer done!")
        up2 = self.conv_up2(up1_concat)
        #do it be working?!?!
        return up2

a = tf.ones([128, 550, 1000])
print(a)

model = Segmentor()

model.call(a)

