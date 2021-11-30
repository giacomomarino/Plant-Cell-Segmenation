import tensorflow as tf
import numpy as np
#from tf import keras
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

    def call(self, inputs):
        down1 = self.conv_down1(inputs)
        down2 = self.conv_down2(down1)
        bottom = self.bottom(down2)
        bottom_concat = concatenate([down2, bottom], axis=3)
        up1 = self.conv_up1(bottom_concat)
        up1_concat = concatenate([down1, up1], axis=3)
        up2 = self.conv_up2(up1_concat)
        return up2


    '''
    def model_maker(input_size):
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
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

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

        model = Model(input=inputs, output=conv7)
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def call(self, inputs):'''


