import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from partitioning import *



class Segmentor(tf.keras.Model):
    def __init__(self):
        super(Segmentor, self).__init__()
        # hyperparams
        self.batch_size = 8
        self.alpha = 0.0002
        self.optimizer = Adam(learning_rate=self.alpha)
        self.hd = 16

        self.conv_down1 = Sequential([Conv2D(self.hd * 1, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
                                      Conv2D(self.hd * 1, 3, activation='relu', padding='same', kernel_initializer='he_normal')])
                                      #MaxPooling2D(pool_size=(2, 2))])
        self.mp1 = MaxPooling2D(pool_size=(2, 2))
        self.conv_down2 = Sequential([Conv2D(self.hd * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
                                      Conv2D(self.hd * 3, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
                                      Dropout(0.5)])
                                      #MaxPooling2D(pool_size=(2, 2))])
        self.mp2 = MaxPooling2D(pool_size=(5, 5))
        self.bottom = Sequential([Conv2D(self.hd * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
                                  Conv2D(self.hd * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
                                  UpSampling2D(size=(5, 5))])
        # this net needs to concat w/ bottom output and down2 output
        self.conv_up1 = Sequential([Conv2D(self.hd * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
                                    Conv2D(self.hd * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
                                    UpSampling2D(size=(2, 2))])
        self.conv_up2 = Sequential([Conv2D(self.hd * 1, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
                                    Conv2D(self.hd * 1, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
                                    Conv2D(1, 1, activation='sigmoid', padding='same', kernel_initializer='he_normal')])
                                    #Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
                                    #Conv2D(1, 1, activation='sigmoid')])

    def call(self, inputs):
        #print('inputs in call', inputs)
        inputs = tf.expand_dims(inputs, 3)
        #print("inputs shape be", inputs.shape, inputs)
        down1 = self.conv_down1(inputs)
        #print("down1 layer done!", down1.shape)
        downpool1 = self.mp1(down1)
        down2 = self.conv_down2(downpool1)
        #print("down2 layer done!", down2.shape)
        downpool2 = self.mp2(down2)
        bottom = self.bottom(downpool2)
        #print("bottom layer done!", bottom.shape)
        bottom_concat = concatenate([bottom, down2], axis=3)
        #print("bottom concated,", bottom_concat.shape)
        up1 = self.conv_up1(bottom_concat)
        up1_concat = concatenate([down1, up1], axis=3)
        #print("up1 concat layer done!")
        up2 = self.conv_up2(up1_concat)
        #print("up2 layer done!", up2)

        '''
        parted =[]
        for k in range(up2.shape[0]):
            part = tf.expand_dims(partitioner(up2[k]),0)
            part = tf.cast(part, dtype=tf.float32)
            parted.append(part)
        #print(parted)
        preds = tf.concat(parted,0)
        print('done call, returning')
        '''
        return up2



    @tf.function
    def accuracy_function(self, label, pred, threshold):

        if label.shape != pred.shape:
            pred = tf.reshape(pred, label.shape)

        thresh = tf.constant(threshold, dtype=tf.dtypes.float32)

        # binarize predictions

        binarized = tf.math.greater(pred, thresh)

        # Measure accuraced
        correct = tf.equal(binarized, tf.cast(label, dtype=tf.dtypes.bool))

        accuracy = tf.reduce_mean(tf.cast(correct, tf.dtypes.float32))

            # Measure scores
        #precision = precision_score(tf.reshape(label, [-1]), tf.reshape(binarized, [-1]))
        #recall = recall_score(tf.reshape(label, [-1]), tf.reshape(binarized, [-1]))
        #f1 = 2 * ((precision * recall) / (precision + recall))
        return accuracy

        
    @tf.function
    def loss_function(self, logits, labels):
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        #print("loss layer inited")
        loss = bce(labels, logits)

        return tf.reduce_mean(loss)

