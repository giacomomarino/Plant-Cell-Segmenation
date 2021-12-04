import tensorflow as tf
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import *
from tensorflow.keras.preprocessing.image import *
import matplotlib.pyplot as plt

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()


a = tf.ones([4, 550, 1000,1])
print(a)
a = tf.expand_dims(a, axis=0)
print(a)
logs = []
for i in range(a.shape[0]):
    logs.append(a[i])

#print(logs)
display(logs)

