import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    export_path = '/home/allen/tensorflow_sample/SaveNet'
    reload_sm_keras = tf.keras.models.load_model(export_path)
    #reload_sm_keras.summary()
    (x_train,y_train),(x_test123,y_test123)=tf.keras.datasets.mnist.load_data()
    print(x_test123.shape)
    print(y_test123.shape)
    x_test123=x_test123/255
    x_test123=x_test123.reshape((10000,28,28,1))
    print(np.mean(reload_sm_keras.predict_classes(x_test123)==y_test123))

    #print(np.mean(reload_sm_keras.predict_classes(x_test)==y_test))


if __name__ == "__main__":
    main()
    