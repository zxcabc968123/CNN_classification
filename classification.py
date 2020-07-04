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
    #get the dataset from mnist
    (x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()
    #normalize
    x_train=x_train/255
    x_test=x_test/255
    print('data shape')
    print(x_train.shape)
    print(x_test.shape)
    # plt.imshow(x_test[0])
    # plt.show()
    x_train=x_train.reshape((60000,28,28,1))
    x_test=x_test.reshape((10000,28,28,1))
    print(x_train.shape)
    print(x_test.shape)
    # plt.imshow(x_test[0].squeeze(),cmap="gray")
    # plt.show()
    #create nateworrk
    CNN=keras.Sequential(name='CNN')
    #add convolution layer filter 32 3*3 activation funtion relu
    CNN.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
    #add pooling layer 2*2 
    CNN.add(layers.MaxPooling2D((2,2)))
    #add convolution layer filter 16 3*3 activation funtion relu
    CNN.add(layers.Conv2D(16,(3,3),activation='relu'))
    #add pooling layer 2*2
    CNN.add(layers.MaxPooling2D((2,2)))
    #Flat matrix to enter DNN network
    CNN.add(layers.Flatten())
    #add DNN layer btw. to do multi_object(10) classification the last layer is 10 soft max 
    CNN.add(layers.Dense(128,activation='relu'))
    CNN.add(layers.Dense(64,activation='relu'))
    CNN.add(layers.Dense(10,activation='softmax'))
    #define the optimizer(how to update parameter) and loss funtion:cross entropy 
    CNN.compile(optimizer=('Adam'),loss=keras.losses.sparse_categorical_crossentropy,metrics=['accuracy'])
    CNN.fit(x_train,y_train,batch_size=1000,epochs=5)
    #test CNN on testdata
    print('mean123:')
    print(np.mean(CNN.predict_classes(x_train)==y_train))
    print('mean:')
    print(np.mean(CNN.predict_classes(x_test)==y_test))
    print('save CNN')
    export_path='/home/allen/tensorflow_sample/SaveNet'
    tf.saved_model.save(CNN, export_path)

if __name__ =="__main__":
    
    main()




    





    


