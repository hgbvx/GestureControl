#Convolution neural network

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.client import device_lib
import time
from path import Path


NAME = "UP-vs-DOWN-cnn-64-2-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

start = time.time()

X = np.load(Path.features)
Y = np.load(Path.labels)

X = X/255.0

dense_layers = [0]
layer_sizes = [64]
conv_layers = [4]

model = Sequential()

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:


            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'],
                          )

            model.fit(X, Y,
                      batch_size=32,
                      epochs=15,
                      validation_split=0.3)

model.save('test_model_dc.model')

end = time.time()
print(end - start)

