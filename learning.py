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

NAME = "UP-vs-DOWN-cnn-64-2-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

start = time.time()

x = np.load('C:/Users/Lenovo/PycharmProjects/handDetect/DataSets/features.npy')
y = np.load('C:/Users/Lenovo/PycharmProjects/handDetect/DataSets/labels.npy')

x = x/255.0

model = Sequential()
model.add(Conv2D(64, (12, 12), input_shape = x.shape[1:]))  # ... WINDOW ...
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))  # TODO możliwe mnozenie przez 4

model.add(Conv2D(64, (12,12)))  # ... WINDOW ...
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])

model.fit(x, y, batch_size=16, epochs=5, validation_split=0.1, callbacks=[tensorboard])
end = time.time()
print(end - start)

