#testing the model
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import pickle
import cv2

def prepare(filepath):
    IMG_SIZE = 100
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # return the image with shaping that TF wants.


CATEGORIES = ["Si", "Thumb"]

model_dc = tf.keras.models.load_model('test_model_dc.model')

pickle_in = open("X_test.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y_test.pickle", "rb")
y = pickle.load(pickle_in)

X = X/255.0

#predictions = model_dc.predict([prepare('catty3.jpg')])
predictions = model_dc.predict([X])
IMG_CNT = 0

print(predictions)
for i in range(8):
    predictions[i][0] = round(predictions[i][0])
    #print(int(predictions[i][0]))
    print(i)
    print("Actual category : " + CATEGORIES[y[i]])
    print("Predicted category: " + CATEGORIES[int(predictions[i][0])])
    #print(CATEGORIES[int(predictions[IMG_CNT][0])])

image = X[IMG_CNT]

plt.imshow(image,cmap=plt.cm.binary)
plt.title(CATEGORIES[int(predictions[IMG_CNT][0])])
plt.show()