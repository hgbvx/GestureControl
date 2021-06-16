import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
from path import Path

def prepare(filepath):
    IMG_SIZE = 100
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # return the image with shaping that TF wants.


CATEGORIES = ["Si", "Thumb"]

model_dc = tf.keras.models.load_model('test_model_dc.model')

X = np.load(Path.testFeatures)
y = np.load(Path.testlabels)


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