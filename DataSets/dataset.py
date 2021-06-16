import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
import pickle

#DATADIR = "C:\\Users\Lenovo\PycharmProjects\handDetect\OutputImages"
DATADIR = "E:/programywdomku/GR_Python/GR_datas_img/kagglecatsanddogs_3367a/PetImages"

CATEGORIES = ["Cat", "Dog"]

IMG_SIZE = 70

training_data = []
testing_data = []


def create_training_data():
	for category in CATEGORIES:
		path = os.path.join(DATADIR, category)  # path to up and down
		class_num = CATEGORIES.index(category)
		for img in os.listdir(path):
			try:
				img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
				new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
				#training_data.append([new_array, class_num])
				testing_data.append([new_array, class_num])
			except Exception as e:
				pass

create_training_data()

#print(len(training_data))

#random.shuffle(training_data)

#for sample in training_data[:10]:
	#print(sample[1])

print(len(testing_data))

random.shuffle(testing_data)

for sample in testing_data[:10]:
	print(sample[1])

x = []
y = []
#change for training
for features, label in testing_data:
	x.append(features)
	y.append(label)

x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1) # 1 to 3 if color

#Enter needed filename
np.save('features_test.npy',x)
np.save('labels_test.npy',y)

