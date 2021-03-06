import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
from matplotlib import pyplot as plt
import tensorflow as tf


# Calculations of angles
#joint_list = [[8, 7, 6], [12, 11, 10], [4, 3, 2], [1, 0, 5]]
joint_list = [[4, 0, 8]]

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_hands.HandLandmark.WRIST

cap = cv2.VideoCapture(0)


def draw_finger_angles(image, results, joint_list):
	# Loop through hands
	for hand in results.multi_hand_landmarks:
		# Loop through joint sets
		for joint in joint_list:
			a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y])  # First coord
			b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y])  # Second coord
			c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y])  # Third coord

			radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
			angle = np.abs(radians * 180.0 / np.pi)

			if angle > 180.0:
				angle = 360 - angle

			cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(b, [640, 480]).astype(int)),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
	return image

def get_finger_angles(image, results, joint_list):
	# Loop through hands
	for hand in results.multi_hand_landmarks:
		# Loop through joint sets
		for joint in joint_list:
			a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y])  # First coord
			b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y])  # Second coord
			c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y])  # Third coord

			radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
			angle = np.abs(radians * 180.0 / np.pi)

			if angle > 180.0:
				angle = 360 - angle

	return angle

def get_label(index, hand, results):
	output = None
	for idx, classification in enumerate(results.multi_handedness):
		if classification.classification[0].index == index:
			# Process results
			label = classification.classification[0].label
			score = classification.classification[0].score
			text = '{} {}'.format(label, round(score, 2))

			# Extract Coordinates
			coords = tuple(np.multiply(
				np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
				[640, 480]).astype(int))

			output = text, coords

	return output

def prepare(frm):
	IMG_SIZE = 100
	img_array = frm
	new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
	return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # return the image with shaping that TF wants.

def normalize(x, x_min, x_max):
	return (x-x_min)/(x_max-x_min)

def volume(angle):
	if angle > 6.0 and angle < 45:
		v = normalize(angle, 6.0, 45.0) * 100
		print(v)





# os.mkdir('Output images') # comment if not save
model_dc = tf.keras.models.load_model('test_model_dc.model')

angles_flag = False

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=2) as hands:
	while cap.isOpened():
		ret, frame = cap.read()

		# Shift colors for media pipe
		image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# Flip on horizontal
		image = cv2.flip(image, 1)

		# Set flag
		image.flags.writeable = False

		# Detections
		results = hands.process(image)

		# Set flag true
		image.flags.writeable = True

		# Re shift colors
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

		prep_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		frame_x = prepare(prep_image)

		frame_x = frame_x/255.0

		predictions = model_dc.predict([frame_x])

		# print(predictions)



# print(results)

		# Rendering results
		if results.multi_hand_landmarks:
			for num, hand in enumerate(results.multi_hand_landmarks):
				mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
										  mp_drawing.DrawingSpec(color=(21, 122, 21), thickness=2, circle_radius=4),
										  mp_drawing.DrawingSpec(color=(121, 144, 250), thickness=2, circle_radius=2))
				results.multi_hand_landmarks[0]

				results.multi_handedness[0].classification[0].index == num

				round(results.multi_handedness[0].classification[0].score, 2)
				if get_label(num, hand, results):
					text, coord = get_label(num, hand, results)
					cv2.putText(image, text, coord, cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

				ang = get_finger_angles(image, results, joint_list)

				if ang < 6:
					angles_flag = True
				if ang > 45:
					angles_flag = False

				if angles_flag == True:
					vol = volume(ang)
					if vol != None:
						print(vol)

		#cv2.imwrite(os.path.join('Output Images/down', '{}.jpg'.format(uuid.uuid1())), image) # comment if not save
		cv2.imshow('Hand Tracking', image)

		if cv2.waitKey(10) and 0xFF == ord('q'):
			break

#plt
cap.release()
cv2.destroyAllWindows()

