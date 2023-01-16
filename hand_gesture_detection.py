# TechVidvan hand Gesture Recognizer

# import necessary packages
import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import threading

from playsound import playsound
# from pygame import mixer
from tensorflow.keras.models import load_model
# from multiprocessing import Process, Value


BASE_PATH = 'sounds'
sounds = [fn.split('.')[0] for fn in os.listdir(BASE_PATH)]

frame_counter = 0
className = 'show me your hand!'

# queue = Value('b', False)
# manager = Manager()
# manage_dict = manager.dict({'playing': False})
playing = False

def play_sound(name):
	global playing
	sound_path = os.path.join(BASE_PATH, f'{name}.mp3')
	playsound(sound_path)
	# queue.value = False
	playing = False

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
	# Read each frame from the webcam
	_, frame = cap.read()
	# if frame is None:	raise Exception("Check your camera and run the script again!")
	# if frame is None:	continue
	x, y, c = frame.shape

	# Flip the frame vertically
	frame = cv2.flip(frame, 1)
	framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# Get hand landmark prediction
	result = hands.process(framergb)

	# print(result)

	# post process the result
	if result.multi_hand_landmarks:
		landmarks = []
		for handslms in result.multi_hand_landmarks:
			for lm in handslms.landmark:
				# print(id, lm)
				lmx = int(lm.x * x)
				lmy = int(lm.y * y)

				landmarks.append([lmx, lmy])

			# Drawing landmarks on frames
			mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)


			# predict one frame in second
			if frame_counter % 25 == 0:
				# Predict gesture
				prediction = model.predict([landmarks])
				classID = np.argmax(prediction)
				print(prediction)
				print(classID)

				_play = False
				if classNames[classID] != className:
					_play = True
					className = classNames[classID]

				if _play and className in sounds and not playing:
					# queue.value = True
					playing = True

					# ps = Process(target=play_sound, args=[className])
					ps = threading.Thread(target=play_sound, args=(className,))
					ps.start()
					# ps.join()
	else:
		className = 'show me your hand!'

	# show the prediction on the frame
	# print('playing: ', queue.value)
	print('playing: ', playing)
	cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

	# Show the final output
	cv2.imshow("Output", frame) 

	if cv2.waitKey(1) == ord('q'):
		break

	frame_counter += 1

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()