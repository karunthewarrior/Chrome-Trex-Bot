import time
import cv2
import mss
import numpy as np
import pyautogui
import thread
import os


global x,y 

def mouseClick(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print x,y
def relu(x):
	x[x<0] = 0
	return x

def backprop_relu(x):
	x[x<=0] = 0
	x[x>0] = 1
	return x

def sigmoid(x):
	o = 1/(1 + np.exp(-x))
	return o

def forward_propagate(X):
	Z1 = np.dot(model["W1"],X)
	A1 = relu(Z1)
	Z2 = np.dot(model["W2"],A1)
	A2 = sigmoid(Z2)
	return A2,A1,Z1

def back_propagate(X,Y,A2,A1,Z1,model,m):	
	dZ2 = A2 - Y
	dW2 = np.dot(dZ2,A1.T)/m
	dZ1 = np.dot(model["W2"].T,dZ2) * backprop_relu(Z1)
	dW1 = np.dot(dZ1,X)/m
	params = {'dW1':dW1, 'dW2':dW2}
	return params	

def step(prev_img,action):
	global step_count,score
	global game_over_img
	step_count +=1
	if(action == 1):
		os.system("xdotool key space")
	with mss.mss() as sct:
		monitor = {'top': 220, 'left': 115, 'width': 300, 'height': 80}
		img = np.array(sct.grab(monitor))
		img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)     
		ret, img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
		kernel = np.ones((2,2),np.uint8)
		# mod_img = prev_img - cv2.dilate(img,kernel,iterations = 1) 
		mod_img = cv2.dilate(img,kernel,iterations = 1) 
		cv2.imshow("image",mod_img)
		roi = img[13:37,114:141]

		input_vector = img.astype(np.float).ravel()
		input_vector = np.where(input_vector == 255, 1, 0)

		if(np.array_equal(roi,game_over_img)):
			reward = -100
			end_flag = 1
			step_count = 0
		else:
			reward = score/4
			end_flag = 0

		return input_vector, reward, end_flag

global game_over_img
global step_count,score 
step_count = 0
score = 0
num_episode = 0 

m = 3 #batch size

game_over_img = cv2.imread("game_over.jpg")
game_over_img = cv2.cvtColor(game_over_img,cv2.COLOR_BGRA2GRAY)
ret, game_over_img = cv2.threshold(game_over_img,127,255,cv2.THRESH_BINARY)
with mss.mss() as sct:
	monitor = {'top': 220, 'left': 115, 'width': 300, 'height': 80}
	img = np.array(sct.grab(monitor))
	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)     
	ret, img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
	kernel = np.ones((2,2),np.uint8)
	mod_img = cv2.dilate(img,kernel,iterations = 1)
	cv2.imshow("image",mod_img)
	roi = img[13:37,114:141]

	X = img.astype(np.float).ravel()
	X = np.where(X == 255, 1, 0)

	while(1):
		score +=1
		# A2,A1,Z1 = forward_propagate(X)
		# action = 1 if np.random.uniform() < A2 else 0
		# Y = 1 if action == 1 else 0
		X,r,end_flag = step(mod_img,0)
		
		cv2.setMouseCallback('image',mouseClick)
		if(end_flag==1):
			time.sleep(1)
			print "EPISODE OVER"
			os.system("xdotool key space")               
			score = 0
			end_flag= 0
			num_episode+=1

		if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()                   
			break