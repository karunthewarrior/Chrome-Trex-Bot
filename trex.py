import time
import cv2
import mss
import numpy as np
import pyautogui
import thread
import os


global x,y 

# def mouseClick(event,x,y,flags,param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print x,y

def step(action):
	global step_count,score
	global game_over_img
	step_count +=1
	if(action == 1):
		os.system("xdotool key space")
	with mss.mss() as sct:
		monitor = {'top': 220, 'left': 115, 'width': 340, 'height': 80}
		img = np.array(sct.grab(monitor))
		img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)     
		ret, img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
		kernel = np.ones((2,2),np.uint8)
		mod_img = cv2.dilate(img,kernel,iterations = 1)
		cv2.imshow("image",mod_img)
		roi = img[13:37,154:185]

		input_vector = img.astype(np.float).ravel()
		input_vector = np.where(input_vector == 255, 1, 0)

		if(np.array_equal(roi,game_over_img)):
			reward = -100
			end_flag = 1
			step_count = 0 
		else:
			reward = score/4
			end_flag = 0

		return input_vector, reward,end_flag

global game_over_img
global step_count,score 
step_count = 0
score = 0
game_over_img = cv2.imread("game_over.jpg")
game_over_img = cv2.cvtColor(game_over_img,cv2.COLOR_BGRA2GRAY)
ret, game_over_img = cv2.threshold(game_over_img,127,255,cv2.THRESH_BINARY)
with mss.mss() as sct:
	monitor = {'top': 220, 'left': 115, 'width': 340, 'height': 80}
	while(1):
		score +=1
		x,y,z = step(1)
		print y 
		if(z==1):
			time.sleep(1)
			os.system("xdotool key space")               
			score = 0
			z= 0
		if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()                                   
			break