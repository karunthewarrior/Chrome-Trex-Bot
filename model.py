import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class twolayerfc(nn.Module):
    def __init__(self):
        super(twolayerfc, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(300*60,600),
            nn.ReLU(True),
            nn.Linear(600,1),
            nn.Sigmoid()
        )

    def forward(self, x):
    	x = self.fc(x)
    	return x


class RLModel(object):
	def __init__(self):		
		self.c = 0
		self.all_inputs = []
		self.model = twolayerfc()
		self.gamma = 0.99
		self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.001
			,weight_decay=1e-5)

	def pre_process(self,img):
		self.c +=1
		img = img[30:150,0:600]
		img = cv2.resize(img, (300, 60), interpolation = cv2.INTER_CUBIC)
		img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		ret, img = cv2.threshold(img,50,255,cv2.THRESH_BINARY)
		if self.c==1:
			self.prev_img = img
			return img,self.c,self.all_inputs
		else:			
			# diff_img = img - self.prev_img
			diff_img = img
			self.prev_img = img 
			img_vec = diff_img.reshape(-1,1)/255
			self.all_inputs.append(img_vec)
			return img_vec,self.c,self.all_inputs

	def load_data(self,all_inputs):
		x = np.hstack(all_inputs)
		x_t = Variable(torch.from_numpy(np.transpose(x)).float())
		return x_t

	def discount_rewards(self,r):
		d_r = r
		sum = 0
		print r.size
		for i in reversed(xrange(r.size-5,r.size)):
			if r[i] <0: sum = 0
			sum = sum * self.gamma + r[i]
			d_r[i] = sum
		return d_r

	def forward_pass(self,x):
		x_t = Variable(torch.from_numpy(np.transpose(x)).float())
		out = self.model(x_t)
		return out.data.numpy()

	def forward_all(self,all_inputs):
		x_all = self.load_data(all_inputs)
		o_t = self.model(x_all)
		return o_t

	def backward_pass(self,all_inputs,y,r):
		d_r = discount_rewards(r)
		o_t = self.forward_all(all_inputs)
		y = np.asarray(y).reshape(-1,1)
		y_t = Variable(torch.from_numpy(y).float())
		self.loss =  (-d_r * (y_t * torch.log(o_t) + (1-y_t) * torch.log (1-o_t))).sum()
		self.optimizer.zero_grad()
		self.loss.backward()
		self.optimizer.step()
		return loss.data



