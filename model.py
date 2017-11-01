import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

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
		self.all_inputs = []
		self.model = twolayerfc()
		self.gamma = 0.99
		self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.001
			,weight_decay=1e-5)

	def pre_process(self,img,score,c):
		img = img[30:150,0:600]
		img = cv2.resize(img, (300, 60), interpolation = cv2.INTER_CUBIC)
		img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		ret, img = cv2.threshold(img,50,255,cv2.THRESH_BINARY)
		if c==1:
			self.prev_img = img
			self.prev_score = score
			return img,self.all_inputs,self.prev_score
		else:			
			r = score - self.prev_score
			self.prev_score = score
			diff_img = img - self.prev_img
			self.prev_img = img 
			img_vec = diff_img.reshape(-1,1)/255
			self.all_inputs.append(img_vec)
			return img_vec,self.all_inputs,r
	
	def clear_inputs(self):
		self.all_inputs = []

	def load_data(self,all_inputs):
		x = np.hstack(all_inputs)
		x_t = Variable(torch.from_numpy(np.transpose(x)).float())
		return x_t

	def discount_rewards(self,r):
		r = np.asarray(r)
		r.reshape(-1,1)
		d_r = r
		sum = 0
		for i in reversed(xrange(r.size-20,r.size)):
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
		d_r = Variable(torch.from_numpy(self.discount_rewards(r)).float())
		o_t = self.forward_all(all_inputs)
		print "OUTPUT MAX: ",np.max(o_t.data.numpy())
		y = np.asarray(y).reshape(-1,1)
		y_t = Variable(torch.from_numpy(y).float())
		self.loss =  (-d_r * (y_t * torch.log(o_t) + (1-y_t) * torch.log (1-o_t))).sum()
		self.optimizer.zero_grad()
		self.loss.backward()
		self.optimizer.step()
		return self.loss.data
