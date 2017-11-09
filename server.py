import sys
import cv2
import os
import numpy as np

import flask as flk
import random, json
import model as m
from time import sleep

app = flk.Flask(__name__)
count = 0
num_episodes = 1
batch_size = 10
obj = m.RLModel()
y = []
r = []
i = 0 
def dataURL2img(dataurl):
    img_b64 = dataurl.split(',')[1]
    img = img_b64.decode('base64')
    pixarr = np.fromstring(img, np.uint8)
    img = cv2.imdecode(pixarr, cv2.IMREAD_UNCHANGED)
    return img

@app.route('/')
def render():
    # serve index template
    return flk.render_template('chrome-dino.html')

@app.route('/receiver', methods = ['POST'])
def update_game():
    # read json and send action
    global count,i,num_episodes,y,r
    count+=1
    game_data = dict(flk.request.form)

    observation = dataURL2img(game_data['data'][0])
    score = float(game_data['reward'][0])
    done = game_data['state'][0] == 'true'

    action = "UP"
    x,all_inputs,reward = obj.pre_process(observation,score,count)
    if(count >1):
        i+=1
        cv2.imshow("input",x.reshape(60,300)*255)
        cv2.waitKey(1)
        # time.sleep(1)
        # print i
        o = obj.forward_pass(x)

        action = "UP" if np.random.uniform() < o else "RUNNING"
        y.append(1) if action == "UP" else y.append(0)
        # if(count == 50):
            # loss = obj.backward_pass(all_inputs,y,r)
        if(done):
            sleep(0.1)
            reward = -score
            count = 0
            num_episodes +=1
            action = "restart"
            print "Episode number: ",num_episodes
        r.append(reward)
        # print len(r),len(y)

        if(num_episodes%batch_size==0):
            print "Backward pass"
            loss = obj.backward_pass(all_inputs,y,r)
            print "LOSS: ",loss
            r = []
            y = []
            obj.clear_inputs()
            num_episodes +=1
    return action

if __name__ == "__main__":
    app.run("0.0.0.0", "8000")
