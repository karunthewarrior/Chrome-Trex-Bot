import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# from flask import Flask, render_template, request, redirect, Response
import flask as flk
import random, json
import os
# from torchRL import RLModel

# for linking openCV on Mac with VENV
# export PYTHONPATH=/usr/local/lib/python2.7/site-packages:$PYTHONPATH

app = flk.Flask(__name__)
count = 0

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
    global count
    # read json and send action
    game_data = dict(flk.request.form)

    observation = dataURL2img(game_data['data'][0])
    reward = float(game_data['reward'][0])
    done = game_data['state'][0] == 'true'

    if (count == 0):
        action = 'DOWN'
    elif (count%100):
        action = 'UP'
    else:
        action = 'RUNNING'

    count += 1

    return action

if __name__ == "__main__":
    app.run("0.0.0.0", "8000")
