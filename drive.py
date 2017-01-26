import argparse
import base64
import json
import csv

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

import utils

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None
writer = None


@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image = utils.preprocess(image,target_size=(80,80))
    image_array = np.asarray(image)
    transformed_image_array = image_array[None, :, :, :]

    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    new_steering_angle = float(model.predict(transformed_image_array, batch_size=1))

    # Simple filter. Makes the output better but obsures the results from the model. 
    #new_steering_angle = 0.20*(float(steering_angle)/100) + 0.80*(new_steering_angle)

    # Testing slow down when the angle is higher, this limits oscillations and errors
    #throttle = 0.4 if abs(float(new_steering_angle)) < 0.05 else 0.2

    # Hardware limitations on the emulator machine force me to limit the speed since
    # it increases the FPS for the testing
    throttle = 0.4 if abs(float(speed)) < 15 else 0.0
    print( new_steering_angle, throttle, speed)
    send_control(new_steering_angle, throttle)
    #write_data(new_steering_angle, throttle, speed)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)

def write_data(steering_angle, throttle, speed):
    writer.writerow([steering_angle, throttle, speed])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        #   model = model_from_json(json.loads(jfile.read()))\
        #
        # instead.
        model = model_from_json(jfile.read())

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
