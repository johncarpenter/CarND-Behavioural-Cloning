import random
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from generator import RegressionImageDataGenerator

datagen = RegressionImageDataGenerator(horizontal_flip=True,
    horizontal_flip_value_transform=lambda val: -val,)

datagen_blank = RegressionImageDataGenerator()

def read_drive_log(path="./data/driving_log.csv"):
    print("Reading file {}".format(path))
    drive_input = pd.read_csv(path, header=None,names=['center','left','right','steering_angle','throttle','brake','speed'])
    return drive_input

def smooth_data(angles, window=10):
    print("Smoothing Steering Angle Data with window size {}".format(window))
    pangles = pd.DataFrame(angles)
    pangles = pangles.rolling(window=window).mean()
    pangles = pangles.fillna(0)
    return list(pangles[0])


def steering_angle_generator(filenames, values, target_size=(80,80),path="./data/IMG/"):
    return datagen.flow(load_images(filenames,target_size=target_size,root=path),values)


def load_images(paths, target_size,root="./data/IMG/"):
    images = np.zeros((len(paths), *target_size, 3))
    for i, p in enumerate(paths):

        file = root + os.path.basename(p)

        images[i] = load_img_from_file(file,target_size=target_size)

    return images

def load_img_from_file(filename, target_size=(80,80)):
    img = load_img(filename)
    img = preprocess(img,target_size)
    img = img_to_array(img,dim_ordering='tf')
    img = img[:,:,::-1]
    return img

def preprocess(img, target_size=(80,80)):

    img = crop_image(img)
    img = img.resize(target_size)
    return img

def crop_image(img):
    w,h = img.size
    return img.crop((0,60,w,h))
