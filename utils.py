import random
import pandas as pd
import numpy as np
import os


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator( rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.2,
        fill_mode='nearest')

datagen_blank = ImageDataGenerator()

def read_drive_log(path="./data/driving_log.csv"):
    print("Reading file {}".format(path))
    drive_input = pd.read_csv(path, header=None,names=['center','left','right','steering_angle','throttle','brake','speed'])
    return drive_input

def smooth_data(angles, window=10):
    print("Smoothing Steering Angle Data with window size {}".format(window))
    pangles = pd.DataFrame(angles)
    pangles = pangles.rolling(window=window).mean()
    pangles = pangles.fillna(0)
    return np.asarray(pangles[0],dtype=np.float64)


def steering_angle_generator(filenames, values, target_size=(80,80),path="./data/IMG/"):
    return datagen_blank.flow(load_images(filenames,target_size=target_size,root=path),values)

def load_images(paths, target_size,root="./data/IMG/"):
    images = np.zeros((len(paths), *target_size, 3))
    for i, p in enumerate(paths):

        file = root + os.path.basename(p)

        img = load_img(file, target_size=target_size)
        img = img_to_array(img,dim_ordering='tf')
        images[i] = img

    return images


def load_img_from_file(filename, target_size=(40,40)):
    img = load_img(filename,target_size=target_size)
    x = img_to_array(img,dim_ordering='tf')
    x = x.reshape((1,) + x.shape)
    return x
