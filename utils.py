import random
import pandas as pd
import numpy as np


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator()

def read_drive_log():
    # todo move to input parameter
    drive_input = pd.read_csv('driving_log.csv', header=None,names=['center','left','right','steering_angle','throttle','brake','speed'])
    return drive_input


def steering_angle_generator(filenames, values):
    return datagen.flow(load_images(filenames,target_size=(160, 320)),values)

def load_images(paths, target_size):
    images = np.zeros((len(paths), *target_size, 3))
    for i, p in enumerate(paths):
        img = load_img(p, target_size=target_size)
        img = img_to_array(img, dim_ordering='tf')
        images[i] = img

    return images


def load_img_from_file(filename, target_size=(320,160)):
    img = load_img(filename,target_size=target_size)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    return x
