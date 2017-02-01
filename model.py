
import os
import sys
import threading
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random
import json

from sklearn.model_selection import train_test_split

from keras.preprocessing import image
from keras.models import Model,Sequential, model_from_json
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D
from keras.layers import Activation, Flatten, Dense
from keras.applications import VGG16
from keras.layers import Input, Lambda
from keras.layers import Dropout, BatchNormalization, ELU
from keras.optimizers import Adam
from keras.regularizers import l2

from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.utils import shuffle

import utils
from generator import RegressionImageDataGenerator

datagen = RegressionImageDataGenerator(
    channel_shift_range=0.2,
    width_shift_range=0.2,
    width_shift_value_transform=lambda val, shift: val - shift,
    horizontal_flip=True,
    horizontal_flip_value_transform=lambda val: -val)


def prepare_model_vgg16(input_shape=(224,224,3)):
    """
    Pretrained VGG16 model with fine-tunable last two layers
    Wasn't used in testing
    """
    input_image = Input(shape = (224,224,3))

    base_model = VGG16(input_tensor=input_image,include_top=False)

    for layer in base_model.layers[:-3]:
        layer.trainable = False

    W_regularizer = l2(0.01)

    x = base_model.get_layer("block5_conv3").output
    x = AveragePooling2D((2, 2))(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(100, activation="elu", W_regularizer=l2(0.01))(x)
    x = Dense(1, activation="linear")(x)


    model = Model( input=input_image, output=x)
    return model

"""
Model based on Nvidia paper https://arxiv.org/pdf/1604.07316v1.pdf
Images resized to 80,80 instead of 60,200
"""
def prepare_model_nvidia(input_shape=(80,80,3)):

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=input_shape))
    model.add(Conv2D(24, 5, 5, subsample=(2, 2), border_mode="valid"))
    model.add(ELU())
    model.add(Conv2D(36, 5, 5, subsample=(2, 2), border_mode="valid"))
    model.add(ELU())
    model.add(Conv2D(48, 5, 5, subsample=(2, 2), border_mode="valid"))
    model.add(ELU())
    model.add(Conv2D(64, 3, 3, subsample=(1, 1), border_mode="valid"))
    model.add(ELU())
    model.add(Conv2D(64, 3, 3, subsample=(1, 1), border_mode="valid"))
    model.add(ELU())
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(100))
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(50))
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(10))
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(1,activation='tanh'))

    return model

'''
Simple evaluation model. Wasn't sufficient to model this data set accuractely
'''
def prepare_model(input_shape=(80,80,3)):
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=input_shape))
    model.add(Conv2D(16, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Conv2D(24, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Conv2D(36, 2, 2, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Conv2D(48, 2, 2, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(10))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1, activation="tanh"))
    return model



def train(model, train_generator,validation_generator, additional_samples = 1):
    print("Training-------------------------")
    checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto')
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

    return model.fit_generator(
        train_generator,
        samples_per_epoch=train_generator.N * additional_samples,
        nb_epoch=10, # it will auto stop
        verbose=1,
        validation_data=validation_generator,
        nb_val_samples=validation_generator.N,
        callbacks=[checkpoint,early_stopping])

def evaluate(model, test_generator):
    print("Testing--------------------------")
    metrics = model.evaluate_generator(test_generator,test_generator.N)

    print("%s: %.4f" % (model.metrics_names[0], metrics[0]))
    print("%s: %.2f%%" % (model.metrics_names[1], metrics[1]*100))

    return metrics


def save(model, prefix):
    """save model for future inspection and continuous training
    """
    model_file = prefix + ".json"
    with open(model_file, 'w') as fp:
        fp.write(model.to_json())

def render_results(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()



if __name__ == '__main__':
    #redirect logs to file
    old_stdout = sys.stdout

    log_file = open("processing.log","w")

    sys.stdout = log_file

    # import model and wieghts if exists
    try:
        with open('model.json', 'r') as jfile:
            model = model_from_json(jfile.read())
                # import weights
        model.load_weights('model.h5')

        print("Imported model and weights")

    # If the model and weights do not exist, create a new model
    except Exception as error:
        print("Contructing new model")
        model = prepare_model_nvidia(input_shape=(80,80,3))

    model.summary()

    save(model,'umodel')

    model.compile(loss='mse',
        optimizer=Adam(lr=0.0001),
        metrics=['accuracy'])

    log_paths = [
        ("./data/track1-recovery/driving_log.csv","./data/track1-recovery/IMG/")]
        #("./data/track1-recovery/driving_log.csv","./data/track1-recovery/IMG/"),
	    #("./data/track1/driving_log.csv","./data/track1/IMG/"),
        #("./data/track2/driving_log.csv","./data/track2/IMG/"),
        #("./data/track2-b/driving_log.csv","./data/track2-b/IMG/"),
        #("./data/track2-recovery/driving_log.csv","./data/track2-recovery/IMG/")]

    image_resize = (80,80)

    images = []
    angles = []

    for path in log_paths:

        log_path = path[0]
        img_path = path[1]

        data = utils.read_drive_log(path=log_path)
        n_samples = data.shape[0]
        print("Read {} Records.".format(n_samples))

        center_imgs = np.asarray(data['center'])
        left_imgs = np.asarray(data['left'])
        right_imgs = np.asarray(data['right'])

        steer = np.asarray(data['steering_angle'])

        tmp_images = []
        tmp_angles = []

        for index, angle in enumerate(steer):
            tmp_images.append(img_path + os.path.basename(center_imgs[index]))
            tmp_angles.append(angle)
            #tmp_images.append(img_path + os.path.basename(left_imgs[index]))
            #tmp_angles.append(angle-0.2)
            #tmp_images.append(img_path + os.path.basename(right_imgs[index]))
            #tmp_angles.append(angle+0.2)

        # Smoothing data didn't increase accuracy but left here as a reference
        #tmp_pangles = utils.smooth_data(tmp_angles,window=5)



        images += tmp_images
        angles += tmp_angles


    #file = img_path + os.path.basename(images[0])
    #plt.imshow(utils.load_img_from_file(file,target_size=(80,80)))
    #plt.show()

    images = np.asarray(images)
    angles = np.asarray(angles,dtype=np.float64)

    print("Processing {} Records.".format(angles.shape[0]))

    images,  angles = shuffle(images, angles, random_state=0)

    X_train,test_images,y_train,test_angles = train_test_split(images,angles,test_size=0.15)

    X_validation,X_test,y_validation,y_test = train_test_split(test_images,test_angles,test_size=0.25)

    print("Number of Training Images:",X_train.shape[0])
    print("Number of Training Steering Angles:",y_train.shape[0])
    print("Number of Validation Images:",X_validation.shape[0])
    print("Number of Validation Steering Angles:",y_validation.shape[0])

    print("Number of Test Images:",X_test.shape[0])
    print("Number of Test Steering Angles:",y_test.shape[0])

    train_generator = utils.steering_angle_generator(X_train,y_train,target_size=image_resize,image_generator=datagen)
    val_generator = utils.steering_angle_generator(X_validation,y_validation,target_size=image_resize)
    test_generator = utils.steering_angle_generator(X_test,y_test,target_size=image_resize)

    history = train(model, train_generator , val_generator, additional_samples=2)
    #render_results(history)

    evaluate(model,test_generator)

    # Close the log file
    sys.stdout = old_stdout

    log_file.close()
