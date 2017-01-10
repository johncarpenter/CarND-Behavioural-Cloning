
import os
import threading
import h5py
import numpy as np

import random
import json

from sklearn.model_selection import train_test_split

from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Input, Flatten, Dense, Lambda, merge
from keras.layers import Dropout, BatchNormalization, ELU

from keras.callbacks import EarlyStopping, ModelCheckpoint

import utils

def prepare_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=( 160, 320, 3)))
    model.add(Conv2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Conv2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Conv2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    return model



def train(model, train_generator,validation_generator):

    checkpoint = ModelCheckpoint('sample.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto')
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

    return model.fit_generator(
        train_generator,
        samples_per_epoch=train_generator.N,
        nb_epoch=2,
        verbose=1,
        validation_data=validation_generator,
        nb_val_samples=validation_generator.N)

def save(model, prefix):
    """save model for future inspection and continuous training
    """
    model_file = prefix + ".json"
    weight_file = prefix + ".h5"
    with open(model_file, 'w') as fp:
        fp.write(model.to_json())
    model.save_weights(weight_file)


if __name__ == '__main__':
    model = prepare_model()
    model.summary()

    model.compile(loss='mse',
        optimizer='adam',
        metrics=['accuracy'])

    data = utils.read_drive_log()
    n_samples = data.shape[0]
    print("Read {} Records.".format(n_samples))

    X_train,X_validation,y_train,y_validation = train_test_split(data['center'],data['steering_angle'],test_size=0.4)

    print("Number of Images:",X_train.shape[0])
    print("Number of Steering Angles:",y_train.shape[0])

    train_generator = utils.steering_angle_generator(X_train,y_train)
    val_generator = utils.steering_angle_generator(X_validation,y_validation)

    history = train(model, train_generator , val_generator)
    print(history)

    save(model,'sample')
