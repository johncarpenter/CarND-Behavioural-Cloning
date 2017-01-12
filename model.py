
import os
import threading
import h5py
import numpy as np

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

import utils


def prepare_model_vgg16():
    """Pretrained VGG16 model with fine-tunable last two layers
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
"""
def prepare_model_nvidia():

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=( 80, 80, 3)))
    model.add(Conv2D(24, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Conv2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Conv2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Conv2D(64, 3, 3, subsample=(1, 1), border_mode="same"))
    model.add(ELU())
    model.add(Conv2D(64, 3, 3, subsample=(1, 1), border_mode="same"))
    model.add(ELU())
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dropout(.8))
    model.add(ELU())
    model.add(Dense(100))
    model.add(Dropout(.8))
    model.add(ELU())
    model.add(Dense(50))
    model.add(Dropout(.8))
    model.add(ELU())
    model.add(Dense(10))
    model.add(Dropout(.8))
    model.add(ELU())
    model.add(Dense(1,activation='linear'))

    return model

def prepare_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=( 80, 80, 3)))
    model.add(Conv2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Conv2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Conv2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(1024))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    return model



def train(model, train_generator,validation_generator):

    checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto')
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

    return model.fit_generator(
        train_generator,
        samples_per_epoch=train_generator.N,
        nb_epoch=5,
        verbose=1,
        validation_data=validation_generator,
        nb_val_samples=validation_generator.N,
        callbacks=[checkpoint,early_stopping])

def save(model, prefix):
    """save model for future inspection and continuous training
    """
    model_file = prefix + ".json"
#    weight_file = prefix + ".h5"
    with open(model_file, 'w') as fp:
        fp.write(model.to_json())
#    model.save_weights(weight_file)


if __name__ == '__main__':
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
        model = prepare_model_nvidia()

    model.summary()

    save(model,'model')

    model.compile(loss='mse',
        optimizer=Adam(lr=0.001),
        metrics=['accuracy'])

    data = utils.read_drive_log()
    n_samples = data.shape[0]
    print("Read {} Records.".format(n_samples))

    center_imgs = np.asarray(data['center'])
    steer = np.asarray(data['steering_angle'])

    X_train,X_validation,y_train,y_validation = train_test_split(center_imgs,steer,test_size=0.25)

    print("Number of Images:",X_train.shape[0])
    print("Number of Steering Angles:",y_train.shape[0])

    train_generator = utils.steering_angle_generator(X_train,y_train)
    val_generator = utils.steering_angle_generator(X_validation,y_validation)

    history = train(model, train_generator , val_generator)
