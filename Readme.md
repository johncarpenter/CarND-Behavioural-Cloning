#Behavioral Cloning

###Udacity Self-Driving Car Project #3

This project is a demonstration of deep-learning applied to simulated car navigation. For this project we used a driving simulator to gather steering angles and images from a front-facing camera. Running a deep-learning algorithm on the images we attempted to generated the corresponding steering angle to apply to the simulator.

***Note: This project requires the Udacity driving simulator. This simulator is not yet publicly available.***

####Walkthough

*Demo Video*

[![Video of Performance](http://img.youtube.com/vi/MlIZx79stNk/0.jpg)](http://www.youtube.com/watch?v=MlIZx79stNk)

*Track1*

![GIF of Track1](http://i.giphy.com/5OddbSOQo0Rry.gif)

*Track2*

![GIF of Track2](http://i.giphy.com/cktuObHgXo2nS.gif)



####Getting Started Running the Simulation

1. Clone the project

  ```
  git clone git@github.com:johncarpenter/CarND-Behavioural-Cloning.git
  ```

2. Install the dependencies.

  This project is built on Keras/Tensorflow, and most easily managed with the Anaconda environment manager.

  Install the dependencies in Linux/OSX using;

  ```
  conda env create -f environment.yml
  source activate BehavioralCloning
  ```

  If you elect to build it the hard way or on windows, the dependencies are all in the environment.yml file.

3. Run the simulator

There are two model files included in the project directory ```model.json``` and ```udacity-model.json```. The program will search for the ```.h5``` weights file that is named according to the model.

```python
python drive.py [model].json
```

and execute the simulator. ***Note: The model requires a fairly high frame rate and good or better quality graphics output. Without that the performance drops significantly***

####Building a weights model

Using the source provided you can rebuild the deep-learning module with either generated or provided driving data. I have provided links to the two data sets I used. Editing the ```model.py``` file you can choose the correct csv and IMG directories to process.

[Udacity Raw Data - Track 1](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)

[Self-Generated Data - Track 1 and 2](http://static.2linessoftware.com/data.zip)

All the parameters for adjust the model and reading in the datasets are located within the ```model.py``` code. Once that is configured you can execute the model with the following command


```python
python model.py
```

### Architecture of Deep Learning module

The model chosen for this analysis was based upon a research report produced by [NVIDIA] (https://arxiv.org/pdf/1604.07316v1.pdf). Other models include pretrained VGG16 and another custom model, they are included in the code for reference purposed only.

![ConvNet Diagram](https://i.imgur.com/dgmlseC.png)

The model performed well enough in practice and was able to complete the simulation with a moderately sized dataset.

*Preprocessing*

Preprocessing the images proved to be very effective at speeding up the training and arriving at a better result. The following steps were taken for preprocessing the images. In order;

1. *Crop* The top 54 pixels were cropped out of the image. They didn't contain any road geometry so they were not required
2. *Resize* The images were resized to 80x80. This was chosen after discussion with other class members based upon their results. Smaller images were also easier to manage
3. *Augment* During the training images were augmented with additional samples;
  a. ***channel shift*** Color range was shifted slightly to handle different lighting conditions
  b. ***width shift*** Images were shifted in width to help center the car on the track
  c. ***flip image*** Flipping the steering angle and image was very effective at supplementing all the data

There was a number of experiments with different augmentations. Namely using the left and right images and some smoothing of the steering angle output data sets. In both cases the augmentations didn't seem to improve the accuracy of the model and often caused some issues with noisy data.

Generally the best results were achieved with 2-3 * num samples for augmentation

Sample Image (BGR)

![sample image](http://i.imgur.com/bfADnS0.png)

*Training*

The training algorithm used a simple Mean-Squared Error algorithm with the Adam(loss=0.0001) function. There wasn't much experimentation on the algorithms as the results converged fairly well in practice.

A checkpoint function and a early stopping algorithms were added into the code. The checkpoint function ensured that only the best weights were saved from the training set, and were updated only when there was an improvement. The early stopping algorithm monitored the loss function and if there didn't seem to be an improvement would halt the iterations. This should prevent overfitting of the data

The data was split into training, validation and testing sets prior to any augmentation. Only the training set was augmented. 15% of the data was reserved for validation and of that set, 25% was reserved for testing.

*iterations*

The code is designed to incrementally add additional training sets. By re-running the ```model.py``` with an existing ```model.json``` and ```model.h5``` files will build upon those existing training sets.
