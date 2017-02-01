#Behavioral Cloning

###Udacity Self-Driving Car Project #3

This project is a demonstration of deep-learning applied to simulated car navigation. For this project we used a driving simulator to gather steering angles and images from a front-facing camera. Running a deep-learning algorithm on the images we attempted to generated the corresponding steering angle to apply to the simulator.

***Note: This project requires the Udacity driving simulator. This simulator is not yet publicly available.***

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

and execute the simulator. ***Note: The model requires a fairly high frame rate and good or better quality graphics output. Without that the performance drops significantly ***

####Building a weights model

Using the source provided you can rebuild the deep-learning module with either generated or provided driving data. I have provided links to the two data sets I used. Editing the ```model.py``` file you can choose the correct csv and IMG directories to process.

[Udacity Raw Data - Track 1](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)

[Self-Generated Data - Track 1 and 2](http://static.2linessoftware.com/data.zip)

All the parameters for adjust the model and reading in the datasets are located within the ```model.py``` code. Once that is configured you can execute the model with the following command


```python
python model.py
```

### Architecture of Deep Learning module

The model chosen for this analysis was based upon a research report produced by [NVIDIA] (https://arxiv.org/pdf/1604.07316v1.pdf). The

![ConvNet Diagram](https://i.imgur.com/dgmlseC.png)
