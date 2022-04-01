# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 16:32:26 2022

@author: safik
"""

import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import os
import sys
import argparse

# Importing libraries related to Image Preprocessing
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

# Importing libraries related to CNN Model building
from tensorflow import keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Activation,Dropout,BatchNormalization
from keras.models import Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image

from sys import argv

random.seed(hash("seriously you compete with me") % 2**32 - 1)
np.random.seed(hash("i am mohammed safi") % 2**32 - 1)
tf.random.set_seed(hash("ur rahman khan") % 2**32 - 1)


#!pip install wandb
import wandb

from wandb.keras import WandbCallback

#Declaring some global variables
img_shape=(256,256,3)
entity_name = "safikhan"
project_name = "assgn2 trial"



def build_cnn(conv_activation , dense_activation, num_filters, conv_filter_size, pool_filter_size, batch_norm, dense_layer, dropout):
  
  '''
  This function is actually used to build the CNN Model with the given specifications
  
  It returns the model
  
  The various parameters are as listed below
  
  # conv_activation     : "List" activation used for convolution layer
  # dense_act           : "String" acitvation used for densely connected layers
  # num_filters         : "List" number of activation filters for each layer
  # conv_filter_size    : "List" kernel sizes for convultion layers
  # pool_filter_size    : "List" kernel sizes for maxpooling layers
  # batch_norm          : "Boolean" set to True, if you are using batch normalization
  # dim_final           : "Integer" dimensionality of output space after 5 blocks of convultion, maxpooling blocks
  # dropout             : "float or double" specify the dropout % for regularization (in decimals)
  '''
  model=Sequential()
  # Adding the first conv,activ,maxpool layers
  model.add(Conv2D(filters=num_filters[0],
                   kernel_size=conv_filter_size[0],
                   input_shape=img_shape))
  
  # Add batch normalization layer, if the user specifies
  if batch_norm:
    model.add(BatchNormalization())
  
  model.add(Activation(conv_activation[0]))
  model.add(MaxPool2D(pool_size=pool_filter_size[0],strides=(2,2)))

  # Adding the next 4 layer blocks
  for i in range(1,5):
    model.add(Conv2D(filters=num_filters[i],kernel_size=conv_filter_size[i]))
    if batch_norm:
      model.add(BatchNormalization())
    model.add(Activation(conv_activation[i]))
    model.add(MaxPool2D(pool_size=pool_filter_size[i],strides=(2,2)))
  
  # Flattening the feature map to a column vector
  model.add(Flatten())
  model.add(Dense(units=dense_layer ,activation = dense_activation))
  # Adding dropout regularization
  model.add(Dropout(dropout))
  # Adding the final Dense layer with dimensions equal to number of classes and softmax activation
  model.add(Dense(10,activation="softmax"))

  return model



def get_data():
    
    '''
  This function is used to get the train and the validation data from the given 
  path. It automatically does the train-val split and reserves 10% of random
  points and returns them as validation dataset
  
  It returns the two dataset objects one each for train and validation
  
  It takes in no parameters 
  '''
    path=r"nature_12K/inaturalist_12K/train"

    # Training dataset:
    train_dataset=tf.keras.preprocessing.image_dataset_from_directory(
        directory=path,
        labels='inferred',
        label_mode='categorical',
        color_mode='rgb',
        batch_size=32,
        image_size=(256,256),
        shuffle=True,
        validation_split=0.1,
        subset='training'
    )
    # Validation dataset:
    valid_dataset=tf.keras.preprocessing.image_dataset_from_directory(
        directory=path,
        labels='inferred',
        label_mode='categorical',
        color_mode='rgb',
        batch_size=32,
        image_size=(256,256),
        shuffle=True,
        validation_split=0.1,
        subset='validation'
    )
    
    #return train_data, valid_data
    return train_dataset, valid_dataset




def get_augmented_data():
    
    '''
       This function is used to get the train and the validation data from the given 
       path and augment it with the required specs. It automatically does the train-val 
       split and reserves 10% of random points and returns them as validation dataset
  
    It returns the two dataset iterators one each for train and validation
  
    It takes in no parameters 
    '''

    path = r"nature_12K/inaturalist_12K/train"

    training_data_augmentation=ImageDataGenerator(rescale=1./255,
                                        height_shift_range=0.2,
                                        width_shift_range=0.2,
                                        horizontal_flip=True,
                                        zoom_range=0.2,
                                        fill_mode="nearest",
                                        validation_split = 0.1)

    # Validation data is not being augmented
    validation_data_augmentation=ImageDataGenerator(
        validation_split=0.1
    )

    train_aug=training_data_augmentation.flow_from_directory(path, shuffle=True,
                                                             subset='training')
    valid_aug=validation_data_augmentation.flow_from_directory(path,shuffle=True,
                                                             subset='validation')

    return train_aug, valid_aug


def get_test_data():
    
    '''
    This function is used to get the train and the test data from the given 
    path. It uses the whole train folder for train data and the whole val 
    folder for test data
  
    It returns the two dataset objects one each for train and test
  
    It takes in no parameters 
    '''
    
    path_train = r"nature_12K/inaturalist_12K/train"
    path_test = r"nature_12K/inaturalist_12K/val"
    # Training dataset:
    train_dataset=tf.keras.preprocessing.image_dataset_from_directory(
        directory=path_train,
        labels='inferred',
        label_mode='categorical',
        color_mode='rgb',
        batch_size=32,
        image_size=(256,256),
        shuffle=True
    )
    # Test dataset:
    test_dataset=tf.keras.preprocessing.image_dataset_from_directory(
        directory=path_test,
        labels='inferred',
        label_mode='categorical',
        color_mode='rgb',
        #batch_size=32,
        image_size=(256,256),
        shuffle=True
    )
    
    #return train_data, valid_data
    return train_dataset, test_dataset


def get_augmented_test_data():
    
    '''
         This function is used to get the train and the test data from the given 
         path and augment it with the required specs. It uses the whole train
         data folder and return the train data with augmentation.
         It doesnt augment the test data
  
        It returns the two dataset iterators one each for train and test datsets
  
        It takes in no parameters 
    '''

    path_train = r"nature_12K/inaturalist_12K/train"
    path_test = r"nature_12K/inaturalist_12K/val"

    training_data_augmentation=ImageDataGenerator(rescale=1./255,
                                        height_shift_range=0.2,
                                        width_shift_range=0.2,
                                        horizontal_flip=True,
                                        zoom_range=0.2,
                                        fill_mode="nearest")

    # Validation data is not being augmented
    test_data_augmentation=ImageDataGenerator(rescale=1. /255
                                              )

    train_aug=training_data_augmentation.flow_from_directory(path_train, shuffle=True,
                                                             )
    test_aug=test_data_augmentation.flow_from_directory(path_test,shuffle=True,
                                                             )

    return train_aug, test_aug




def train():
    
    '''
    This function is used to train the Convolutional neural
    networks by using wandb. It fetches the required hyperparameters from the 
    wandb config dictionary and calls the build_cnn function, then compiles the
    model and fits the train data set while doing cross validation with the 
    validation dataset.
    It logs the accuracy and loss on wandb under the appropriate run name
    
    It doesnt take any parameters
    
    It doesnt return anything
    '''
    
    
    config_defaults = {
      'batch_norm': True,
      'num_filters': 32,
      'filter_org': 0.5,
      'dropout': 0.0,
      'data_augmentation': True,
      'num_epochs' : 10,
      'batch_size': 64,
      'dense_layer': 64,
      'learning_rate': 0.001,
      'kernel_size': 3
      
    }

  # Initializing the wandb run
    wandb.init(config=config_defaults)
    config = wandb.config


    #preparing the hyperparameters for the build function
    conv_activation = ["relu","relu","relu","relu","relu"]
    dense_activation = "relu"

    #preparing the number of filters for each layer
    num_filters = []
    filters = config.num_filters
    for i in range(5):
        num_filters.append(filters)
        filters = filters * config.filter_org
    
    #preparing the filter size tuple for each layer
    conv_filter_size = []
    F = config.kernel_size
    for i in range(5):
        conv_filter_size.append((F,F))

    pool_filter_size = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]
    
   
    #tf.keras.backend.clear_session()

    #Creating model architecture here
    model = build_cnn(conv_activation, dense_activation, num_filters, conv_filter_size, 
                      pool_filter_size, config.batch_norm, config.dense_layer, 
                      config.dropout)
    model.summary()

    #Getting the data here
    if config.data_augmentation:
        train_data, valid_data = get_augmented_data()
    else:
        train_data, valid_data = get_data()

    #creating the run name
    name_run = str(config.batch_norm) + "_" + str(config.num_filters) + \
                "_" + str(config.filter_org) + "_" + str(config.dropout) + \
                "_" + str(config.data_augmentation) + "_" + str(config.num_epochs)
    
    wandb.run.name = name_run
    wandb_log = True

    #compiling the model
    model.compile(optimizer = tf.keras.optimizers.Adam(config.learning_rate),
              loss = tf.keras.losses.CategoricalCrossentropy(name='loss'),
              metrics = [tf.keras.metrics.CategoricalAccuracy(name='acc')])
    #fitting the model with the train data
    history = model.fit(x = train_data,
                  epochs=config.num_epochs,
                  validation_data=valid_data,
                  callbacks = [WandbCallback()]
                  )

    #wandb.run.save()
    wandb.run.finish()





def sweeper(entity_name, project_name):
    
    '''
    This function is used to call the sweep functionality of Wandb
    We have defined the possible hyperparameter values and Wandb
    used bayes method to search efficiently over these hyperparamters to 
    get the appropriate hyperparamter. 
    All the results are logged in Wandb
    
    It takes the entity name and project name as paramters
    
    It doesnt return anything
    '''
    
    #Declaring the necessary hyperparameters.
    hyperparameters={
    
    'batch_norm': {
        'values':[True, False]
    },
    'num_filter':{
        'values':[32, 64, 128, 256]

    },

    'filter_org':{
        'values':[0.5, 1, 2]
    },

    'dropout':{
        'values':[0.0, 0.5, 0.6, 0.4]
    },

    'data_augmentation':{
        'values':[True, False]
    },

    'num_epochs': {
        'values':[10, 20, 30]
    },

    'batch_size': {
        'values':[32, 64, 128]
    },

    'dense_layer': {
        'values':[32, 64, 128, 512]
    },

    'learning_rate': {
        'values':[0.001, 0.0001]
    },

    'kernel_size': {
        'values': [3, 5, 7]
    }
    
    }   

    #Using bayes method for hyperparameter sweeps to curb the unnecessary configurations
    sweep_config = {
      'method' : 'bayes',
      'metric' :{
          'name': 'val_acc',
          'goal': 'maximize'
      },
      'parameters': hyperparameters
    }
    
    
    #declaring the sweep id and calling the sweep
    sweep_id = wandb.sweep(sweep_config, entity=entity_name, project=project_name)
    wandb.agent(sweep_id, train)
    
    





def testing(entity_name, project_name):
    
    '''
    This function is used to run the best model on the test dataset.
    It creates a model using the best set of hyperparameters, 
    trains the model again on all the train data and evaluates it using the test data
    
    It takes the entity name and project name as paramters
    
    It doesnt return anything
    '''
    

    best_hyperparameters = {
      'batch_norm': True,
      'num_filters': 32,
      'filter_org': 2,
      'dropout': 0.4,
      'data_augmentation': False,
      'num_epochs' : 10,
      'batch_size': 128,
      'dense_layer': 512,
      'learning_rate': 0.0001,
      'kernel_size': 3
    }

    wandb.init(config=best_hyperparameters, project = project_name, entity=entity_name)
    config = wandb.config

    conv_activation = ["relu","relu","relu","relu","relu"]
    dense_activation = "relu"

    num_filters = []
    filters = config.num_filters
    for i in range(5):
        num_filters.append(filters)
        filters = filters * config.filter_org
    
    conv_filter_size = []
    F = config.kernel_size
    for i in range(5):
        conv_filter_size.append((F,F))

    pool_filter_size = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]
    
   
    #tf.keras.backend.clear_session()

    #Creating model architecture here
    model = build_cnn(conv_activation, dense_activation, num_filters, conv_filter_size, 
                      pool_filter_size, config.batch_norm, config.dense_layer, 
                      config.dropout)
    model.summary()

    #Getting the data here


    if config.data_augmentation:
        train_data, test_data = get_augmented_data()
    else:
        train_data, test_data = get_test_data()

    name_run = "Test run 3"
    
    wandb.run.name = name_run
    wandb_log = True

    model.compile(optimizer = tf.keras.optimizers.Adam(config.learning_rate),
              loss = tf.keras.losses.CategoricalCrossentropy(name='loss'),
              metrics = [tf.keras.metrics.CategoricalAccuracy(name='acc')])

    history = model.fit(x = train_data,
                  epochs=config.num_epochs,
                  callbacks = [WandbCallback()]
                  )
    
    test_loss, test_accuracy = model.evaluate(x = test_data)
    print(test_loss)
    print(test_accuracy)
    wandb.log({'test_accuracy':test_accuracy, 'test_loss':test_loss})

    #model.save('/content/drive/MyDrive/nature_12K')

    #wandb.run.save()
    wandb.run.finish()
    
    model.save("best_model.h5")
    
    return model




    





def testing_cnn(batch_norm, num_filters, filter_org, dropout, data_augmentation, num_epochs, batch_size, dense_layer, learning_rate, kernel_size):

    '''
    This function is used to train the model based on the hyperparameters given
    by the user and evaluate this model on the test data. It doesnt do validation 
    split as it is not doing cross validation to find the best set of hyperparamters.
    It doesnt log anything to wandb.
    
    It takes the below parameters
    # batch_norm          : "Boolean" set to True, if you are using batch normalization
    # num_filters         : "int" number of activation filters for each layer
    # filter_org          : "int" how are filters progressing in each layer
    # dropout             : "float or double" specify the dropout % for regularization (in decimals)
    # data_augmentation   : "Boolean" whether to augment data or not
    # num_epochs          : "int" how many epochs to run
    # batch_size          : "int" how many data points in a batch
    # dense_layer         : "int" how many neurons in the dense layer
    # learning_rate       : "float" what learning rate to use to train the model
    # kernel_size         : "int" the size of each kernel
    
    
    It doesnt return anything
    '''
    
    #defining the activation function for each conv layer
    conv_activation = ["relu","relu","relu","relu","relu"]
    dense_activation = "relu"
    
    #print(filter_org)
    #print(type(filter_org))
    
    #arranging the filters for each layer
    num_filter = []
    filters = num_filters
    for i in range(5):
        num_filter.append(filters)
        filters = filters * filter_org
    
    #creating filter tuples for each image
    conv_filter_size = []
    F = kernel_size
    for i in range(5):
        conv_filter_size.append((F,F))

    pool_filter_size = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]
    
   
    #tf.keras.backend.clear_session()

    #Creating model architecture here
    model = build_cnn(conv_activation, dense_activation, num_filter, conv_filter_size, 
                      pool_filter_size, batch_norm, dense_layer, 
                      dropout)
    model.summary()

    #Getting the data here
    if data_augmentation:
        train_data, test_data = get_augmented_test_data()
    else:
        train_data, test_data = get_test_data()

    name_run = "Test run 3"
    
    #compiling the model
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate),
              loss = tf.keras.losses.CategoricalCrossentropy(name='loss'),
              metrics = [tf.keras.metrics.CategoricalAccuracy(name='acc')])
    
    #fitting the model
    history = model.fit(x = train_data,
                  epochs=num_epochs,
                  )
    
    #evaluating the model on test data
    test_loss, test_accuracy = model.evaluate(x = test_data)
    print("The test Loss is :", test_loss)
    print("The test Accuracy is:", test_accuracy)
    
    
    #saving this model
    model.save("best_model.h5")
    
    return model





if __name__ == "__main__":
    
    entity_name = "safikhan"
    project_name = "assgn2 trial"
    
    #parsing the various command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', type=str, required=True, help="Do you want to sweep or not: Enter 'yes' or 'no'")
    
    parser.add_argument('--batchNorm', type=bool, required=('no' in argv), help="Batch Normalization: True or False")
    parser.add_argument('--numFilters', type=int, required=('no' in argv), help="Number of Filters: integer value")
    parser.add_argument('--filterOrg', type=float, required=('no' in argv), help="Filter Organization: float value")
    parser.add_argument('--dropout', type=float, required=('no' in argv), help="Dropout: float value")
    parser.add_argument('--dataAugment', type=bool, required=('no' in argv), help="Data Augmentation: True or False")
    parser.add_argument('--numEpochs', type=int, required=('no' in argv), help="Number of Epochs: integer value")
    parser.add_argument('--batchSize', type=int, required=('no' in argv), help="Batch Size: integer value")
    parser.add_argument('--denseLayer', type=int, required=('no' in argv), help="Dense Layer size: integer value")
    parser.add_argument('--learningRate', type=float, required=('no' in argv), help="Learning Rate: float value")
    parser.add_argument('--kernelSize', type=int, required=('no' in argv), help="Kernel Size: integer value")
    
    args = parser.parse_args()
    
    
    if args.sweep == 'no':
        batch_norm = args.batchNorm
        num_filters = args.numFilters
        filter_org = args.filterOrg
        #print(filter_org)
        #print(type(filter_org))
        dropout = args.dropout
        data_augmentation = args.dataAugment
        num_epochs = args.numEpochs
        batch_size = args.batchSize
        dense_layer = args.denseLayer
        learning_rate = args.learningRate
        kernel_size = args.kernelSize
        model = testing_cnn(batch_norm, num_filters, filter_org, dropout, data_augmentation, num_epochs, batch_size, dense_layer, learning_rate, kernel_size)
        
    else:
        sweeper(entity_name, project_name)
        testing(entity_name, project_name)
        

    
    
    
    
                      
    
    
        






