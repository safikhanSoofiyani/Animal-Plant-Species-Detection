# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 11:29:24 2022

@author: safik
"""
import random
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import os
import math
import argparse

# Importing libraries related to Image Preprocessing
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

from sys import argv

# Importing libraries related to CNN Model building
from tensorflow import keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Activation,Dropout,BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import ResNet50, InceptionV3, InceptionResNetV2, Xception

random.seed(hash("seriously you compete with me") % 2**32 - 1)
np.random.seed(hash("i am mohammed safi") % 2**32 - 1)
tf.random.set_seed(hash("ur rahman khan") % 2**32 - 1)
tf.random.set_seed(hash("ur rahman khan") % 2**32 - 1)

import wandb
from wandb.keras import WandbCallback

input_image_shape = (256, 256, 3)
entity_name = "safi-vamsi-cs6910"
project_name = "Assignment 2"


def get_data():
    '''
    This function is used to get the train and the validation data from the given 
    path. It automatically does the train-val split and reserves 10% of random
    points and returns them as validation dataset
  
    It returns the two dataset objects one each for train and validation
  
    It takes in no parameters 
    '''
    path=r"nature_12K/inaturalist_12K/train"

    
    training_data_augmentation=ImageDataGenerator(rescale=1./255,
                                        validation_split = 0.1)

    # Validation data is not being augmented
    validation_data_augmentation=ImageDataGenerator(
        rescale=1./255,
        validation_split=0.1
    )

    train_dataset=training_data_augmentation.flow_from_directory(path,shuffle=True,seed=19,subset='training')
    valid_dataset=validation_data_augmentation.flow_from_directory(path,shuffle=True,seed=19,subset='validation')

    return train_dataset, valid_dataset


def get_augmented_data():
    
    '''
       This function is used to get the train and the validation data from the given 
       path and augment it with the required specs. It automatically does the train-val 
       split and reserves 10% of random points and returns them as validation dataset
  
    It returns the two dataset iterators one each for train and validation
  
    It takes in no parameters 
    '''
    
    path=r"nature_12K/inaturalist_12K/train"
    training_data_augmentation=ImageDataGenerator(rescale=1./255,
                                        height_shift_range=0.2,
                                        width_shift_range=0.2,
                                        horizontal_flip=True,
                                        zoom_range=0.2,
                                        fill_mode="nearest",
                                        validation_split = 0.1)

    # Validation data is not being augmented
    validation_data_augmentation=ImageDataGenerator(
        rescale=1./255,
        validation_split=0.1
    )

    train_aug=training_data_augmentation.flow_from_directory(path,shuffle=True,seed=19,subset='training')
    valid_aug=validation_data_augmentation.flow_from_directory(path,shuffle=True,seed=19,subset='validation')

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

    
    training_data=ImageDataGenerator(rescale=1./255)

    test_data=ImageDataGenerator(rescale=1./255)

    train_dataset = training_data.flow_from_directory(path_train,shuffle=True,seed=19)
    test_dataset = test_data.flow_from_directory(path_test,shuffle=True,seed=19)

    #return train_aug, valid_aug

    
    #return train_data, valid_data
    return train_dataset, test_dataset


def get_test_augmented_data():
    
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

    # test data is not being augmented
    test_data_augmentation=ImageDataGenerator(
        rescale=1./255
        )

    train_aug=training_data_augmentation.flow_from_directory(path_train,shuffle=True,seed=19)
    test_aug=test_data_augmentation.flow_from_directory(path_test,shuffle=True,seed=19)

    return train_aug, test_aug




def build_model(model_name, dense_activation, dense_layer, dropout, trainable_layers):
    '''
    This function is actually used to build the CNN Model with the given specifications
  
    It returns the model
  
    The various parameters are as listed below
  
    # model_name          : "String" name of the pretrained model
    # dense_activation    : "String" acitvation used for densely connected layer
    # dense_layer         : "Integer" number of neurons in the dense layer
    # dropout             : "float or double" specify the dropout % for regularization 
    # trainable_layers    : "int" number of layers to be unfrozen
    '''
    input_image_shape = (256, 256, 3)
    input_layer = Input(shape = input_image_shape)

    # add a pretrained model without the top dense layer
    if model_name == 'ResNet50':
      pretrained_model = ResNet50(include_top = False, weights='imagenet',input_tensor = input_layer)
    elif model_name == 'InceptionV3':
      pretrained_model = InceptionV3(include_top = False, weights='imagenet',input_tensor = input_layer)
    elif model_name == 'InceptionResNetV2':
      pretrained_model = InceptionResNetV2(include_top = False, weights='imagenet',input_tensor = input_layer)
    else:
      pretrained_model = Xception(include_top = False, weights='imagenet',input_tensor = input_layer)
    
    #First freezing all the layers
    for layer in pretrained_model.layers:
        layer.trainable=False 
    
    #Unfreezing some of the top layers
    if trainable_layers != 0:
      for layer in pretrained_model.layers[-trainable_layers:]:
        layer.trainable=True

    model = Sequential()
    model.add(pretrained_model)
    #add pretrained model
    model.add(Flatten()) 
    # The flatten layer is essential to convert the feature map into a column vector
    model.add(Dense(dense_layer, activation=dense_activation))
    #add a dense layer
    model.add(Dropout(dropout)) 
    # For dropout
    model.add(Dense(10, activation="softmax"))#softmax layer

    return model


def wandb_train():
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
        "model_name": "InceptionV3",
        "data_augmentation": True,
        "dense_layer": 128,
        "dropout": 0.0,
        "trainable_layers": 0,
        "batch_size": 64,
        "num_epochs": 10
        
        }
   
    # Initialize a new wandb run
    wandb.init(config=config_defaults)
    
    # collecting the configuration information
    config = wandb.config

    # Local variables, values obtained from wandb config
    model_name = config.model_name
    data_augmentation = config.data_augmentation
    dense_layer = config.dense_layer
    dropout = config.dropout
    trainable_layers = config.trainable_layers
    batch_size = config.batch_size
    num_epochs = config.num_epochs
    dense_activation = "relu"
    
    
    # run name
    run_name='Model_{}_trainable_{}_Data_aug_{}_dropout_{}_dense_layer_{}'.format(config.model_name,
                                                                             config.trainable_layers,
                                                                             config.data_augmentation,
                                                                             config.dropout,
                                                                             config.dense_layer,                                                                       
                                                                             )

    wandb.run.name = run_name

    # Create the data generators

    if data_augmentation == False:
        train_data, validation_data = get_data()
    else:
        train_data, validation_data = get_augmented_data()
    
    # Define the model
    model = build_model(model_name, dense_activation, dense_layer, dropout, trainable_layers)


    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss=tf.keras.losses.CategoricalCrossentropy(name='loss'),
                metrics=[tf.keras.metrics.CategoricalAccuracy(name='acc')])


    # To save the model with best validation accuracy
    mc = ModelCheckpoint('nature_12K/best_model.h5', monitor='val_accuracy', mode='max', verbose=0, save_best_only=True)

    history = model.fit(train_data,
                        validation_data = validation_data,
                        epochs=num_epochs, 
                        callbacks=[WandbCallback()]
                        )
    
    
    # Meaningful name for the run

    wandb.run.finish()
    return history






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
    
    hyperparameters = {

        "model_name":{
            'values': ["InceptionV3", "ResNet50", "InceptionResNetV2", "Xception"]
        },

        "data_augmentation": {
            "values": [True, False]
        },

        "dense_layer": {
            "values": [64, 128, 256, 512]
        },

        "dropout": {
            "values": [0.0, 0.1, 0.2, 0.3]
        },
        
        "trainable_layers": {
            "values": [0, 10, 15, 20]
        },

        "batch_size": {
            "values": [64, 128]
        },

        "num_epochs":{
            "values": [5, 10, 15]
        }
        
    }
    
    sweep_config = {

        "method": "bayes",
        "metric": {
                "name":"val_acc",
                "goal": "maximize"
                },
  
        "parameters": hyperparameters
        }

    sweep_id = wandb.sweep(sweep_config, entity=entity_name, project=project_name)
    wandb.agent(sweep_id, wandb_train)
    
    
def testing(entity_name, project_name):
    
    '''
    This function is used to run the best model on the test dataset.
    It creates a model using the best set of hyperparameters, 
    trains the model again on all the train data and evaluates it using the test data
    
    It takes the entity name and project name as paramters
    
    It doesnt return anything
    '''

    best_hyperparameters = {
        "model_name": "Xception",
        "data_augmentation": True,
        "dense_layer": 128,
        "dropout": 0.3,
        "trainable_layers": 20,
        "batch_size": 64,
        "num_epochs": 15
        
    }

    # Initialize a new wandb run
    wandb.init(config=best_hyperparameters, project = project_name, entity=entity_name)
    
    # collecting the configuration information
    config = wandb.config

    # Local variables, values obtained from wandb config
    model_name = config.model_name
    data_augmentation = config.data_augmentation
    dense_layer = config.dense_layer
    dropout = config.dropout
    trainable_layers = config.trainable_layers
    batch_size = config.batch_size
    num_epochs = config.num_epochs
    dense_activation = "relu"
    
    
    # run name
    run_name='test run - partb'

    wandb.run.name = run_name

    # Create the data generators

    if data_augmentation == False:
        train_data, test_data = get_test_data()
    else:
        train_data, test_data = get_test_augmented_data()
    
    # Define the model
    model = build_model(model_name, dense_activation, dense_layer, dropout, trainable_layers)

    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss=tf.keras.losses.CategoricalCrossentropy(name='loss'),
                metrics=[tf.keras.metrics.CategoricalAccuracy(name='acc')])


    # To save the model with best validation accuracy
    model.save("best_pretrained_model.h5")
    history = model.fit(train_data,
                        epochs=num_epochs, 
                        callbacks=[WandbCallback()]
                        )
    
    test_loss, test_accuracy = model.evaluate(x = test_data)
    print(test_loss)
    print(test_accuracy)
    wandb.log({'test_accuracy':test_accuracy, 'test_loss':test_loss})
    
    
    # Meaningful name for the run

    
    wandb.run.finish()
    return history



def testing_cnn(model_name, dropout, data_augmentation, num_epochs, batch_size, dense_layer, learning_rate, trainable_layers, dense_activation):

    '''
    This function is used to train the model based on the hyperparameters given
    by the user and evaluate this model on the test data. It doesnt do validation 
    split as it is not doing cross validation to find the best set of hyperparamters.
    It doesnt log anything to wandb.
    
    It takes the below parameters
    # model_name          : "string" name of the pretrained model to be chosen
    # dropout             : "float or double" specify the dropout % for regularization (in decimals)
    # data_augmentation   : "Boolean" whether to augment data or not
    # num_epochs          : "int" how many epochs to run
    # batch_size          : "int" how many data points in a batch
    # dense_layer         : "int" how many neurons in the dense layer
    # learning_rate       : "float" what learning rate to use to train the model
    # trainable_layers    : "int" number of layers that are to be kept unfrozen
    # dense_activation    : "string" activation function of the dense layer

    
    
    It doesnt return anything
    '''
    
    
    
   
    #tf.keras.backend.clear_session()


    # Define the model
    model = build_model(model_name, dense_activation, dense_layer, dropout, trainable_layers)
    model.summary()

    #Getting the data here
    if data_augmentation:
        train_data, test_data = get_test_augmented_data()
    else:
        train_data, test_data = get_test_data()

    name_run = "Test run 3"
    
    #compiling the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate),
                loss=tf.keras.losses.CategoricalCrossentropy(name='loss'),
                metrics=[tf.keras.metrics.CategoricalAccuracy(name='acc')])
    
    #fitting the model
    history = model.fit(x = train_data,
                  epochs=num_epochs,
                  )
    
    #evaluating the model on test data
    test_loss, test_accuracy = model.evaluate(x = test_data)
    print("The test Loss is :", test_loss)
    print("The test Accuracy is:", test_accuracy)
    
    
    #saving this model
    model.save("best_model_pretrained.h5")
    
    return model


if __name__ == "__main__":
    
    entity_name = "safikhan"
    project_name = "assgn2 trial"
    
    #parsing the various command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', type=str, required=True, help="Do you want to sweep or not: Enter 'yes' or 'no'")
    
    
    parser.add_argument('--model', type=str, required=('no' in argv), help="Pretrained model to use: string value")
    parser.add_argument('--dropout', type=float, required=('no' in argv), help="Dropout: float value")
    parser.add_argument('--dataAugment', type=bool, required=('no' in argv), help="Data Augmentation: True or False")
    parser.add_argument('--numEpochs', type=int, required=('no' in argv), help="Number of Epochs: integer value")
    parser.add_argument('--batchSize', type=int, required=('no' in argv), help="Batch Size: integer value")
    parser.add_argument('--denseLayer', type=int, required=('no' in argv), help="Dense Layer size: integer value")
    parser.add_argument('--learningRate', type=float, required=('no' in argv), help="Learning Rate: float value")
    parser.add_argument('--trainLayers', type=int, required=('no' in argv), help="Number of trainable layers: integer value")
    parser.add_argument('--denseAct', type=str, required=('no' in argv), help="Activation function for the dense layer: string value")
    
    
    args = parser.parse_args()
    
    
    if args.sweep == 'no':
        
        model_name = args.model       
        dropout = args.dropout
        data_augmentation = args.dataAugment
        num_epochs = args.numEpochs
        batch_size = args.batchSize
        dense_layer = args.denseLayer
        learning_rate = args.learningRate
        trainable_layers = args.trainLayers
        dense_activation = args.denseAct
        
        if model_name not in ["InceptionV3", "ResNet50", "InceptionResNetV2", "Xception"]:
            print("Please enter a valid model name from the list given in sirs page")
            exit()
        model = testing_cnn(model_name, dropout, data_augmentation, num_epochs, batch_size, dense_layer, learning_rate, trainable_layers, dense_activation)
        
    else:
        sweeper(entity_name, project_name)
        testing(entity_name, project_name)
        

    
    