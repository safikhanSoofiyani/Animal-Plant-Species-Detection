# CS6910-Assignment-2 : Convolutional Neural Networks
Assignment 2 submission for the course CS6910 Fundamentals of Deep Learning. 

Team Members : Vamsi Sai Krishna Malineni (OE20S302), Mohammed Safi Ur Rahman Khan (CS21M035) 

---
## General Instructions:
1. Install the required libraries using the following command :

```python 
pip install -r requirements.txt
```
2. The solution to the assignment is divided into two parts: `PART A` and `PART B`. You can find the juypter notebooks in the respective folders.

3. The jupyter notebooks can be run sequentially cell wise.

4. If you are running the jupyter notebooks on colab, the libraries from the `requirements.txt` file are preinstalled, with the `exception` of `wandb`. You can install wandb by using the following command :
```python
!pip install wandb
```
5. The dataset for this assignment can be found at : https://storage.googleapis.com/wandb_datasets/nature_12K.zip
---
## PART A : Training from Scratch
<br/> 1. Dataset for training and validation is prepared using the following functions :
  * Un-augmented Dataset: 
  ```python 
  get_data()
  ```
  * Augmented Dataset:
  ```python 
  get_augmented_data()
  ```
<br/> 2. Buidling a small CNN network with 5 convolution layers can be done by using the following method :
```python
build_cnn(conv_activation , dense_activation, num_filters, conv_filter_size, pool_filter_size, batch_norm, dense_layer, dropout)
```
where :
  *`conv_activation`     : dtype="List"    activation used for convolution layer
  * `dense_act`           : dtype="String"  acitvation used for densely connected layers
  * `num_filters`         : dtype="List"    number of activation filters for each layer
  * `conv_filter_size`   : dtype="List"    kernel sizes for convultion layers
  * `pool_filter_size`    : dtype="List"    kernel sizes for maxpooling layers
  * `batch_norm`          : dtype="Boolean" set to True, if you are using batch normalization
  * `dim_final`          : dtype="Integer" dimensionality of output space after 5 blocks of convultion, maxpooling blocks
  * `dropout`             : dtype="float or double" specify the dropout % for regularization (in decimals)

<br/> 3. The different hyper parameter configurations are specified in the following method
```python
sweeper(entity_name, project_name)
```
where
  * `entity_name` : Enter the wandb entity name
  * `project_name` : Enter the wandb project name

<br/> 4. The configuration for wandb sweeps is :
```python
hyperparameters={
    'batch_norm':        {'values':[True, False]},
    'num_filter':        {'values':[32, 64, 128, 256]},
    'filter_org':        {'values':[0.5, 1, 2]},
    'dropout':           {'values':[0.0, 0.5, 0.6, 0.4]},
    'data_augmentation': {'values':[True, False]},
    'num_epochs':        {'values':[10, 20, 30]},
    'batch_size':        {'values':[32, 64, 128]},
    'dense_layer':       {'values':[32, 64, 128, 512]},
    'learning_rate':     {'values':[0.001, 0.0001]},
    'kernel_size':       {'values': [3, 5, 7]}
    }   
sweep_config = {
      'method' : 'bayes','metric' :{'name': 'val_acc','goal': 'maximize'},
      'parameters': hyperparameters
    }
```
<br/> The following function will define the model, trains the model according to the hyperparameters given to it by wandb and logs the metrics to wandb.

```python
 train()
``` 
<br/>Use the following function to run the wandb sweeps 
```python 
sweeper(entity_name,project_name)
```
<br/> 5. Use the following function to generate test dataset to determine the test accuracy of model with best performance in terms of validation accuracy.
```python
get_test_data()
```
<br/> 6. Use the following function to determine the test accuracy of the best performing model, and log the metrics to wandb
```python
testing(entity_name,project_name)
```
<br/> 7. The trained model can be accessed at: https://drive.google.com/file/d/1aInmPFMV_rpJI_xPDP45h7XD4sGi1KaC/view?usp=sharing

<br/> 8. Guided Backpropagation with modified ReLU activation function is performed, and the patterns that excite the neurons are plotted and logged into wandb. 
<br/> The guided backpropagation is performed using this function:
```python
guided_backpropagation(neuron_number, conv_layer)
```
where
 * `neuron_number`: The neuron at which you want to see the pattern
 * `conv_layer`: The layer at which you want to visualize the pattern

---
## PART B : Using a Pre-Trained Model for Image Classification Task

<br/> 1. The images are resized to `(256,256,3)` irrespective of the size of the image from the dataset.

<br/> 2. The following function is used to build a model based on a pretrained model:
```python 
build_model(model_name, dense_activation, dense_layer, dropout, trainable_layers)
```
where 
 * `model_name`           : Enter the name of a pretrained model ("String")
 * `dense_activation`     : Enter the name of the activation function for dense layer ("String")
 * `dense_layer`          : Enter the number of units in the dense layer ("Integer")
 * `dropout`              : Enter the percent of drop out in decimals ("Double/float")
 * `trainable_layers`     : Enter the number of layers to be tuned ("Integer")

<br/> 3. The available models for pretraining are :
 * `ResNet50`
 * `InceptionV3`
 * `InceptionResNetV2`
 * `Xception`

<br/> 4. Training with wandb: Use the following function to perform wandb sweeping 
```python 
wandb_train()
```
<br/>  The hyperparameters for sweeping are :
```python
hyperparameters = {"model_name":{'values': ["InceptionV3", "ResNet50", "InceptionResNetV2", "Xception"]},
        "data_augmentation": {"values": [True, False]},
        "dense_layer": {"values": [64, 128, 256, 512]},
        "dropout": {"values": [0.0, 0.1, 0.2, 0.3]},        
        "trainable_layers": {"values": [0, 10, 15, 20]},
        "batch_size": {"values": [64, 128]},
        "num_epochs":{"values": [5, 10, 15]}
        }
sweep_config = {
  "method": "bayes",
  "metric": {"name":"val_acc","goal": "maximize"}, 
  "parameters": hyperparameters
   }
```
<br/> 5. The test data can be accessed using the following function:
* Un-augmented data: Use this function if the best model for your data set returns that data augmentation is not necessary
```python 
get_test_data()
```
* Augmented data: Use this function if the best model for your data set returns that data augmentation is necessary
```python
get_test_augmented_data()
```
<br/> 6. Run the following code, to test the best performing model on the test dataset
```python 
testing(entity_name, project_name)
```

## RESULTS
The results and the learnings from this assignment can be found here: https://wandb.ai/safi-vamsi-cs6910/Assignment%202/reports/Assignment-2--VmlldzoxNzY2Njky
