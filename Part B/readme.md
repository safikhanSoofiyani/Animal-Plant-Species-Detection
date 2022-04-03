---
# PART B : Fine tuning a Pre-trained model
## Running the code
As mentioned earlier, there are two files in the folder of Part B. One is a jupyter notebook and the other is the python code file.<br>
The jupyter notebook has the outputs still intact so that can be used for reference. <br>
The python file has all the functions and the code used in the jupyter file (along with some additional code that can be used to run from the command line)<br>
<br>
### Running the python file from the terminal
The python file can be run from the terminal by passing the various command line arguments. Please make sure that the unzipped folder of the dataset is as it is present in the same directory as this python file <br>
There are two modes of running this file <br> <br>
**1. Running the hyperparameter sweeps using wandb**<br>
```sh
python assgn2B.py --sweep yes
```
The code will now run in the sweep mode and will enable wandb integration and log all the data in wandb. Make sure you have wandb installed if you want to run in this mode. Also, change the entity name and project name in the code before running in this mode<br> <br>
**2. Running in normal mode**<br>
```sh
python assgn2B.py --sweep no --model xxx --dropout xxx --dataAugment xxx --numEpochs xxx --batchSize xxx --denseLayer xxx --learningRate xxx --trainLayers xxx --denseAct xxx
```
Replace `xxx` in above with the appropriate parameter you want to train the model with<br>
For example: 

```sh
python assgn2B.py --sweep no --model Xception --dropout 0.3 --dataAugment False --numEpochs 10 --batchSize 32 --denseLayer 128 --learningRate 0.001 --trainLayers 10 --denseAct relu
```
**Description of various command line arguments**<br>
1. `--sweep` : Do you want to sweep or not: Enter 'yes' or 'no'. If this is 'yes' then below arguments are not required. Enter below arguments only if this is 'no'<br>
2. `--model` : Pretrained model to use: string value <br>
3. `--dropout` : Dropout: float value
4. `--dataAugment` : Data Augmentation: True or False
5. `--numEpochs` : Number of Epochs: integer value
6. `--batchSize` : Batch Size: integer value
7. `--denseLayer` : Dense Layer size: integer value
8. `--learningRate` : Learning Rate: float value
9. `--trainLayers` : Number of trainable layers: integer value
10. `--denseAct` : Dense Layer Activation function: string value

### Running the jupyter notebook

This can be run in a sequential manner. i.e., one cell at a time. This notebook also has the code for plotting the various images required for the assignment.


## General Functions Description
### 1. Getting Train Dataset 
<br/>  Dataset for training and validation is prepared using the following functions :
  * Un-augmented Dataset: 
  ```python 
  get_data()
  ```
  * Augmented Dataset:
  ```python 
  get_augmented_data()
  ```
### 2. Building Model
<br/> The following function is used to build a model based on a pretrained model:
```python 
build_model(model_name, dense_activation, dense_layer, dropout, trainable_layers)
```
where 
 * `model_name`           : Enter the name of a pretrained model ("String")
 * `dense_activation`     : Enter the name of the activation function for dense layer ("String")
 * `dense_layer`          : Enter the number of units in the dense layer ("Integer")
 * `dropout`              : Enter the percent of drop out in decimals ("Double/float")
 * `trainable_layers`     : Enter the number of layers to be tuned ("Integer")
<br>
The available models for pretraining are :
 * `ResNet50`
 * `InceptionV3`
 * `InceptionResNetV2`
 * `Xception`


### 3. Hyperparameter Sweeps
```python
sweeper(entity_name, project_name)
```
where
  * `entity_name` : Enter the wandb entity name
  * `project_name` : Enter the wandb project name

<br/>  The various hyperparameters used are :
```python
hyperparameters = {
        "model_name":{'values': ["InceptionV3", "ResNet50", "InceptionResNetV2", "Xception"]},
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
### 4. Training the model
<br/> The following function will define the model, train the model according to the hyperparameters given to it by wandb and logs the metrics to wandb.

```python
 wandb_train()
``` 
<br/>Use the following function to run the wandb sweeps 
```python 
sweeper(entity_name,project_name)
```
### 5. Getting Test Dataset
<br/> The test data can be accessed using the following function:
* Un-augmented data: Use this function if the best model for your data set returns that data augmentation is not necessary
```python 
get_test_data()
```
* Augmented data: Use this function if the best model for your data set returns that data augmentation is necessary
```python
get_test_augmented_data()
```

### 6. Testing the best model
<br/> Use the following function to determine the test accuracy of the best performing model, and log the metrics to wandb
```python
testing(entity_name,project_name)
```
