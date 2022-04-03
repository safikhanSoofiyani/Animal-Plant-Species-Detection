# PART A : Training from Scratch
**NOTE:** Since the ipynb file is large, it may not render properly on GitHub. So, kindly download the file to view in jupyter or use [nbviewer.org](nbviewer.org) to view this notebook
## Running the code
As mentioned earlier, there are two files in the folder of Part A. One is a jupyter notebook and the other is the python code file.<br>
The jupyter notebook has the outputs still intact so that can be used for reference. <br>
The python file has all the functions and the code used in the jupyter file (along with some additional code that can be used to run from the command line)<br>
<br>
### Running the python file from the terminal
The python file can be run from the terminal by passing the various command line arguments. Please make sure that the unzipped folder of the dataset is as it is present in the same directory as this python file<br>
There are two modes of running this file <br> <br>
**1. Running the hyperparameter sweeps using wandb**<br>
```sh
python assgn2.py --sweep yes
```
The code will now run in the sweep mode and will enable wandb integration and log all the data in wand. Make sure you have wandb installed if you want to run in this mode. Also, change the entity name and project name in the code before running in this mode<br> <br>
**2. Running in normal mode**<br>
```sh
python assgn2.py --sweep no --batchNorm xxx --numFilters xxx --filterOrg xxx --dropout xxx --dataAugment xxx --numEpochs xxx --batchSize xxx --denseLayer xxx --learningRate xxx --kernelSize xxx --denseAct xxx --convAct xxx
```
Replace `xxx` in above with the appropriate parameter you want to train the model with<br>
For example: 

```sh
python assgn2.py --sweep no --batchNorm True --numFilters 32 --filterOrg 2 --dropout 0.4 --dataAugment False --numEpochs 10 --batchSize 128 --denseLayer 512 --learningRate 0.0001 --kernelSize 3 --denseAct relu --convAct relu
```
**Description of various command line arguments**<br>
1. `--sweep` : Do you want to sweep or not: Enter 'yes' or 'no'. If this is 'yes' then below arguments are not required. Enter below arguments only if this is 'no'<br>
2. `--batchNorm` : Batch Normalization: True or False <br>
3. `--numFilters` : Number of Filters: integer value <br>
4. `--filterOrg` : Filter Organization: float value
5. `--dropout` : Dropout: float value
6. `--dataAugment` : Data Augmentation: True or False
7. `--numEpochs` : Number of Epochs: integer value
8. `--batchSize` : Batch Size: integer value
9. `--denseLayer` : Dense Layer size: integer value
10. `--learningRate` : Learning Rate: float value
11. `--kernelSize` : Kernel Size: integer value
12. `--denseAct` : Dense Layer Activation function: string value
13. `--convAct` : Conv Layer Activation function: string value


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
<br/>  Buidling a small CNN network with 5 convolution layers can be done by using the following method :
```python
build_cnn(conv_activation , dense_activation, num_filters, conv_filter_size, pool_filter_size, batch_norm, dense_layer, dropout)
```
where :
  * `conv_activation`     : dtype="List"    activation used for convolution layer
  * `dense_activation`    : dtype="String"  acitvation used for densely connected layers
  * `num_filters`         : dtype="List"    number of activation filters for each layer
  * `conv_filter_size`    : dtype="List"    kernel sizes for convultion layers
  * `pool_filter_size`    : dtype="List"    kernel sizes for maxpooling layers
  * `batch_norm`          : dtype="Boolean" set to True, if you are using batch normalization
  * `dim_final`           : dtype="Integer" dimensionality of output space after 5 blocks of convultion, maxpooling blocks
  * `dropout`             : dtype="float or double" specify the dropout % for regularization (in decimals)

### 3. Hyperparameter Sweeps
<br/> The hyperparamter sweeps can be run using the following method
```python
sweeper(entity_name, project_name)
```
where
  * `entity_name` : Enter the wandb entity name
  * `project_name` : Enter the wandb project name

<br/>  The various hyperparameters used are :
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

### 4. Training the model
<br/> The following function will define the model, train the model according to the hyperparameters given to it by wandb and logs the metrics to wandb.

```python
 train()
``` 
<br/>Use the following function to run the wandb sweeps 
```python 
sweeper(entity_name,project_name)
```
### 5. Getting Test Dataset
<br/>  Use the following function to generate test dataset to determine the test accuracy of model with best performance in terms of validation accuracy.
```python
get_test_data()
```

### 6. Testing the best model
<br/> Use the following function to determine the test accuracy of the best performing model, and log the metrics to wandb
```python
testing(entity_name,project_name)
```
<br/> - The best trained model can be accessed at: https://drive.google.com/file/d/1aInmPFMV_rpJI_xPDP45h7XD4sGi1KaC/view?usp=sharing

<br><br> The ipynb file contains all the necessary plots and the code to get them. The plots include<br>
1. Plotting images with their true and predicted labels<br>
2. Plotting the filters and the feature maps for a random image<br>
3. Visualizing 10 random neurons using guided backpropagation
