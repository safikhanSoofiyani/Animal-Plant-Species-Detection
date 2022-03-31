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
