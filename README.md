# CS6910-Assignment-2
Assignment 2 submission for the course CS6910 Fundamentals of Deep Learning. 

Team Members : Vamsi Sai Krishna Malineni (OE20S302), Mohammed Safi Ur Rahman Khan (CS21M035) 

---
## General Instructions:
1. Install the required libraries using the following command :

```python 
pip install -r requirements.txt
```
2. The solution to the assignment is divided into two parts: PART A and PART B. You can find the juypter notebooks in the respective folders.
3. The jupyter notebooks can be run sequentially cell wise.
4. If you are running the jupyter notebooks on colab, the libraries from the requirements.txt file are preinstalled, with the exception of wandb. You can install wandb by using the following command :
```python
!pip install wandb
```
## PART A : Training from Scratch
1. Dataset for training and validation is prepared using the following functions :
  * Un-augmented Dataset: 
  ```python 
  get_data()
  ```
  * Augmented Dataset:
  ```python 
  get_augmented_data()
  ```
2. Buidling a small CNN network with 5 convolution layers can be done by using the following method :
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
3. The different hyper parameter configurations are specified in the following method
```python
sweeper(entity_name, project_name)
```
where
  *`entity_name` : Enter the wandb entity name
  *`project_name` : Enter teh wandb project name
5. 
