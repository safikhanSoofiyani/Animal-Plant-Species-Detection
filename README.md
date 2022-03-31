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
1. Buidling a small CNN network with 5 convolution layers can be done by using the following method :
```python
build_cnn(conv_activation , dense_activation, num_filters, conv_filter_size, pool_filter_size, batch_norm, dense_layer, dropout)
```
