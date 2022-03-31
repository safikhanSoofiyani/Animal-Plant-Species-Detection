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
