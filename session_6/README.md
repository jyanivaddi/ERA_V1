# ERA V1 Session 6 - Detecting handwritten digits under 20K parameters with an accuracy of 99.4%

## Contents

* [Introduction](#Introduction)
* [Model] (#Model)
* [Summary of Experiments](#Experiments)
* [Results & Takeaways] (#Results)
* [Code Stucture]
   * [S6.ipynb](#S6ipynb`)
   * [s6_model.py](#modelpy`)
   * [s6_utils.py](#utilspy`)


## Introduction
<p>
In this session's assignment, I built a deep learning model to predict handwritten digits from the MNIST dataset. The model contains 14372 parameters and achieved the required test accuracy of 99.4% within 13 training epochs. The maximum test accuracy during the 20 epochs of training was 99.6%.
</p>


## Model Architecture


## Summary of experiments

<p> 
I tested several model architectures and model configurations and below  is a summary of the model design and the performance achieved. I realized that none of these networks worked well since the RF was too low and the network too shallow. Having 32 output channels in the network is making the model significantly heavy and making me to exhaust the 20k parameter limit. When I tried to limit the max channels to 16, I was able to add several more layers and get to much higher RF without breaching the parameter limit.
</p>

### Architecture 1: Max. accuracy: 98.1%
Num parameters: 5828   
Max. RF: 10
![experiment 1](doc/experiment_1.png)


### Architecture 2: Max. accuracy: 98.5%
Num parameters: 14468  
 Max. RF: 8
![experiment 2](doc/experiment_2.png)


### Architecture 3: Accuracy 98.75%
Num parameters: 19764   
Max. RF: 10
![experiment 3](doc/experiment_3.png)


## model.py
<p>This script contains the definition of the deep learning model that we are trying to build. The model is defined as a class and contains the definitions of all the convolution and fully connected layers. 

The model is summarized as:
</p>


----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
----------------------------------------------------------------

            Conv2d-1           [-1, 32, 26, 26]             320
            Conv2d-2           [-1, 64, 24, 24]          18,496
            Conv2d-3          [-1, 128, 10, 10]          73,856
            Conv2d-4            [-1, 256, 8, 8]         295,168
            Linear-5                   [-1, 50]         204,850
            Linear-6                   [-1, 10]             510

----------------------------------------------------------------

Total params: 593,200 <br>
Trainable params: 593,200 <br>
Non-trainable params: 0 <br>

----------------------------------------------------------------

## utils.py

<p> This file contains all the support functions needed to train, test and analyze the model. There are plotting utilities to preview the dataset, and to monitor the accuracy and loss for both train and test for each epoch.</p>

### Model training
<p> The train method performs the following steps for each epoch:

1. Set the model to train mode
2. for each batch in the epoch:

   1. Send the data and labels to the device (GPU or CPU)
   2. Set the gradient to zero
   3. Predict the outputs from the model for the batch
   4. Compute loss
   5. Perform back propagation to calculate gradients
   6. Update the weights for each layer
3. At the end of the epoch, track the accuracy and loss for this epoch for monitoring the training process
</p>

### Model testing
<p>  The test method performs the following steps for each epoch:

1. Set the model to evaluation mode (i.e., disable gradient computation.)
2. for each batch in the epoch:

   1. Send the data and labels to the device (GPU or CPU)
   2. Compute outputs from the model for the current batch
   3. Update the test losses and accuracy

</p>

