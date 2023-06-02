# ERA V1 Session 5 - Introduction to Pytorch

## Contents

* [Introduction](#Introduction)
 
* [S5.ipynb](#S5ipynb`)

* [model.py](#modelpy)

* [utils.py](#utilspy)


## Introduction
<p>
In this repository, we build a deep learning model to predict handwritten digits
using the MNIST dataset. 
</p>


## S5.ipynb

<p> This notebook contains the script to run the model end to end. Since the notebook refers to external python script 
files, we need to make sure that these files are available for colab to locate. In order to do that, we need to upload the 
python files to Google Drive and mount the path to google drive to the local file system inside colab environment.
This is done by adding the following code to the top of the block:
</p>

```
import sys
from google.colab import drive
drive.mount('/content/drive')
sys.path.insert(0,'/content/drive/MyDrive/ERA_V1/session_5')
```

<p>
Once the directory is mounted, then we can run the notebook one cell at a time. 
</p>

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

