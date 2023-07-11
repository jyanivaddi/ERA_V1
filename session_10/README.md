# ERA V1 Session 10 - Custom Resnet for CIFAR10 image classification

## Contents
* [Introduction](#Introduction)
* [Dataset](#Dataset)
* [Model](#Model)
* [Convolutions](#Convolutions)
* [Results](#Results)
* [Learnings](#Learnings)

# Introduction
In this module, we build a custom resnet model to perform image classification on CIFAR-10 dataset. With this model, we show that by using OnecycleLR scheduling policy, we can achieve a **90.3%** test accuracy within __24__ training epochs for this dataset. We take inspiration from [this](https://myrtle.ai/learn/how-to-train-your-resnet-8-bag-of-tricks/) blog post to build our model. 

# Dataset
The dataset we use in this notebook is called **[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)**. This dataset consists of 60000 RGB images of size 32 x 32. There are a total of 10 classes and 6000 images per class. There are 50000 training images and 10000 test images. we set the batch size to 128 during training so there are 391 batches of images per epoch. 

The images below show some representative images from the dataset
![no_labels](doc/dataset_images_no_labels.png)

The following images show few more samples, along with their labels
![labels](doc/dataset_images.png)

## Image Augmentations using Albumentations library
Image augmentations are used in order to increase the dataset size and as a way of inproving model regularization. By transforming the dataset arbitrarily we ensure that the network does not memorize the train dataset. In our model, we use  [Albumentations](https://albumentations.ai/) library for implementing various image augmentations. The library contains several augmentation methods and it seamlessly integrates with Pytorch. 

The three different image transformation methods we use in this model are:

* [HorizontalFlip](https://albumentations.ai/docs/api_reference/full_reference/#albumentations.augmentations.geometric.transforms.HorizontalFlip): This method randomly flips an image horizontally 
* [RandomResizedCrop](https://albumentations.ai/docs/api_reference/full_reference/#albumentations.augmentations.crops.transforms.RandomResizedCrop): This method shifts, scales, or rotates the image 
* [Cutout](https://albumentations.ai/docs/api_reference/full_reference/#albumentations.augmentations.dropout.coarse_dropout.CoarseDropout): This method randomly cuts out a square of size 8 x 8 in the image. 

The below image demonstrates how the image transforms look on a sample image
![transforms](doc/augmentations.png)


# Model
Our model contains four convolution blocks, each of which has three convolution layers. The output from the fourth block is passed through an adaptive average pooling layer, followed by a 1x1 convolution to reduce the number of output channels to 10. Finally log softmax function is used to calculate the class probabilities.

```B1-> B2 -> B3 -> B4 -> AAP -> C5 -> O```  is the model structure 
where ```B``` represents a convolution block of three layers, ```C``` represents a 1 x 1 convolution block, ```AAP``` represents an Adaptive Average Pooling block, and ```O``` indicates the log softmax output. The model contains a total of 171840 parameters. The model achieved a max. receptive field of 77 for the cifar-10 input data of size 3 x 32 x 32. 

Each block ```B``` contains the following layers:

**Layer 1**: 3D convolution with kernel size of 3 

**Layer 2**: Depthwise separable convolution

**Layer 3**: Dilated convolution



The receptive field computations for this model are shown below:
![model_arch](doc/RF_calculations.png)

<!--## Dilated Convolution
In dilated convolution, the kernel size is effectively increased by using alternate 
![dilated](doc/dilation.gif)
![depthwise](doc/Depthwise-separable-convolution-block.png)
## Regularization
A uniform drop out value of 5% was used in all the convolution blocks except the final 1 x 1 convolution to prevent overfitting to the train set. 

Feature normalization is performed using batch normalization at each convolution layer

Additionally, data transformations mentioned in the Dataset section were implemtented to augment the dataset. 
-->

## Model Summary
Here is a summary of the model we used to perform classification. 

```
=====================================================================================================================================================================
Layer (type:depth-idx)                   Kernel Shape              Input Shape               Output Shape              Param #                   Mult-Adds
=====================================================================================================================================================================
CustomResnet                             --                        [1, 3, 32, 32]            [1, 10]                   --                        --
├─Sequential: 1-1                        --                        [1, 3, 32, 32]            [1, 64, 32, 32]           --                        --
│    └─Conv2d: 2-1                       [3, 3]                    [1, 3, 32, 32]            [1, 64, 32, 32]           1,728                     1,769,472
│    └─BatchNorm2d: 2-2                  --                        [1, 64, 32, 32]           [1, 64, 32, 32]           128                       128
│    └─ReLU: 2-3                         --                        [1, 64, 32, 32]           [1, 64, 32, 32]           --                        --
│    └─Dropout: 2-4                      --                        [1, 64, 32, 32]           [1, 64, 32, 32]           --                        --
├─Layer: 1-2                             --                        --                        --                        --                        --
│    └─Sequential: 2-5                   --                        [1, 64, 32, 32]           [1, 128, 16, 16]          --                        --
│    │    └─Conv2d: 3-1                  [3, 3]                    [1, 64, 32, 32]           [1, 128, 32, 32]          73,728                    75,497,472
│    │    └─MaxPool2d: 3-2               2                         [1, 128, 32, 32]          [1, 128, 16, 16]          --                        --
│    │    └─BatchNorm2d: 3-3             --                        [1, 128, 16, 16]          [1, 128, 16, 16]          256                       256
│    │    └─ReLU: 3-4                    --                        [1, 128, 16, 16]          [1, 128, 16, 16]          --                        --
│    │    └─Dropout: 3-5                 --                        [1, 128, 16, 16]          [1, 128, 16, 16]          --                        --
│    └─ResidualBlock: 2-6                --                        --                        --                        --                        --
│    │    └─Sequential: 3-6              --                        [1, 128, 16, 16]          [1, 128, 16, 16]          147,712                   37,748,992
│    │    └─Sequential: 3-7              --                        [1, 128, 16, 16]          [1, 128, 16, 16]          147,712                   37,748,992
├─Sequential: 1-3                        --                        [1, 128, 16, 16]          [1, 256, 8, 8]            --                        --
│    └─Conv2d: 2-7                       [3, 3]                    [1, 128, 16, 16]          [1, 256, 16, 16]          294,912                   75,497,472
│    └─MaxPool2d: 2-8                    2                         [1, 256, 16, 16]          [1, 256, 8, 8]            --                        --
│    └─BatchNorm2d: 2-9                  --                        [1, 256, 8, 8]            [1, 256, 8, 8]            512                       512
│    └─ReLU: 2-10                        --                        [1, 256, 8, 8]            [1, 256, 8, 8]            --                        --
│    └─Dropout: 2-11                     --                        [1, 256, 8, 8]            [1, 256, 8, 8]            --                        --
├─Layer: 1-4                             --                        --                        --                        --                        --
│    └─Sequential: 2-12                  --                        [1, 256, 8, 8]            [1, 512, 4, 4]            --                        --
│    │    └─Conv2d: 3-8                  [3, 3]                    [1, 256, 8, 8]            [1, 512, 8, 8]            1,179,648                 75,497,472
│    │    └─MaxPool2d: 3-9               2                         [1, 512, 8, 8]            [1, 512, 4, 4]            --                        --
│    │    └─BatchNorm2d: 3-10            --                        [1, 512, 4, 4]            [1, 512, 4, 4]            1,024                     1,024
│    │    └─ReLU: 3-11                   --                        [1, 512, 4, 4]            [1, 512, 4, 4]            --                        --
│    │    └─Dropout: 3-12                --                        [1, 512, 4, 4]            [1, 512, 4, 4]            --                        --
│    └─ResidualBlock: 2-13               --                        --                        --                        --                        --
│    │    └─Sequential: 3-13             --                        [1, 512, 4, 4]            [1, 512, 4, 4]            2,360,320                 37,749,760
│    │    └─Sequential: 3-14             --                        [1, 512, 4, 4]            [1, 512, 4, 4]            2,360,320                 37,749,760
├─MaxPool2d: 1-5                         4                         [1, 512, 4, 4]            [1, 512, 1, 1]            --                        --
├─Linear: 1-6                            --                        [1]                       [10]                      5,120                     51,200
=====================================================================================================================================================================
Total params: 6,573,120
Trainable params: 6,573,120
Non-trainable params: 0
Total mult-adds (M): 379.31
=====================================================================================================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 4.65
Params size (MB): 26.29
Estimated Total Size (MB): 30.96
=====================================================================================================================================================================
```

## Optimizer
For this model, we used Stochastic Gradient Descent optimizer with negative log likelihood loss function at an initial learning rate of 0.1. 

A learning rate scheduler called [ReduceLROnPleateau](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html) was used to automatically adjust the learning rate based on the model performance. The scheduler was changes the learning rate to 10% of the current rate when the test loss hasn't reduced by atleast 0.1 over the previous 4 epochs.

Figure below shows how the learning rate varied from a starting value of 0.01 over 30 epochs of training.

![LR_scheduler](doc/lr_scheduler.png)


# Results

**The model was run for 30 epochs and has achieved a maximum validation accuracy of 85.87**. The table below shows the training log over 30 epochs.
```
+-------+---------------------+-----------------------+----------------+--------------+
| Epoch |      Train loss     |        Val loss       | Train Accuracy | Val Accuracy |
+-------+---------------------+-----------------------+----------------+--------------+
|   1   |  1.5742168827932708 | 0.0029146114587783813 |     43.854     |    50.96     |
|   2   |  1.1938890224816847 | 0.0022604991614818574 |     58.224     |    61.34     |
|   3   |  0.9751202147834155 | 0.0020353579759597776 |     66.34      |    66.83     |
|   4   |  0.8359070256048319 | 0.0017099724173545838 |     70.842     |    70.24     |
|   5   |  0.8299364003599906 | 0.0017883456230163575 |     71.164     |    70.09     |
|   6   |  0.8120121566616759 |  0.002074887454509735 |     71.724     |    65.29     |
|   7   |  0.7742981107867494 | 0.0020398482739925386 |     73.024     |    66.85     |
|   8   |  0.7619756192577128 | 0.0016085849463939666 |     73.48      |    73.13     |
|   9   |  0.755055057150977  | 0.0015251864731311798 |     73.736     |     74.0     |
|   10  |  0.7169662014562257 | 0.0015407093584537505 |     74.928     |    73.27     |
|   11  |  0.7098500619129259 | 0.0014445123374462127 |     75.256     |    75.18     |
|   12  |  0.6992844507402304 | 0.0013512782871723174 |     75.468     |    76.43     |
|   13  |  0.6766707021362928 | 0.0013968487322330474 |     76.47      |    76.81     |
|   14  |  0.6574481312109499 | 0.0011788516283035278 |     77.052     |    79.38     |
|   15  |  0.6400070853379308 |  0.001109265074133873 |      77.7      |    80.71     |
|   16  |  0.6265086248821142 | 0.0015220060527324677 |     78.276     |     75.5     |
|   17  |  0.5959175186497825 | 0.0012828315496444702 |     79.236     |    78.13     |
|   18  |  0.5715064540201303 | 0.0010273142158985138 |     80.018     |    82.59     |
|   19  |  0.545961357196983  | 0.0009553893476724625 |     80.922     |    83.59     |
|   20  |  0.502489734669121  | 0.0009888218343257905 |     82.52      |    83.01     |
|   21  |  0.4564004957067723 | 0.0009257473468780518 |     84.132     |    84.37     |
|   22  |  0.4085405447653362 | 0.0007858679652214051 |     85.744     |     86.4     |
|   23  | 0.34982549840090227 |  0.000621778267621994 |     87.852     |    89.33     |
|   24  | 0.28568454907864943 | 0.0005673721894621849 |      90.3      |    90.31     |
+-------+---------------------+-----------------------+----------------+--------------+
```
The plots below show accuracy and loss computation over 19 epochs of training using all the three normalization methods described above. In our model, we used the group norm with 2 groups. 

![metrics](doc/model_training.png)


Below figures shows some examples of incorrect predictions the model made in all the three normalization configurations. In each image, the first class indicates the ground truth and the second indicates the model prediction. 

![bn_results](doc/misclassification.png) 

# Learnings
Some takeaways from this exercise:
* Dilated convolutions can enhance the receptive field without compromising on the spatial resolution. These can be used in place of max pooling or strided convolutions.
* Depthwise separable convolutions are a great tool to reduce the number of parameters while maitaining the capacity. As can be seen in the model, it is far easier to add 64 channel convolutions by using depthwise separable convolution while still keeping the total number of parameters under 200000. These are especially useful when working with limited infrastructure necessitates lighter models.
* Albumentations is a great library to incorporate augmentations in deep learning models. 
