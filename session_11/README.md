# ERA V1 Session 11 - Grad CAM 

## Contents
* [Introduction](#Introduction)
* [Code Links](#Code-Links)
* [Model](#Model)
* [Grad CAM Visualization](#Optimizer-and-Scheduler)

# Introduction
In this module, we train a Resnet18 model to perform image classification on CIFAR-10 dataset. We will also generate Grad-Cam saliency maps on some of the misclassified images to try and understand what features caused the wrong model prediction.

# Code Links
* The main repo can be found [here](https://github.com/jyanivaddi/dl_hub/tree/main). This repository contains all the model related stuff, helper methods, and visualization utilities. 
* The Resnet model used in the notebook is cloned from the following repository: [Resnet Model](https://github.com/kuangliu/pytorch-cifar)
* The GradCAM visualization is generated using the following repository: [GradCAM](https://github.com/jacobgil/pytorch-grad-cam)

# Model
The model used in this notebook is RESNET 18. Here is a summary of the model we used to perform classification. The table below is generated using the python package [torchinfo](https://pypi.org/project/torchinfo/) This provides a mode intuitive and detailed model summary than the torchsummary package. 

```
=====================================================================================================================================================================
Layer (type:depth-idx)                   Kernel Shape              Input Shape               Output Shape              Param #                   Mult-Adds
=====================================================================================================================================================================
ResNet                                   --                        [1, 3, 32, 32]            [1, 10]                   --                        --
├─Conv2d: 1-1                            [3, 3]                    [1, 3, 32, 32]            [1, 64, 32, 32]           1,728                     1,769,472
├─BatchNorm2d: 1-2                       --                        [1, 64, 32, 32]           [1, 64, 32, 32]           128                       128
├─Sequential: 1-3                        --                        [1, 64, 32, 32]           [1, 64, 32, 32]           --                        --
│    └─BasicBlock: 2-1                   --                        [1, 64, 32, 32]           [1, 64, 32, 32]           --                        --
│    │    └─Conv2d: 3-1                  [3, 3]                    [1, 64, 32, 32]           [1, 64, 32, 32]           36,864                    37,748,736
│    │    └─BatchNorm2d: 3-2             --                        [1, 64, 32, 32]           [1, 64, 32, 32]           128                       128
│    │    └─Conv2d: 3-3                  [3, 3]                    [1, 64, 32, 32]           [1, 64, 32, 32]           36,864                    37,748,736
│    │    └─BatchNorm2d: 3-4             --                        [1, 64, 32, 32]           [1, 64, 32, 32]           128                       128
│    │    └─Sequential: 3-5              --                        [1, 64, 32, 32]           [1, 64, 32, 32]           --                        --
│    └─BasicBlock: 2-2                   --                        [1, 64, 32, 32]           [1, 64, 32, 32]           --                        --
│    │    └─Conv2d: 3-6                  [3, 3]                    [1, 64, 32, 32]           [1, 64, 32, 32]           36,864                    37,748,736
│    │    └─BatchNorm2d: 3-7             --                        [1, 64, 32, 32]           [1, 64, 32, 32]           128                       128
│    │    └─Conv2d: 3-8                  [3, 3]                    [1, 64, 32, 32]           [1, 64, 32, 32]           36,864                    37,748,736
│    │    └─BatchNorm2d: 3-9             --                        [1, 64, 32, 32]           [1, 64, 32, 32]           128                       128
│    │    └─Sequential: 3-10             --                        [1, 64, 32, 32]           [1, 64, 32, 32]           --                        --
├─Sequential: 1-4                        --                        [1, 64, 32, 32]           [1, 128, 16, 16]          --                        --
│    └─BasicBlock: 2-3                   --                        [1, 64, 32, 32]           [1, 128, 16, 16]          --                        --
│    │    └─Conv2d: 3-11                 [3, 3]                    [1, 64, 32, 32]           [1, 128, 16, 16]          73,728                    18,874,368
│    │    └─BatchNorm2d: 3-12            --                        [1, 128, 16, 16]          [1, 128, 16, 16]          256                       256
│    │    └─Conv2d: 3-13                 [3, 3]                    [1, 128, 16, 16]          [1, 128, 16, 16]          147,456                   37,748,736
│    │    └─BatchNorm2d: 3-14            --                        [1, 128, 16, 16]          [1, 128, 16, 16]          256                       256
│    │    └─Sequential: 3-15             --                        [1, 64, 32, 32]           [1, 128, 16, 16]          8,448                     2,097,408
│    └─BasicBlock: 2-4                   --                        [1, 128, 16, 16]          [1, 128, 16, 16]          --                        --
│    │    └─Conv2d: 3-16                 [3, 3]                    [1, 128, 16, 16]          [1, 128, 16, 16]          147,456                   37,748,736
│    │    └─BatchNorm2d: 3-17            --                        [1, 128, 16, 16]          [1, 128, 16, 16]          256                       256
│    │    └─Conv2d: 3-18                 [3, 3]                    [1, 128, 16, 16]          [1, 128, 16, 16]          147,456                   37,748,736
│    │    └─BatchNorm2d: 3-19            --                        [1, 128, 16, 16]          [1, 128, 16, 16]          256                       256
│    │    └─Sequential: 3-20             --                        [1, 128, 16, 16]          [1, 128, 16, 16]          --                        --
├─Sequential: 1-5                        --                        [1, 128, 16, 16]          [1, 256, 8, 8]            --                        --
│    └─BasicBlock: 2-5                   --                        [1, 128, 16, 16]          [1, 256, 8, 8]            --                        --
│    │    └─Conv2d: 3-21                 [3, 3]                    [1, 128, 16, 16]          [1, 256, 8, 8]            294,912                   18,874,368
│    │    └─BatchNorm2d: 3-22            --                        [1, 256, 8, 8]            [1, 256, 8, 8]            512                       512
│    │    └─Conv2d: 3-23                 [3, 3]                    [1, 256, 8, 8]            [1, 256, 8, 8]            589,824                   37,748,736
│    │    └─BatchNorm2d: 3-24            --                        [1, 256, 8, 8]            [1, 256, 8, 8]            512                       512
│    │    └─Sequential: 3-25             --                        [1, 128, 16, 16]          [1, 256, 8, 8]            33,280                    2,097,664
│    └─BasicBlock: 2-6                   --                        [1, 256, 8, 8]            [1, 256, 8, 8]            --                        --
│    │    └─Conv2d: 3-26                 [3, 3]                    [1, 256, 8, 8]            [1, 256, 8, 8]            589,824                   37,748,736
│    │    └─BatchNorm2d: 3-27            --                        [1, 256, 8, 8]            [1, 256, 8, 8]            512                       512
│    │    └─Conv2d: 3-28                 [3, 3]                    [1, 256, 8, 8]            [1, 256, 8, 8]            589,824                   37,748,736
│    │    └─BatchNorm2d: 3-29            --                        [1, 256, 8, 8]            [1, 256, 8, 8]            512                       512
│    │    └─Sequential: 3-30             --                        [1, 256, 8, 8]            [1, 256, 8, 8]            --                        --
├─Sequential: 1-6                        --                        [1, 256, 8, 8]            [1, 512, 4, 4]            --                        --
│    └─BasicBlock: 2-7                   --                        [1, 256, 8, 8]            [1, 512, 4, 4]            --                        --
│    │    └─Conv2d: 3-31                 [3, 3]                    [1, 256, 8, 8]            [1, 512, 4, 4]            1,179,648                 18,874,368
│    │    └─BatchNorm2d: 3-32            --                        [1, 512, 4, 4]            [1, 512, 4, 4]            1,024                     1,024
│    │    └─Conv2d: 3-33                 [3, 3]                    [1, 512, 4, 4]            [1, 512, 4, 4]            2,359,296                 37,748,736
│    │    └─BatchNorm2d: 3-34            --                        [1, 512, 4, 4]            [1, 512, 4, 4]            1,024                     1,024
│    │    └─Sequential: 3-35             --                        [1, 256, 8, 8]            [1, 512, 4, 4]            132,096                   2,098,176
│    └─BasicBlock: 2-8                   --                        [1, 512, 4, 4]            [1, 512, 4, 4]            --                        --
│    │    └─Conv2d: 3-36                 [3, 3]                    [1, 512, 4, 4]            [1, 512, 4, 4]            2,359,296                 37,748,736
│    │    └─BatchNorm2d: 3-37            --                        [1, 512, 4, 4]            [1, 512, 4, 4]            1,024                     1,024
│    │    └─Conv2d: 3-38                 [3, 3]                    [1, 512, 4, 4]            [1, 512, 4, 4]            2,359,296                 37,748,736
│    │    └─BatchNorm2d: 3-39            --                        [1, 512, 4, 4]            [1, 512, 4, 4]            1,024                     1,024
│    │    └─Sequential: 3-40             --                        [1, 512, 4, 4]            [1, 512, 4, 4]            --                        --
├─Linear: 1-7                            --                        [1, 512]                  [1, 10]                   5,130                     5,130
=====================================================================================================================================================================
Total params: 11,173,962
Trainable params: 11,173,962
Non-trainable params: 0
Total mult-adds (M): 555.43
=====================================================================================================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 9.83
Params size (MB): 44.70
Estimated Total Size (MB): 54.54
=====================================================================================================================================================================
```

# Optimizer and Scheduler
For this model, we use an [Adam optimizer](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) for implementing back propagation. Adam optimizer uses a per-parameter learning rate unlike stochastic gradient that uses a single learning rate for all the parameters. We use a [Cross Entropy](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) as loss function. 

In order the schedule the learning rate, and to achieve faster convergence, we use [OnecycleLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html) scheduler. First proposd by [Leslie Smith](https://arxiv.org/abs/1708.07120), this method works by first increasing the LR to a high value in the initial few epochs, followed by a gradually decreasing trend. The high learning rate helps the model to reach closer to the global minima and the subsequent reduction in the LR stabilizes the optimizer and gives a more accurate minima.

For this model, the OneCycleLR is defined as follows:
```
scheduler = OneCycleLR(
        optimizer,
        max_lr = 4.65E-02,
        steps_per_epoch=98,
        epochs = 24,
        pct_start = 0.208,
        div_factor=2000,
        three_phase=False,
        final_div_factor= 100,
        anneal_strategy='linear',
        verbose=False
        )
```
In the code above, ```steps_per_epoch``` indicates the length of train loader. Since we set the batch size to 512, there are 98 steps per epoch. ```pct_start``` is the percentage of epochs at which peak LR is applied after which the learning rate is "annealed". The starting learning rate is achieved by dividing the maximum learning rate by ```div_factor```. Afer several iterations, we set this parameter to be 2000 to get the 90% accuracy. We disable three phases learning in our model, so the ```final_div_factor``` is not applicable.

## Detecting max LR value
The maximum value of learning rate to be used in the onecycleLr policy is calculated using a python package called [LRFinder](https://github.com/davidtvs/pytorch-lr-finder) This module identifies the optimum value of learning rate by iterating over a range of learning rates and determines the value at which the gradients are near maximum. In this model, a peak LR value of 4.65 was calculated as shown below:
![Max_LR_calculation](doc/max_lr_annotated.png)

Figure below shows how the learning rate varied from a starting value of 2.32E-05 over 24 epochs of training. There are 98 batches in the train loader and for 24 training epochs, a total of 2352 steps were taken by the scheduler 

![LR_scheduler](doc/lr_annotated.png)


# Results

**The model was run for 24 epochs and has achieved a maximum validation accuracy of 90.31**. The table below shows the training log over 24 epochs. This table is generated using a python package [PrettyTable](https://pypi.org/project/prettytable/)
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
The plots below show accuracy and loss computation over 24 epochs of training. 

![metrics](doc/loss_annotated.png)
![metrics2](doc/accuracy_annotated.png)


# Learnings
Some takeaways from this exercise:
* Very fast convergence and high validation accuracy can be achieved by using OneCycleLR policy.
* In onecycleLr, the learning rate scheduler has to be stepped at each batch instead of each epoch. The parameter ```div_factor``` plays an important role in determining the final accuracy.
* Adding regularization such as dropout, and image agumentations helped the model not to overfit training set. 
