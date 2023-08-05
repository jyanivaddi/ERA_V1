# ERA V1 Session 12 - Pytorch Lightning 

# Introduction
In this module, we explore pytorch lightning for building end to end deep learning applications. [Lightning](https://www.pytorchlightning.ai/index.html) is a wrapper built on top of pytorch to improve the speed and convenience of building deep learning applications. Lightning eliminates most of the boiler plate code that needs to be otherwise included in pytorch and help us focus on developing great applications! 

In this notebook, we retrain the custom resnet model that we wrote in [Session 10](https://github.com/jyanivaddi/ERA_V1/tree/master/session_10) using torch lightning. Using lightning, we achieved a validation accuracy of **93%** on the CIFAR10 dataset.   

# Code Links
* The main repo can be found [here](https://github.com/jyanivaddi/dl_hub/tree/main).
  * The custom resnet model that we built in session 10 can be found [here](https://github.com/jyanivaddi/dl_hub/blob/main/models/custom_resnet.py) and the corresponding Lightning model can be found [here](https://github.com/jyanivaddi/dl_hub/blob/main/models/pl_custom_resnet.py)
  * The Datamodule for pytorch lightning defition is defined [here](https://github.com/jyanivaddi/dl_hub/blob/main/dataloaders/pl_custom_cifar10_datamodule.py). Using this datamodule, we read the train, test, and validation datasets on CIFAR10 using the image augmentations defined with [albumentations](https://albumentations.ai/) library. 
  * The main code that instantiates the model, runs the training, inference, and test is defined [here](https://github.com/jyanivaddi/dl_hub/blob/main/PL_main.py) 
  * Several helper functions to compute misclassified images,GradCAM images, etc is defined in the [utils](https://github.com/jyanivaddi/dl_hub/tree/main/utils) module.


# Model
The model used in this notebook same as the one we used in session 10 and was explained [here](https://github.com/jyanivaddi/ERA_V1/tree/master/session_10#Model) In this notebook, we use the same model but write a pytorch lightning wrapper that contains the following methods:

# Optimizer
The details of the optimizer and scheduler can be found from the session 10 note book [here](https://github.com/jyanivaddi/ERA_V1/blob/master/session_10/README.md#Optimizer-and-Scheduler)

# Results
The model achieved a **93%** validation accuracy on the CIFAR10 dataset in 24 epochs. The plots below show the validation and train losses over the training duration
![metrics]


