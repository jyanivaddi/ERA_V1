import numpy as np
from typing import Any
from torchvision import datasets, transforms
import albumentations as A
import albumentations.augmentations as AA
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader



class OxfordPetDataset(Dataset):

    def __init__(self, dataset, transforms=None):
        self.transforms = transforms
        self.dataset = dataset

    
    def __len__(self) -> int:
        return len(self.dataset)

    
    def __getitem__(self, index) -> Any:
        image, label = self.dataset[index]
        image = np.array(image)
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]
        return (image, label)


def load_oxford_pet_data(train_transforms, test_transforms, batch_size, **kwargs):
    train_data = datasets.CIFAR10('../data', train=True, download=True)
    test_data = datasets.CIFAR10('../data', train=False, download=True)
    class_names = {"0": "airplane",
               "1": "automobile",
               "2": "bird",
               "3": "cat",
               "4": "deer",
               "5": "dog",
               "6": "frog",
               "7": "horse",
               "8": "ship",
               "9": "truck"}
    train_loader = DataLoader(OxfordPetDataset(train_data, transforms = None),
                              batch_size=batch_size, 
                              shuffle=True, 
                              **kwargs)
    test_loader = DataLoader(OxfordPetDataset(test_data, transforms = None),
                             batch_size=batch_size, 
                             shuffle=True, 
                             **kwargs)
    return train_loader, test_loader, class_names


