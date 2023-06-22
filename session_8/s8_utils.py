import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt


def load_cifar10_data(train_transforms, test_transforms):
    train_data = datasets.CIFAR10('../data', train=True, download=True, transform=train_transforms)
    test_data = datasets.CIFAR10('../data', train=False, download=True, transform=test_transforms)
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
    return train_data, test_data, class_names


def preview_batch_images(train_loader):
    """
    show the first 12 images from the batch
    """
    batch_data, batch_label = next(iter(train_loader))
    fig = plt.figure()
    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0))
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])

def plot_statistics(train_losses, train_acc, test_losses, test_acc, target_test_acc = 99):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].axhline(target_test_acc, color='r')
    axs[1, 1].set_title("Test Accuracy")

def get_incorrect_preictions(model, test_loader, device):
    model.eval()
    incorrect_predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            for d, t, p, o in zip(data, target,pred, output):
                if not p.eq(t.view_as(p)).item():
                    incorrect_predictions.append(
                        [d.cpu(), t.cpu(), p.cpu(),o[p.item()].cpu()]
                    )
    return incorrect_predictions

def preview_images(train_loader, class_names, num_rows = 5, num_cols = 5):
    batch_data, batch_label = next(iter(train_loader))
    num_images_to_preview = num_rows*num_cols
    for cnt in range(num_images_to_preview):
        plt.subplot(num_rows,num_cols,cnt+1)
        plt.tight_layout()
        this_img = np.asarray(batch_data[cnt])
        plt.imshow(this_img.transpose((1,2,0)))
        plt.title(class_names[str(batch_label[cnt].item())])
        plt.xticks([])
        plt.yticks([])
    plt.show()


def show_incorrect_predictions(incorrect_predictions, class_names, num_rows = 5, num_cols = 5):
    cnt = 0
    num_images_to_preview = num_rows*num_cols
    for this_pred in incorrect_predictions:
        orig_img = this_pred[0]
        target_label = this_pred[1]
        predicted_img = this_pred[2]
        output_label = this_pred[3]
        plt.subplot(num_rows,num_cols,cnt+1)
        plt.tight_layout()
        this_img = np.asarray(orig_img)
        plt.imshow(this_img.transpose((1,2,0)))
        title_str = f"{class_names[str(target_label.item())]}/{class_names[str(output_label.item())]}"
        plt.title(title_str)
        plt.xticks([])
        plt.yticks([])
        cnt+=1
        if cnt == num_images_to_preview:
            break
    plt.show()