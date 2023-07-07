import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt



def get_incorrect_predictions(model, test_loader, device):
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
    inv_transforms = transforms.Compose([transforms.Normalize((0.,0.,0.,),
                                            (1./0.247,1./0.244,1./0.262)),
                                        transforms.Normalize((-0.491,-0.482,-0.447),
                                                             (1.0,1.0,1.0))])
    for cnt in range(num_images_to_preview):
        plt.subplot(num_rows,num_cols,cnt+1)
        plt.tight_layout()
        normalized_tensor_img = inv_transforms(batch_data[cnt].squeeze())
        this_img = np.asarray(normalized_tensor_img)
        plt.imshow(this_img.transpose((1,2,0)))
        plt.title(class_names[str(batch_label[cnt].item())])
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()

def show_incorrect_predictions(incorrect_predictions, class_names, num_rows = 5, num_cols = 5):
    cnt = 0
    num_images_to_preview = num_rows*num_cols
    fig = plt.figure()
    for this_pred in incorrect_predictions:
        orig_img = this_pred[0]
        inv_transforms = transforms.Compose([transforms.Normalize((0.,0.,0.,),
                                            (1./0.247,1./0.244,1./0.262)),
                                        transforms.Normalize((-0.491,-0.482,-0.447),
                                                             (1.0,1.0,1.0))])
        target_label = this_pred[1]
        predicted_label = this_pred[2]
        output_label = this_pred[3]
        un_normalized_tensor_img = inv_transforms(orig_img.squeeze())
        un_normalized_numpy_img = np.asarray(un_normalized_tensor_img)
        ax = fig.add_subplot(num_rows, num_cols, cnt+1,xticks=[],yticks=[])
        plt.subplot(num_rows,num_cols,cnt+1)
        plt.imshow(un_normalized_numpy_img.transpose((1,2,0)))
        title_str = f"{class_names[str(target_label.item())]}/{class_names[str(predicted_label.item())]}"
        ax.set_title(title_str,fontsize=8)
        cnt+=1
        if cnt == num_images_to_preview:
            break
    #plt.tight_layout()
    plt.show()

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

def plot_statistics_groups(train_losses_list, train_acc_list, test_losses_list, test_acc_list, target_test_acc = 99):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(range(1,20),train_losses_list[0])
    axs[0,0].plot(range(1,20),train_losses_list[1])
    axs[0,0].plot(range(1,20),train_losses_list[2])
    axs[0, 0].set_title("Training Loss per epoch")
    axs[0,0].legend(["BatchNorm","LayerNorm","GroupNorm"],loc='best')
    axs[1, 0].plot(range(1,20),train_acc_list[0])
    axs[1,0].plot(range(1,20),train_acc_list[1])
    axs[1,0].plot(range(1,20),train_acc_list[2])
    axs[1, 0].set_title("Training Accuracy per epoch")
    axs[1,0].legend(["BatchNorm","LayerNorm","GroupNorm"],loc='best')
    axs[0, 1].plot(range(1,20),test_losses_list[0])
    axs[0,1].plot(range(1,20),test_losses_list[1])
    axs[0,1].plot(range(1,20),test_losses_list[2])
    axs[0, 1].set_title("Test Loss per epoch")
    axs[0,1].legend(["BatchNorm","LayerNorm","GroupNorm"],loc='best')
    axs[1, 1].plot(range(1,20),test_acc_list[0])
    axs[1,1].plot(range(1,20),test_acc_list[1])
    axs[1,1].plot(range(1,20),test_acc_list[2])
    axs[1, 1].axhline(target_test_acc, color='r')
    axs[1, 1].set_title("Test Accuracy per epoch")
    axs[1,1].legend(["BatchNorm","LayerNorm","GroupNorm"],loc='best')
