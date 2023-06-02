import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from model import Net
from utils import preview_batch_images, plot_statistics, train, test, load_mnist_data

"""CODE BLOCK: 2"""

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

"""CODE BLOCK: 3"""

# Train data transformations
train_transforms = transforms.Compose([
    transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
    transforms.Resize((28, 28)),
    transforms.RandomRotation((-15., 15.), fill=0),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])

# Test data transformations
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

"""CODE BLOCK: 4"""
train_data, test_data = load_mnist_data(train_transforms, test_transforms)

"""CODE BLOCK: 5"""

batch_size = 512

kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}

test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
train_loader = torch.utils.data.DataLoader(train_data, **kwargs)

"""CODE BLOCK: 6"""
batch_data, batch_label = next(iter(train_loader))
preview_batch_images(batch_data, batch_label)


"""CODE BLOCK: 7"""
"""CODE BLOCK: 8"""

# Data to plot accuracy and loss graphs
train_losses = []
test_losses = []
train_acc = []
test_acc = []

test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

"""CODE BLOCK: 9"""

"""CODE BLOCK: 10"""

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, verbose=True)
# New Line
criterion = nn.CrossEntropyLoss()
#criterion = F.nll_loss
num_epochs = 20

for epoch in range(1, num_epochs+1):
  print(f'Epoch {epoch}')
  train(model, device, train_loader, optimizer, criterion, train_acc, train_losses)
  test(model, device, train_loader, criterion, test_acc, test_losses)
  scheduler.step()

"""CODE BLOCK: 11"""
plot_statistics(train_losses, train_acc, test_losses, test_acc)

from torchsummary import summary
model = Net().to(device)
summary(model, input_size=(1, 28, 28))

