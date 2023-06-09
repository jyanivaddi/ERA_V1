import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1, 16,kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 16,kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        
        self.conv3 = nn.Conv2d(16, 16,kernel_size=3, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, bias=False)
        self.bn4 = nn.BatchNorm2d(16)

        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, bias=False)
        self.bn5 = nn.BatchNorm2d(16)

        self.conv6 = nn.Conv2d(16, 16, kernel_size=3, bias=False)
        self.bn6 = nn.BatchNorm2d(16)

        self.conv7 = nn.Conv2d(16, 16, kernel_size=3, bias=False)
        self.bn7 = nn.BatchNorm2d(16)

        self.conv8 = nn.Conv2d(16,10, kernel_size=1, bias=False)
        self.bn8 = nn.BatchNorm2d(10)
        
        self.gap = nn.AvgPool2d(4)


    def forward(self, x):
        # C1:n_i = 28, n_o = 26, s = 1, j_i = 1, j_o = 1, r_i = 1, r_o = 3
        x = F.relu(self.conv1(x)) # C1
        x = self.bn1(x) # BN1

        # C2:n_i = 26, n_o = 24, s = 1, j_i = 1, j_o = 1, r_i = 3, r_o = 5
        x = F.relu(self.conv2(x)) # C2
        x = self.bn2(x) # BN2
        
        # C3:n_i = 24, n_o = 22, s = 1, j_i = 1, j_o = 1, r_i = 5, r_o = 7
        x = F.relu(self.conv3(x)) # C3
        x = self.bn3(x) # BN3
        
        # C4:n_i = 22, n_o = 20, s = 1, j_i = 1, j_o = 1, r_i = 7, r_o = 9
        x = F.relu(self.conv4(x)) # C4
        x = self.bn4(x) # BN4

        # MP:n_i = 20, n_o = 10, s = 2, j_i = 1, j_o = 2, r_i = 9, r_o = 10
        x = F.relu(F.max_pool2d(x,2)) # MP1

        # C5:n_i = 10, n_o = 8, s = 1, j_i = 2, j_o = 2, r_i = 10, r_o = 14
        x = F.relu(self.conv5(x)) # C5 
        x = self.bn5(x) # BN5

        # C6:n_i = 8, n_o = 6, s = 1, j_i = 2, j_o = 2, r_i = 14, r_o = 18
        x = F.relu(self.conv6(x)) # C6
        x = self.bn6(x) # BN6

        # C7:n_i = 6, n_o = 4, s = 1, j_i = 2, j_o = 2, r_i = 18, r_o = 22
        x = F.relu(self.conv7(x)) # C7 
        x = self.bn7(x) # BN7

        # C8:n_i = 4, n_o = 4, s = 1, j_i = 2, j_o = 2, r_i = 22, r_o = 22
        x = F.relu(self.conv8(x)) # C8 
        x = self.bn8(x) # BN8


        x = self.gap(x)
        x = x.view(-1,10)
        return F.log_softmax(x,dim=1)


def model_summary(model, input_size):
    summary(model, input_size = input_size)



def model_train(model, device, train_loader, optimizer, train_acc, train_losses):
    model.train()
    pbar = tqdm(train_loader)
    train_loss = 0
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        train_loss+=loss.item()
        loss.backward()
        optimizer.step()
        correct+= output.argmax(dim=1).eq(target).sum().item()
        processed+= len(data)
        pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx} Accuracy = {100*correct/processed:0.2f}')

    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))



def model_test(model, device, test_loader, test_acc, test_losses):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc.append(100.*correct/len(test_loader.dataset))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_losses.append(test_loss)