import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm


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
    return  loss.item()



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
    return test_loss



class Model_1_Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Conv Block 1
        # input_size: 28 output_size = 26 
        # rf_in: 1, s = 1, j_in = 1, j_out = 1, rf_out = 3
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        )

        # Conv Block 2
        # input_size: 26 output_size = 24 
        # rf_in: 3, s = 1, j_in = 1, j_out = 1, rf_out = 5
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) 

        # Conv Block 3 
        # input_size: 24 output_size = 22 
        # rf_in: 5, s = 1, j_in = 1, j_out = 1, rf_out = 7
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) 

        # Transition Block 1
        # Max Pool
        # input_size: 22 output_size = 11 
        # rf_in: 7, s = 2, j_in = 1, j_out = 2, rf_out = 8
        self.pool1 = nn.MaxPool2d(2,2) 

        # 1x1 conv Conv Block 4
        # input_size: 11 output_size = 11 
        # rf_in: 8, s = 1, j_in = 2, j_out = 2, rf_out = 8
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        ) 

        # Conv Block 4
        # input_size: 11 output_size = 9 
        # rf_in: 8, s = 1, j_in = 2, j_out = 2, rf_out = 12
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) 

        # Conv Block 5
        # input_size: 9 output_size = 7 
        # rf_in: 12, s = 1, j_in = 2, j_out = 2, rf_out = 16
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        )

        # Conv Block 6
        # input_size: 7 output_size = 5 
        # rf_in: 16, s = 1, j_in = 2, j_out = 2, rf_out = 20
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        )

        # Adaptive Average Pooling
        self.aap = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.aap(x)
        x = x.view(-1,10)
        return F.log_softmax(x,dim=1)


class Model_2_Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Conv Block 1
        # input_size: 28 output_size = 26 
        # rf_in: 1, s = 1, j_in = 1, j_out = 1, rf_out = 3
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        )

        # Conv Block 2
        # input_size: 26 output_size = 24 
        # rf_in: 3, s = 1, j_in = 1, j_out = 1, rf_out = 5
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) 

        # Conv Block 3 
        # input_size: 24 output_size = 22 
        # rf_in: 5, s = 1, j_in = 1, j_out = 1, rf_out = 7
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) 

        # Conv Block 4 
        # input_size: 22 output_size = 20 
        # rf_in: 7, s = 1, j_in = 1, j_out = 1, rf_out = 9
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) 

        # Transition Block 1
        # Max Pool
        # input_size: 20 output_size = 10 
        # rf_in: 9, s = 2, j_in = 1, j_out = 2, rf_out = 10
        self.pool1 = nn.MaxPool2d(2,2) 

        # Conv Block 5 
        # input_size: 10 output_size = 8 
        # rf_in: 10, s = 1, j_in = 2, j_out = 2, rf_out = 14
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) 

        # Conv Block 6 
        # input_size: 8 output_size = 6 
        # rf_in: 14, s = 1, j_in = 2, j_out = 2, rf_out = 18
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) 

        # Conv Block 7 
        # input_size: 6 output_size = 4 
        # rf_in: 18, s = 1, j_in = 2, j_out = 2, rf_out = 22
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) 

        # 1x1 conv 
        # input_size: 4 output_size = 4 
        # rf_in: 22, s = 1, j_in = 2, j_out = 2, rf_out = 22
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        ) 

        # Adaptive Average Pooling
        self.aap = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.pool1(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.aap(x)
        x = x.view(-1,10)
        return F.log_softmax(x,dim=1)

class Model_3_Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Conv Block 1
        # input_size: 28 output_size = 26 
        # rf_in: 1, s = 1, j_in = 1, j_out = 1, rf_out = 3
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        )

        # Conv Block 2
        # input_size: 26 output_size = 24 
        # rf_in: 3, s = 1, j_in = 1, j_out = 1, rf_out = 5
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) 

        # Conv Block 3 
        # input_size: 24 output_size = 22 
        # rf_in: 5, s = 1, j_in = 1, j_out = 1, rf_out = 7
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) 

        # Conv Block 4 
        # input_size: 22 output_size = 20 
        # rf_in: 7, s = 1, j_in = 1, j_out = 1, rf_out = 9
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) 

        # Transition Block 1
        # Max Pool
        # input_size: 20 output_size = 10 
        # rf_in: 9, s = 2, j_in = 1, j_out = 2, rf_out = 10
        self.pool1 = nn.MaxPool2d(2,2) 

        # Conv Block 5 
        # input_size: 10 output_size = 8 
        # rf_in: 10, s = 1, j_in = 2, j_out = 2, rf_out = 14
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) 

        # Conv Block 6 
        # input_size: 8 output_size = 6 
        # rf_in: 14, s = 1, j_in = 2, j_out = 2, rf_out = 18
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) 

        # Conv Block 7 
        # input_size: 6 output_size = 4 
        # rf_in: 18, s = 1, j_in = 2, j_out = 2, rf_out = 22
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) 

        # 1x1 conv 
        # input_size: 4 output_size = 4 
        # rf_in: 22, s = 1, j_in = 2, j_out = 2, rf_out = 22
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) 

        # Adaptive Average Pooling
        self.aap = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.pool1(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.aap(x)
        x = x.view(-1,10)
        return F.log_softmax(x,dim=1)


class Model_4_Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Conv Block 1
        # input_size: 28 output_size = 26 
        # rf_in: 1, s = 1, j_in = 1, j_out = 1, rf_out = 3
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        )

        # Conv Block 2
        # input_size: 26 output_size = 24 
        # rf_in: 3, s = 1, j_in = 1, j_out = 1, rf_out = 5
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) 

        # Conv Block 3 
        # input_size: 24 output_size = 22 
        # rf_in: 5, s = 1, j_in = 1, j_out = 1, rf_out = 7
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) 

        # Transition Block 1
        # Max Pool
        # input_size: 22 output_size = 11 
        # rf_in: 7, s = 2, j_in = 1, j_out = 2, rf_out = 8
        self.pool1 = nn.MaxPool2d(2,2) 

        # Conv Block 4 
        # input_size: 11 output_size = 9 
        # rf_in: 8, s = 1, j_in = 2, j_out = 2, rf_out = 12
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) 

        # Conv Block 5 
        # input_size: 9 output_size = 7 
        # rf_in: 12, s = 1, j_in = 2, j_out = 2, rf_out = 16
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) 

        # Conv Block 6 
        # input_size: 7 output_size = 5 
        # rf_in: 16, s = 1, j_in = 2, j_out = 2, rf_out = 20
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) 

        # Adaptive Average Pooling
        self.aap = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.aap(x)
        x = x.view(-1,10)
        return F.log_softmax(x,dim=1)


class Model_5_Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Conv Block 1
        # input_size: 28 output_size = 26 
        # rf_in: 1, s = 1, j_in = 1, j_out = 1, rf_out = 3
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Conv Block 2
        # input_size: 26 output_size = 24 
        # rf_in: 3, s = 1, j_in = 1, j_out = 1, rf_out = 5
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.1)
        ) 

        # Conv Block 3 
        # input_size: 24 output_size = 22 
        # rf_in: 5, s = 1, j_in = 1, j_out = 1, rf_out = 7
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.1)
        ) 

        # Conv Block 4 
        # input_size: 22 output_size = 20 
        # rf_in: 7, s = 1, j_in = 1, j_out = 1, rf_out = 9
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.1)
        ) 

        # Transition Block 1
        # Max Pool
        # input_size: 20 output_size = 10 
        # rf_in: 9, s = 2, j_in = 1, j_out = 2, rf_out = 10
        self.pool1 = nn.MaxPool2d(2,2) 

        # Conv Block 5 
        # input_size: 10 output_size = 8 
        # rf_in: 10, s = 1, j_in = 2, j_out = 2, rf_out = 14
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(14),
            nn.ReLU(),
            nn.Dropout(0.1)
        ) 

        # Conv Block 6 
        # input_size: 8 output_size = 6 
        # rf_in: 14, s = 1, j_in = 2, j_out = 2, rf_out = 18
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(14),
            nn.ReLU(),
            nn.Dropout(0.1)
        ) 

        # Conv Block 7 
        # input_size: 6 output_size = 4 
        # rf_in: 18, s = 1, j_in = 2, j_out = 2, rf_out = 22
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(14),
            nn.ReLU(),
            nn.Dropout(0.1)
        ) 

        # 1 x 1 conv
        # input_size: 6 output_size = 4 
        # rf_in: 18, s = 1, j_in = 2, j_out = 2, rf_out = 22
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
        ) 

        # Adaptive Average Pooling
        self.aap = nn.AdaptiveAvgPool2d((1,1))


    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.pool1(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.aap(x)
        x = x.view(-1,10)
        return F.log_softmax(x,dim=1)


class Model_6_Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Conv Block 1
        # input_size: 28 output_size = 26 
        # rf_in: 1, s = 1, j_in = 1, j_out = 1, rf_out = 3
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Conv Block 2
        # input_size: 26 output_size = 24 
        # rf_in: 3, s = 1, j_in = 1, j_out = 1, rf_out = 5
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(0.1)
        ) 

        # Transition Block 1
        # Max Pool
        # input_size: 24 output_size = 12 
        # rf_in: 5, s = 2, j_in = 1, j_out = 2, rf_out = 6
        self.pool1 = nn.MaxPool2d(2,2) 


        # Conv Block 3 
        # input_size: 12 output_size = 10 
        # rf_in: 6, s = 1, j_in = 2, j_out = 2, rf_out = 10
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.1)
        ) 

        # Conv Block 5 
        # input_size: 10 output_size = 8 
        # rf_in: 10, s = 1, j_in = 2, j_out = 2, rf_out = 14
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(0.1)
        ) 

        # Conv Block 6 
        # input_size: 8 output_size = 6 
        # rf_in: 14, s = 1, j_in = 2, j_out = 2, rf_out = 18
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(0.1)
        ) 

        # Conv Block 7 
        # input_size: 6 output_size = 4 
        # rf_in: 18, s = 1, j_in = 2, j_out = 2, rf_out = 22
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(0.1)
        ) 

        # Conv Block 7 
        # input_size: 4 output_size = 2 
        # rf_in: 22, s = 1, j_in = 2, j_out = 2, rf_out = 26
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(0.1)
        ) 

        # 1 x 1 conv
        # input_size: 2 output_size = 2 
        # rf_in: 26, s = 1, j_in = 2, j_out = 2, rf_out = 26
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
        ) 

        # Adaptive Average Pooling
        self.aap = nn.AdaptiveAvgPool2d((1,1))


    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.aap(x)
        x = x.view(-1,10)
        return F.log_softmax(x,dim=1)
