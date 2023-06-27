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


class DepthWiseSeparable(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, drop_out_probability=0.05, padding=1):
        super(DepthWiseSeparable, self).__init__()
        self.drop_out_probability = drop_out_probability
        self.g1 = self.grouped_convolution(in_channels, kernel_size, padding) 
        self.i1 = self.one_one_convolution(in_channels, out_channels)
        return

    def __call__(self, x):
        x = self.g1(x)
        x = self.i1(x)
        return x

    def grouped_convolution(self,in_channels,kernel_size, padding):
        # Define Conv Block
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                    out_channels=in_channels, 
                    kernel_size=(kernel_size, kernel_size), 
                    padding=padding, 
                    bias=False,
                    groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Dropout(self.drop_out_probability)
        )
        return conv_block

    def one_one_convolution(self, in_channels, out_channels):
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, drop_out_probability=0.05, use_pool=False, padding=1, dilation_val_last=1, use_dilated_kernel_on_last = False):
        super(Block, self).__init__()
        self.drop_out_probability = drop_out_probability
        #self.conv1 = self.single_convolution(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=1)
        self.conv1 = DepthWiseSeparable(in_channels, out_channels,kernel_size, padding)
        #self.conv2 = self.single_convolution(out_channels, out_channels*2,kernel_size=kernel_size, padding=padding, dilation=1)
        self.conv2 = DepthWiseSeparable(out_channels, out_channels*2,kernel_size, padding)
        self.use_dilated_kernel_on_last = use_dilated_kernel_on_last
        if use_dilated_kernel_on_last:
            self.conv3 = self.single_convolution(out_channels*2, out_channels,kernel_size=kernel_size, padding=padding, dilation=1)
            self.dilated = self.dilated_convolution(out_channels, out_channels, kernel_size=kernel_size, padding='same', dilation=dilation_val_last)
        else:
            if use_pool:
                self.conv3 = self.transition_block_with_max_pool(out_channels*2, out_channels)
            else:
                self.conv3 = self.transition_block_wo_max_pool(out_channels*2, out_channels)


    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_dilated_kernel_on_last:
            x = self.conv3(x)
            x1 = self.dilated(x)
            x = x + x1
        else:
            x = self.conv3(x)
        return x
    

    def dilated_convolution(self, in_channels, out_channels, kernel_size, padding=1, dilation=1):
        # Define Conv Block
        dilated_conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=out_channels, 
                      kernel_size=(kernel_size, kernel_size), 
                      padding=padding, 
                      dilation=dilation, 
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(self.drop_out_probability)
            )
        return dilated_conv_block

    def single_convolution(self,in_channels, out_channels,kernel_size, padding=1, dilation=1):
        # Define Conv Block
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=(kernel_size, kernel_size), 
                    padding=padding, 
                    dilation=dilation,
                    bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(self.drop_out_probability)
        )
        return conv_block
    
    def transition_block_with_max_pool(self, in_channels, out_channels):
        transition_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                        out_channels= out_channels,
                        kernel_size = (1,1),
                        padding=0,
                        bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(self.drop_out_probability),
            nn.MaxPool2d(2,2),
        )
        return transition_block

    def transition_block_wo_max_pool(self, in_channels, out_channels):
        transition_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                        out_channels= out_channels,
                        kernel_size = (1,1),
                        padding=0,
                        bias=False),
        )
        return transition_block
    

# Transition Block 1
# Max Pool
# input_size: 24 output_size = 12 
# rf_in: 5, s = 2, j_in = 1, j_out = 2, rf_out = 6

class Model_Net(nn.Module):

    def __init__(self, base_channels = 3, num_classes = 10, drop_out = 0.05):
        super(Model_Net,self).__init__()
        self.drop_out_probability = drop_out
        self.num_classes = num_classes

        self.block_1_in_channels = base_channels
        self.block_1_out_channels = 64
        self.block1 = Block(self.block_1_in_channels,self.block_1_out_channels,drop_out_probability=self.drop_out_probability, use_dilated_kernel_on_last = True, dilation_val_last=2, use_pool=False, padding=0)
        
        self.block_2_in_channels = self.block_1_out_channels
        self.block_2_out_channels = 32
        self.block2 = Block(self.block_2_in_channels,self.block_2_out_channels,drop_out_probability=self.drop_out_probability, use_dilated_kernel_on_last = True, dilation_val_last = 2, use_pool=False, padding=0)

        self.block_3_in_channels = self.block_2_out_channels
        self.block_3_out_channels = 32
        self.block3 = Block(self.block_3_in_channels,self.block_3_out_channels,drop_out_probability=self.drop_out_probability, use_dilated_kernel_on_last = True, dilation_val_last=2, use_pool=False, padding=0)

        self.aap = nn.AdaptiveAvgPool2d(1)
        self.final = nn.Conv2d(self.block_3_out_channels,num_classes, kernel_size=(1,1),bias=False, padding=0) 
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.aap(x)
        x = self.final(x)
        x = x.view(-1,self.num_classes)
        return F.log_softmax(x)
