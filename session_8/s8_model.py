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

# Transition Block 1
# Max Pool
# input_size: 24 output_size = 12 
# rf_in: 5, s = 2, j_in = 1, j_out = 2, rf_out = 6

# input_size: 12 output_size = 10 
# rf_in: 6, s = 1, j_in = 2, j_out = 2, rf_out = 10


class Model_Net(nn.Module):

    def conv_block(self, in_channels, out_channels, kernel_size=3, padding=0):
        # Define Conv Block
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=out_channels, 
                      kernel_size=(kernel_size, kernel_size), 
                      padding=padding, 
                      bias=False),
            self.norm_func(out_channels),
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
            self.norm_func(out_channels),
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


    def __init__(self, norm_type='bn'):
        super().__init__()

        if norm_type == 'bn':
            self.norm_func = nn.BatchNorm2d
        else:
            self.norm_func = None

        self.drop_out_probability = 0.05

        self.conv_block_1_1 = self.conv_block(3, 16, padding=1) # 32
        self.conv_block_1_2 = self.conv_block(16, 16, padding=1) # 32
        self.transition_block_1 = self.transition_block_with_max_pool(16,32) # 16
        self.conv_block_2_1 = self.conv_block(32,32, padding=1) # 16
        self.conv_block_2_2 = self.conv_block(32,32, padding=1) # 16
        self.conv_block_2_3 = self.conv_block(32,32, padding=1) # 16
        self.transition_block_2 = self.transition_block_with_max_pool(32,64) # 8
        self.conv_block_3_1 = self.conv_block(64,64, padding=1) # 8
        self.conv_block_3_2 = self.conv_block(64,64, padding=1)  # 8
        self.conv_block_3_3 = self.conv_block(64,64, padding=1) # 8
        self.aap = nn.AdaptiveAvgPool2d((1,1)) # 1
        self.transition_block_3 = self.transition_block_wo_max_pool(64,10)


    def forward(self, x):
        x = self.conv_block_1_1(x)
        x = self.conv_block_1_2(x)
        x = self.transition_block_1(x)
        x = self.conv_block_2_1(x)
        x = self.conv_block_2_2(x)
        x = self.conv_block_2_3(x)
        x = self.transition_block_2(x)
        x = self.conv_block_3_1(x)
        x = self.conv_block_3_2(x)
        x = self.conv_block_3_3(x)
        x = self.aap(x)
        x = self.transition_block_3(x)
        x = x.view(-1,10)
        return F.log_softmax(x)
