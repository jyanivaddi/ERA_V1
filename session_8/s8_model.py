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



class Model_Net(nn.Module):

    def __init__(self, norm_type = 'batch_norm',num_groups = 1):
        super().__init__()

        self.NUM_GROUPS=num_groups
        #norm_type='batch_norm'

        if norm_type == 'batch_norm':
            self.norm_func = nn.BatchNorm2d
            self.norm_args = [] 
        elif norm_type == 'layer_norm':
            self.norm_func = nn.GroupNorm
            self.norm_args = [1]
        else:
            self.norm_func = nn.GroupNorm
            self.norm_args = [self.NUM_GROUPS]

        self.drop_out_probability = 0.02

        # input_size: 32 output_size = 32 
        # rf_in: 1, k=3, s = 1, p=1, j_in = 1, j_out = 1, rf_out = 3
        self.conv_block_1_1 = self.conv_block(3, 16, padding=1) 

        # input_size: 32 output_size = 32 
        # rf_in: 3, k=3, s = 1, p=1, j_in = 1, j_out = 1, rf_out = 5
        self.conv_block_1_2 = self.conv_block(16, 32, padding=1) 

        # Max Pool
        # input_size: 32 output_size = 16 
        # rf_in: 5, k=2, s = 2, p=0, j_in = 1, j_out = 2, rf_out = 6
        # 1 x 1 
        # input_size: 16 output_size = 16 
        # rf_in: 6, k=1, s = 1, p=0, j_in = 2, j_out = 2, rf_out = 6
        self.transition_block_1 = self.transition_block_with_max_pool(32,16) 

        # input_size: 16 output_size = 16 
        # rf_in: 6, k=3, s = 1, p=1, j_in = 2, j_out = 2, rf_out = 10
        self.conv_block_2_1 = self.conv_block(16,32, padding=1) 

        # input_size: 16 output_size = 16 
        # rf_in: 10, k=3, s = 1, p=1, j_in = 2, j_out = 2, rf_out = 14
        self.conv_block_2_2 = self.conv_block(32,32, padding=1) 

        # input_size: 16 output_size = 16 
        # rf_in: 14, k=3, s = 1, p=1, j_in = 2, j_out = 2, rf_out = 18
        self.conv_block_2_3 = self.conv_block(32,32, padding=1) 

        # Max Pool
        # input_size: 16 output_size = 8 
        # rf_in: 18, k=2, s = 2, p=0, j_in = 2, j_out = 4, rf_out = 20
        # 1 x 1 
        # input_size: 8 output_size = 8 
        # rf_in: 20, k=1, s = 1, p=0, j_in = 4, j_out = 4, rf_out = 20        
        self.transition_block_2 = self.transition_block_with_max_pool(32,16) 

        # input_size: 8 output_size = 8 
        # rf_in: 20, k=3, s = 1, p=1, j_in = 4, j_out = 4, rf_out = 28
        self.conv_block_3_1 = self.conv_block(16,16, padding=1) 

        # input_size: 8 output_size = 8 
        # rf_in: 28, k=3, s = 1, p=1, j_in = 4, j_out = 4, rf_out = 36
        self.conv_block_3_2 = self.conv_block(16,32, padding=0)  

        # input_size: 8 output_size = 8 
        # rf_in: 36, k=3, s = 1, p=1, j_in = 4, j_out = 4, rf_out = 44
        self.conv_block_3_3 = self.conv_block(32,32, padding=0) 

        # AAP
        # input_size: 8 output_size = 1 
        # rf_in: 44, k=1, s = 1, p=0, j_in = 4, j_out = 4, rf_out = 44        
        self.aap = nn.AdaptiveAvgPool2d((1,1)) 

        # 1 x 1 
        # input_size: 1 output_size = 1 
        # rf_in: 44, k=1, s = 1, p=0, j_in = 4, j_out = 4, rf_out = 44        
        self.transition_block_3 = self.transition_block_wo_max_pool(32,10)

    def conv_block(self, in_channels, out_channels, kernel_size=3, padding=0):
        self.norm_args.append(out_channels)
        # Define Conv Block
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=out_channels, 
                      kernel_size=(kernel_size, kernel_size), 
                      padding=padding, 
                      bias=False),
            self.norm_func(*self.norm_args),
            nn.ReLU(),
            nn.Dropout(self.drop_out_probability)
        )
        self.norm_args.pop(-1)
        return conv_block

    def transition_block_with_max_pool(self, in_channels, out_channels):
        self.norm_args.append(out_channels)
        transition_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                        out_channels= out_channels,
                        kernel_size = (1,1),
                        padding=0,
                        bias=False),
            self.norm_func(*self.norm_args),
            nn.ReLU(),
            nn.Dropout(self.drop_out_probability),
            nn.MaxPool2d(2,2),
        )
        self.norm_args.pop(-1)
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
