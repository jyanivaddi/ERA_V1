import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torchinfo
from tqdm import tqdm

def model_summary(model, input_size):
    torchinfo.summary(model, 
                      input_size = input_size, 
                      batch_dim=0, 
                      col_names=("kernel_size",
                                 "input_size",
                                 "output_size",
                                 "num_params",
                                 "mult_adds"),
                       verbose=1,) 



class ResidualBlock(nn.Module):
    """

    """
    def __init__(self, in_channels, out_channels, kernel_size = 3, drop_out_probability=0.05, padding=1):
        super(ResidualBlock, self).__init__()

        # Conv layer 1
        self.conv1 = self.single_convolution(in_channels, out_channels, kernel_size, drop_out_probability, padding)

        # Conv layer 2
        self.conv2 = self.single_convolution(in_channels, out_channels, kernel_size, drop_out_probability, padding)

    def single_convolution(self,in_channels, out_channels,kernel_size, drop_out_probability, padding):
        """
        Define a convolution layer with batch normalization and drop out
        """
        # Define Conv Block
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=(kernel_size, kernel_size), 
                    padding=padding, 
                    padding_mode='reflect',
                    bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            #nn.Dropout(drop_out_probability)
        )
        return conv_block

    def __call__(self, x):
        x = self.conv1(x) 
        x = self.conv2(x) 
        return x
  

class Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, drop_out_probability=0.05, padding="same"):
        super(Layer, self).__init__()
        self.drop_out_probability = drop_out_probability
        self.padding = padding

        self.conv1 = self.pooled_convolution(in_channels, out_channels,kernel_size)
        self.res_block = ResidualBlock(out_channels, out_channels, kernel_size, drop_out_probability=self.drop_out_probability, padding=self.padding)

    def pooled_convolution(self, in_channels, out_channels, kernel_size):
        # Define Conv Block
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=(kernel_size, kernel_size), 
                    padding=1, 
                    padding_mode='reflect',
                    bias=False),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        return conv_block

    def __call__(self, x):
        x = self.conv1(x) 
        r1 = self.res_block(x) 
        x = x + r1
        return x
    

class CustomResnet(nn.Module):
    """
    Model definition
    parameters:
    ----------
    base_channels: number of base channels in the input data (for cifar-10 it is 3)
    num_classes: number of output classes in the input data (for cifar-10, it is 10)
    drop_out_probability: probability to use for dropout
    """
    def __init__(self, base_channels = 3, num_classes = 10, drop_out_probability = 0.05):
        super(CustomResnet,self).__init__()
        self.base_channels = base_channels
        self.drop_out_probability = drop_out_probability
        self.num_classes = num_classes

        # prep layer - 64 outputs
        prep_layer_out_channels = 64
        self.prep_layer = nn.Sequential(
            nn.Conv2d(in_channels=self.base_channels, 
                      out_channels=prep_layer_out_channels, 
                      kernel_size=(3, 3), 
                      padding=1, 
                      padding_mode='reflect', 
                      bias=False), 
            nn.BatchNorm2d(prep_layer_out_channels), 
            nn.ReLU()
        )

        # Layer 1 - 128 outputs
        layer_1_in_channels = prep_layer_out_channels
        layer_1_out_channels = 128
        self.layer_1 = Layer(layer_1_in_channels, layer_1_out_channels, kernel_size=3, padding=1)

        # Layer 2 - 256 outputs
        layer_2_in_channels = layer_1_out_channels
        layer_2_out_channels = 256
        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=layer_2_in_channels,
                      out_channels=layer_2_out_channels, 
                      kernel_size=(3, 3), 
                      padding=1, 
                      padding_mode='reflect', 
                      bias=False), 
            nn.MaxPool2d(2), 
            nn.BatchNorm2d(layer_2_out_channels), 
            nn.ReLU()
        )

        # Layer 3 - 512 outputs
        layer_3_in_channels = layer_2_out_channels
        layer_3_out_channels = 512
        self.layer_3 = Layer(layer_3_in_channels, layer_3_out_channels, kernel_size=3, padding=1)

        # Pool Layer - kernel size = 4
        self.pool = nn.MaxPool2d(4) 

        # FC layer - outputs = 10
        self.fc = nn.Linear(layer_3_out_channels, self.num_classes, bias=False)

    
    def forward(self, x):
        x = self.prep_layer(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.pool(x)
        x = self.fc(x.squeeze())
        x = x.view(-1,self.num_classes)
        return F.log_softmax(x)
