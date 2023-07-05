import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm

def model_summary(model, input_size):
    summary(model, input_size = input_size)


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
            nn.Dropout(drop_out_probability)
        )
        return conv_block

    def __call__(self, x):
        x = self.conv1(x) 
        x = self.conv2(x) 
        return x
  

class Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, drop_out_probability=0.05, padding="same", padding_mode = 'reflect'):
        super(Layer, self).__init__()
        self.drop_out_probability = drop_out_probability
        self.padding_mode = padding_mode
        self.padding = padding

        self.conv1 = self.pooled_convolution(in_channels, out_channels,kernel_size, padding=1, padding_mode = self.padding_mode)
        self.res_block = ResidualBlock(in_channels, out_channels, kernel_size, drop_out_probability=self.drop_out_probability, padding=self.padding)

    def pooled_convolution(self, in_channels, out_channels, kernel_size):
        # Define Conv Block
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=(kernel_size, kernel_size), 
                    padding=1, 
                    padding_mode=self.padding_mode,
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
    

class Model_Net(nn.Module):
    """
    Model definition
    parameters:
    ----------
    base_channels: number of base channels in the input data (for cifar-10 it is 3)
    num_classes: number of output classes in the input data (for cifar-10, it is 10)
    drop_out_probability: probability to use for dropout
    """
    def __init__(self, base_channels = 3, num_classes = 10, drop_out_probability = 0.05, padding_mode = 'reflect', use_depth_wise_conv=[False, True]):
        super(Model_Net,self).__init__()
        self.drop_out_probability = drop_out_probability
        self.num_classes = num_classes
        self.use_depth_wise_conv = use_depth_wise_conv
        self.padding_mode = padding_mode


        # Block 1
        # Number of input channels: 3
        # Number of output channels: 64
        # Dilation value: 2
        # Convolution layer types: Regular -> Depthwise Separable -> Dilated
        # Input RF: 1
        # Output RF: 
        self.block_1_in_channels = base_channels
        self.block_1_out_channels = 64
        dilation_val_block_1 = 2
        self.block1 = Block(self.block_1_in_channels,self.block_1_out_channels,
                            drop_out_probability=self.drop_out_probability, 
                            dilation_val_last=dilation_val_block_1, 
                            padding=1, padding_mode=self.padding_mode,
                            use_depth_wise_conv = self.use_depth_wise_conv)
        
        # Block 2
        # Number of input channels: 64
        # Number of output channels: 64
        # Dilation value: 4
        # Convolution layer types: Regular -> Depthwise Separable -> Dilated
        self.block_2_in_channels = self.block_1_out_channels
        self.block_2_out_channels = 64
        dilation_val_block_2 = 4
        self.block2 = Block(self.block_2_in_channels,self.block_2_out_channels,
                            drop_out_probability=self.drop_out_probability, 
                            dilation_val_last = dilation_val_block_2, 
                            padding=1, padding_mode=self.padding_mode,
                            use_depth_wise_conv = self.use_depth_wise_conv)

        # Block 3
        # Number of input channels: 64
        # Number of output channels: 32
        # Dilation value: 8
        # Convolution layer types: Regular -> Depthwise Separable -> Dilated
        self.block_3_in_channels = self.block_2_out_channels
        self.block_3_out_channels = 32
        dilation_val_block_3 = 8
        self.block3 = Block(self.block_3_in_channels,self.block_3_out_channels,
                            drop_out_probability=self.drop_out_probability, 
                            dilation_val_last=dilation_val_block_3, 
                            padding=0, padding_mode=self.padding_mode,
                            use_depth_wise_conv=self.use_depth_wise_conv)

        # Block 4
        # Number of input channels: 32
        # Number of output channels: 32
        # Dilation value: 16
        # Convolution layer types: Regular -> Depthwise Separable -> Dilated
        self.block_4_in_channels = self.block_3_out_channels
        self.block_4_out_channels = 32
        dilation_val_block_4 = 16
        self.block4 = Block(self.block_4_in_channels,self.block_4_out_channels,
                            drop_out_probability=self.drop_out_probability, 
                            dilation_val_last=dilation_val_block_4, 
                            padding=0, padding_mode=self.padding_mode,
                            use_depth_wise_conv=self.use_depth_wise_conv)

        # Adaptive Average Pooling
        # Number of input channels: 32
        # Number of output channels: 32
        self.aap = nn.AdaptiveAvgPool2d(1)

        # 1x1 convolution
        # Number of input channels: 32
        # Number of output channels: 10
        self.final = nn.Conv2d(self.block_4_out_channels,num_classes, kernel_size=(1,1),bias=False, padding=0) 
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.aap(x)
        x = self.final(x)
        x = x.view(-1,self.num_classes)
        return F.log_softmax(x)
