import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1, 16,kernel_size=7, bias=False)
        self.conv2 = nn.Conv2d(16, 32,kernel_size=3, bias=False)
        self.conv3 = nn.Conv2d(32, 10, kernel_size=1, bias=False)
        self.gap = nn.AvgPool2d(10)

    def forward(self, x):
        # C1:n_i = 28, n_o = 22, s = 1, j_i = 1, j_o = 1, r_i = 1, r_o = 7
        x = F.relu(self.conv1(x)) 
        # C2:n_i = 22, n_o = 20, s = 1, j_i = 1, j_o = 1, r_i = 7, r_o = 9
        x = F.relu(self.conv2(x))
        # M1:n_i = 20, n_o = 10, s = 2, j_i = 1, j_o = 2, r_i = 9, r_o = 10
        x = F.relu(F.max_pool2d(x,2))
        # C4:n_i = 10, n_o = 20, s = 1, j_i = 2, j_o = 2, r_i = 10, r_o = 12
        x = F.relu(self.conv3(x))
        x = self.gap(x)
        return F.log_softmax(x,dim=1)