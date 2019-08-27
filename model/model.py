import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pickle
from .constants import *

class myModel(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args

        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, KERNEL_NUMS, (kernel_size, EMBEDDED_DIM))
            for kernel_size in KERNEL_SIZES
        ])
        self.leakyRelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(DROP_OUT)
        self.fc1 = nn.Linear(len(KERNEL_SIZES)*KERNEL_NUMS, 256)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):

        x = Variable(x, requires_grad=True)
    
        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv_layers]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x,1)
        x = self.dropout(x)
        x = self.fc1(x)
        
        if not self.training:
            x = self.softmax(x)

        return x
