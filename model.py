import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #input output kernel_size
        self.conv1 = nn.Conv2d(1, 32, (5,5))
        self.conv2 = nn.Conv2d(32, 64, (5,5))
        self.conv3 = nn.Conv2d(64, 128, (5,5))