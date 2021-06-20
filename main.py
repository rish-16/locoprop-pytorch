import torch
from torch import einsum, nn
import torch.nn.functional as F
from locoprop_pytorch import LocoProp

class Net(nn.Module):
    def __init__(self):
        super().__init__() 

    def forward(self, x):
        pass

    def backward(self):
        pass