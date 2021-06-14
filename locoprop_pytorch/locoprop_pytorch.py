import torch
from torch import einsum, nn
from torch import autograd

class LocoProp(autograd.Function):
    def __init__(self):
        super().__init__()

    @staticmethod
    def backward(ctx, grad_output):
        pass