import torch
from torch import einsum, nn
from torch import autograd

class LocoProp(autograd.Function):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        pass