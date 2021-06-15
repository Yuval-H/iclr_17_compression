import torch
from torch import nn
import torch.nn.functional as F
from torch import tensor


class Binarizer(torch.autograd.Function):
    """
    An elementwise function that bins values
    to 0 or 1 depending on a threshold of
    0.5

    Input: a tensor with values in range(0,1)

    Returns: a tensor with binary values: 0 or 1
    based on a threshold of 0.5

    Equation(1) in paper
    """

    @staticmethod
    def forward(ctx, i):
        return (i > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def bin_values(x):
    return Binarizer.apply(x)



class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x): return self.func(x)

