"""
Module implementing activation functions for 
PyTorch models.
"""

import torch
from torch import nn
import torch.nn.functional as F


def get_activation(activation):
    """Return the Module corresponding to the activation."""
    if isinstance(activation, nn.Module):
        return activation
    
    # Separate activation from its arguments (if any)
    activation = activation.split('-')
    activation, args = activation[0], activation[1:]
    args = [float(arg) for arg in args]  # convert arguments to floats

    if activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() in ["leaky", "lrelu", "leakyrelu"]:
        if len(args) == 0:
            args.append(0.1)
        return nn.LeakyReLU(*args)
    elif activation.lower() == "celu":
        return nn.CELU(*args)
    elif activation.lower() == "gelu":
        return nn.GELU(approximate='tanh', *args)
    elif activation.lower() == "softplus":
        if len(args) == 0:
            args.append(100.)
        return nn.Softplus(*args)
    elif activation.lower() == "geglu":
        return GEGLU(*args)
    else:
        raise NotImplementedError(f"Unknown activation \"{activation}\".")


class GEGLU(nn.Module):
    """
    GeGLU activation function.

    Taken from 3DShape2VecSet, Zhang et al., SIGGRAPH23.
    https://github.com/1zb/3DShape2VecSet/blob/master/models_ae.py
    """

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)
    
    def __repr__(self):
        return f"GEGLU()"