"""
Utilities for building models.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation import get_activation


##########
# Layers #
##########

class IgnoreLatentLinear(nn.Linear):
    """
    Linear layer taking as input a latent code that is ignored:
        layer(lat, x) = linear(x).
    """

    def forward(self, lat, x):
        return super().forward(x)


class LatentSequential(nn.Sequential):
    """
    Sequential module with an additional latent code input for its first module.
    """

    def forward(self, lat, x):
        x = self[0](lat, x)
        for module in self[1:]:
            x = module(x)
        return x


class BiasLayer(nn.Module):
    """
    Learnable bias layer.
    """

    def __init__(self, *dims):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(*dims))

    def forward(self, x):
        return x + self.bias

    def __repr__(self):
        return f"BiasLayer({tuple(self.bias.shape)})"


class PartCondLayerNorm(nn.Module):
    """
    Layer Normalization conditioned on parts.

    Assumes the inputs will be of shape (*, P, [D]) where P is the number of parts.
    It computes, with mu/sigma the mean/std of the input along the last dimension(s):
        y = (x - mu) / sigma * gamma + beta,
    and gamma/beta are learnable parameters of shape (P, [D]).
    """
    
    def __init__(self, n_parts, *dims, eps=1e-5, norm_input=True):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n_parts, *dims))
        self.beta = nn.Parameter(torch.zeros(n_parts, *dims))
        self.eps = eps
        self._dim = tuple(range(-len(dims), 0))  # last dimension(s) to normalize
        self.norm_input = norm_input  # whether to normalize the input before the conditioning

    def forward(self, x):
        if self.norm_input:
            mu = x.mean(dim=self._dim, keepdim=True)
            sigma = x.std(dim=self._dim, keepdim=True) + self.eps
            x = (x - mu) / sigma
        return x * self.gamma + self.beta
    
    def __repr__(self):
        n_parts, dims = self.gamma.shape[0], tuple(self.gamma.shape[1:])
        return f"PartCondLayerNorm({n_parts}, {dims}, eps={self.eps})"


class PartLinear(nn.Module):
    """
    Linear layer with part-wise weights.

    Basically, apply a linear layer for each part. (adapted from torch.nn.Linear)
    """
    __constants__ = ['n_parts', 'in_features', 'out_features']
    n_parts: int
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, n_parts: int, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.n_parts = n_parts
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((n_parts, out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(n_parts, out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1. / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = torch.einsum('...pi,pji->...pj', input, self.weight)
        if self.bias is not None:
            output = output + self.bias
        return output

    def extra_repr(self) -> str:
        return f'n_parts={self.n_parts}, in_features={self.in_features}, out_features={self.out_features}, ' + \
               f'bias={self.bias is not None}'


class ModulatedLinear(nn.Module):
    """
    Modulated linear module from Dupont et al., ICML 2022.
    
    Effectively similar to concatenating x and modulation before a single linear layer.
    The interpretation of shift modulations appears when using Sine activations.
    """

    def __init__(self, in_dim, out_dim, modulation_dim, init_fn=None, weight_norm=False):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        if init_fn is not None:
            self.linear.apply(init_fn)
        if weight_norm:
            self.linear = nn.utils.weight_norm(self.linear)

        if modulation_dim > 0:
            self.modulation = nn.Linear(modulation_dim, out_dim, bias=False)
            if weight_norm:
                self.modulation = nn.utils.weight_norm(self.modulation)
        else:
            self.modulation = None

    def forward(self, modulations, x):
        shifts = self.modulation(modulations) if self.modulation is not None else 0.
        return self.linear(x) + shifts


class DemodulatedLinear(nn.Module):
    """
    Demodulated linear module from StyleGAN2, Karras et al., 2020.

    Modulate scaling (or std) by scaling the weights and also renormalize them.
    """

    def __init__(self, in_dim, out_dim, modulation_dim, bias=True, weight_norm=False):
        super().__init__()

        # For linear
        self.weight = nn.Parameter(torch.rand(out_dim, in_dim))
        self._init_param(self.weight, in_dim)
        if bias:
            self.bias = nn.Parameter(torch.rand(out_dim))
            self._init_param(self.bias, in_dim)
        else:
            self.bias = 0.

        # For modulations
        self.modulation = nn.Linear(modulation_dim, in_dim, bias=bias)
        if weight_norm:
            self.modulation = nn.utils.weight_norm(self.modulation)
    
    @torch.no_grad()
    def _init_param(self, param, in_features):
        """Initialize the parameter, assuming it was sampled in [0,1)."""
        k = math.sqrt(1. / in_features)
        param *= 2 * k
        param -= k
    
    def forward(self, modulations, x):
        # Modulate weights
        scales = self.modulation(modulations).unsqueeze(-2)  # [B]x1xI
        weight_1 = self.weight * scales  # [B]xOxI

        # Demodulate/normalize weights (rsqrt() := 1/sqrt())
        weight_2 = weight_1 * torch.rsqrt(weight_1.square().sum(-2, keepdims=True) + 1e-8)  # [B]xOxI

        # Linear layer
        return (x.unsqueeze(-2) @ weight_2.transpose(-2, -1) + self.bias).squeeze(-2)  # [B]xO


class PartModulatedLinear(nn.Module):
    """
    Modified ModulatedLinear to handle parts.

    It will learn a different modulation function for each part.
    """

    def __init__(self, in_dim, out_dim, n_parts, modulation_dim, init_fn=None, weight_norm=False, bias=True):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        if init_fn is not None:
            self.linear.apply(init_fn)
        if weight_norm:
            self.linear = nn.utils.weight_norm(self.linear)

        if modulation_dim > 0:
            self.modulation = PartLinear(n_parts, modulation_dim, out_dim, bias=bias)
            if weight_norm:
                self.modulation = nn.utils.weight_norm(self.modulation)
        else:
            self.modulation = None

    def forward(self, modulations, x):
        shifts = self.modulation(modulations) if self.modulation is not None else 0.
        return self.linear(x) + shifts


##########
# Models #
##########

class MLP(nn.Module):
    """Multi-Layer Perceptron (MLP)."""

    def __init__(self, in_dim, out_dim, n_layers, hidden_dim, activation="relu", dp=0., weight_norm=False):
        """
        Args:
            in_dim (int): dimension of the input.
            out_dim (int): dimension of the output.
            n_layers (int >= 2): number of layers.
            hidden_dim (int): dimension of the hidden layers.
            activation (str): type of activation function.
            dp (float): dropout rate.
            weight_norm (bool): whether to apply weight normalization.
        """
        super().__init__()

        def get_layer(in_dim, out_dim, last_layer=False):
            """Make a single layer."""
            layer = []

            fc = nn.Linear(in_dim, out_dim)
            if weight_norm:
                fc = nn.utils.weight_norm(fc)
            layer.append(fc)

            if not last_layer:
                layer.append(get_activation(activation))
                if dp > 0:
                    layer.append(nn.Dropout(dp))

            return nn.Sequential(*layer)

        self.net = []
        self.net.append(get_layer(in_dim, hidden_dim))
        for _ in range(n_layers - 2):
            self.net.append(get_layer(hidden_dim, hidden_dim))
        self.net.append(get_layer(hidden_dim, out_dim, last_layer=True))
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)
    

class LatentMLP(nn.Module):
    """
    MLP (or similar) with additional latent input: 
        mlp(latent, x).
    """

    def __init__(self, lat_dim, in_dim, out_dim, layer_type, n_layers, hidden_dim, 
                 activation="relu", dp=0., weight_norm=False,
                 part_bias=None, part_norm=None, n_parts=None,
                 part_conv1d=None):
        """
        Args:
            lat_dim (int): dimension of the latent input.
            in_dim (int): dimension of the input.
            out_dim (int): dimension of the output.
            layer_type (str): type of layers ("linear", "modulated", "partmodulated", or "demodulated").
            n_layers (int >= 2): number of layers.
            hidden_dim (int): dimension of the hidden layers.
            activation (str): type of activation function.
            dp (float): dropout rate.
            weight_norm (bool): whether to apply weight normalization.
            part_bias (int, optional): if given (:=n_parts), add a learnable bias for each part.
            part_norm (int, optional): if given (:=n_parts), add a conditional layer norm for each part.
            n_parts (int, optional): number of parts for part-wise layers. (needed for "partmodulated")
            part_conv1d (int, optional): if given (:=n_parts), add 1x1 conv1d between layers for inter-part communication.
        """
        super().__init__()
        self.activ = get_activation(activation)

        assert n_layers >= 2, "At least two layers are required."

        def get_layer(in_dim, out_dim, first_layer=False, last_layer=False):
            """Make a single layer."""
            layer = []

            # First layer must consider latent
            if layer_type == "linear" and (not first_layer or lat_dim == 0):
                fc = IgnoreLatentLinear(in_dim, out_dim)
                if weight_norm:
                    fc = nn.utils.weight_norm(fc)
                layer.append(fc)
            elif layer_type == "modulated" or (layer_type == "linear" and first_layer):
                layer.append(ModulatedLinear(in_dim, out_dim, lat_dim, weight_norm=weight_norm))
            elif layer_type == "partmodulated":
                layer.append(PartModulatedLinear(in_dim, out_dim, n_parts, lat_dim, weight_norm=weight_norm))
            elif layer_type == "demodulated":
                layer.append(DemodulatedLinear(in_dim, out_dim, lat_dim, weight_norm=weight_norm))
            
            # Optional part bias and normalization
            if not last_layer:
                if part_bias is not None:
                    layer.append(BiasLayer(part_bias, out_dim))
                if part_norm is not None:
                    layer.append(PartCondLayerNorm(part_norm, out_dim))

            if not last_layer:
                layer.append(self.activ)
                if dp > 0:
                    layer.append(nn.Dropout(dp))

            return LatentSequential(*layer)

        self.layers = nn.ModuleList()
        self.layers.append(get_layer(in_dim, hidden_dim, first_layer=True))
        for _ in range(n_layers - 2):
            self.layers.append(get_layer(hidden_dim, hidden_dim))
        self.layers.append(get_layer(hidden_dim, out_dim, last_layer=True))

        # Optional 1x1 1D convolutions for inter-part communication
        self.part_conv1d = part_conv1d is not None
        if self.part_conv1d:
            def get_part_conv1d():
                return nn.Sequential(
                    nn.Conv1d(part_conv1d, part_conv1d, 1),
                    self.activ
                )
            self.conv_layers = nn.ModuleList(get_part_conv1d() for _ in range(n_layers - 1))

    def forward(self, lat, x):
        """
        Args:
            lat (torch.Tensor): latent codes. Should have singleton dimensions
                where the latents need to be repeated. E.g., [B, 1, 256] if
                x.shape = [B, N, 3] with N:=is where lat is repeated along.
            x(torch.Tensor): inputs.
        """
        for i, layer in enumerate(self.layers):
            x = layer(lat, x)
            if self.part_conv1d and i < len(self.layers) - 1:
                x = x + self.conv_layers[i](x.reshape(-1, x.shape[-2], x.shape[-1])).reshape(x.shape)
        return x