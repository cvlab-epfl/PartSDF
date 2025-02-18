"""
Module implementing feature transforms for 
PyTorch models.
"""

from math import exp, log

import torch
from torch import nn
import torch.nn.functional as F

try:
	import tinycudann as tcnn
except ImportError:
	tcnn = None


def get_input_features(features):
    if isinstance(features, InputFeatures):
        return features
    elif isinstance(features, nn.Module):
        # Make sure we have outdim
        with torch.no_grad():
            _xyz = torch.randn(1, 3).to(features.B.device)
            features.outdim = features(_xyz).shape[-1]
        return features
    
    # Separate features from its arguments (if any)
    features = features.split('-')
    features, args = features[0], features[1:]
    args = [float(arg) for arg in args]  # convert arguments to floats

    if features.lower() == "fourier":
        if len(args) == 0:
            args.append(1.)
        if len(args) == 1:
            args.append(256)
        return FourierFeatures(*args)
    elif features.lower() in ["positional", "encoding", "pe"]:
        if len(args) == 0:
            args.append(10)
        return PositionalEncoding(*args)
    elif features.lower() in ["hash", "hashgrid"]:
        if len(args) == 0:
            args.append(16)
        return HashGrid(n_levels=args[0])
    else:
        raise NotImplementedError(f"Unknown input features \"{features}\".")


class InputFeatures(nn.Module):
    """Base class for input features."""

    def __init__(self) -> None:
        super().__init__()
        self.outdim = None


class PositionalEncoding(InputFeatures):
    """Positional encoding, as proposed in NeRF, Mildenhall et al., ECCV 2020."""

    def __init__(self, L, indim=3, add=False) -> None:
        super().__init__()
        self.L = L
        self.add = add
        
        factors = 2 ** torch.arange(self.L).unsqueeze(0) * torch.pi
        self.register_buffer('_factors', factors)

        self.outdim = 2 * L * indim
    
    def forward(self, x):
        x_proj = (x.unsqueeze(-1) @ self._factors)
        if self.add:
            x_proj = x_proj.sum(-2)
        else:
            x_proj = x_proj.flatten(-2)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    
    def __repr__(self):
        return f"PositionalEncoding(L={self.L})"


class FourierFeatures(InputFeatures):
    """Gaussian Fourier features, as proposed in Tancik et al., NeurIPS 2020."""

    def __init__(self, scale, mapdim=256, indim=3) -> None:
        super().__init__()
        self.scale = scale
        self.mapdim = mapdim
        indim = indim

        B = torch.randn(self.mapdim, indim) * self.scale**2
        self.register_buffer('B', B)

        self.outdim = 2 * self.mapdim
    
    def forward(self, x):
        x_proj = (2. * torch.pi * x) @ self.B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    
    def __repr__(self):
        return f"FourierFeatures(scale={self.scale}, mapdim={self.mapdim})"


class HashGrid(InputFeatures):
    """Multi-resolution hash encoding, as proposed in Müller et al., SIGGRAPH 2022."""

    def __init__(self, indim=3, n_levels=16, n_features_per_level=2, log2_hashmap_size=15, 
                 base_resolution=16, per_level_scale=1.26, max_resolution=None):
        super().__init__()
        self.config = {
            "otype": "HashGrid",
            "n_levels": n_levels,  # L (in paper)
            "n_features_per_level": n_features_per_level,  # F
            "log2_hashmap_size": log2_hashmap_size,  # log_2(T)
            "base_resolution": base_resolution,  # N_min
            "per_level_scale": per_level_scale  # b
	    }
        if max_resolution is not None:
            self.config["per_level_scale"] = exp((log(max_resolution) - log(base_resolution)) / (n_levels - 1))
        self.N_max = int(self.config['base_resolution'] * self.config['per_level_scale']**(self.config['n_levels']-1))
        
        if tcnn is None:
            print(f"Warning: multi-resolution hash encoding is not installed! Using identity mapping...")
            self.is_identity = True
            self.outdim = indim
        else:
            self.is_identity = False
            self.encoding = tcnn.Encoding(indim, self.config)
            self.outdim = self.encoding.n_output_dims
        
    def forward(self, x):
        if self.is_identity:
            return x
        return self.encoding(x.view(-1, x.shape[-1])).view(*x.shape[:-1], -1)
    
    def __repr__(self):
        return "Identity()" if self.is_identity else \
              f"HashGrid(N_min={self.config['base_resolution']}, N_max={self.N_max}, L={self.config['n_levels']})"