from math import sqrt

import torch
from torch import nn

# Global models
from .deepsdf import (
    DeepSDF, LatentModulated, LatentDemodulated,
)
# Part models (sdf only)
from .parts import sdfnet as parts_sdf
from .parts import get_part_latents, get_part_poses


def get_model(network, **kwargs):
    network = network.lower()
    # Global models (no parts)
    if network == "deepsdf":
        return DeepSDF(**kwargs)
    elif network == "latentmodulated":
        return LatentModulated(**kwargs)
    elif network == "latentdemodulated":
        return LatentDemodulated(**kwargs)
    # Part models (sdf only)
    elif network.split("-")[0] == "partsdf":
        network = network.split("-")[1]
        if network in ["partsdf", "singlemlp"]:
            return parts_sdf.PartSDF(**kwargs)
        else:
            raise NotImplementedError(f"Unknown parts model \"{network}\"")
    else:
        raise NotImplementedError(f"Unkown model \"{network}\"")


def get_latents(n_shapes, dim, max_norm=None, std=None, device=None, spherical=False):
    """Create and initialize latent vectors as embeddings."""
    latents = nn.Embedding(n_shapes, dim, max_norm=max_norm).to(device)
    if spherical:
        with torch.no_grad():
            latents.weight.data /= latents.weight.data.norm(dim=-1, keepdim=True)
    else:
        if std is None:
            std = 1. / sqrt(dim) if dim > 0 else 1.
        nn.init.normal_(latents.weight.data, 0., std)
    return latents