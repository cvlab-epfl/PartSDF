"""
Module for the latent and parametric space of the parts.
"""

import os.path
from typing import Optional
from math import sqrt

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from ...primitives import standardize_quaternion

# Latent
########

def get_part_latents(n_shapes, n_parts, dim, max_norm=None, std=None, device=None, spherical=False):
    """
    Create and initialize part latent vectors as embeddings.

    A single embedding will be of dimension (n_parts, dim).
    """
    latents = PartEmbedding(n_shapes, n_parts, dim, max_norm=max_norm).to(device)
    if spherical:
        with torch.no_grad():
            latents.weight.data /= latents.weight.data.norm(dim=-1, keepdim=True)
    else:
        if std is None:
            std = 1. / sqrt(dim) if dim > 0 else 1.
        nn.init.normal_(latents.weight.data, 0., std)
    return latents


class PartEmbedding(nn.Embedding):
    """
    Module for the part latent vectors as embeddings.

    It will store the embeddings of all shape and all parts together, but will retrieve
    all part embeddings for a given shape index.

    Note: under the hood, the part latents of the same shape are stored consecutively
    in the weight tensor to be contiguous in memory.
    """

    n_parts: int  

    def __init__(self, num_embeddings: int, n_parts: int, embedding_dim: int, padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight: Optional[Tensor] = None, _freeze: bool = False,
                 device=None, dtype=None) -> None:
        assert padding_idx is None, "padding_idx is not implemented for PartEmbedding"
        super().__init__(num_embeddings * n_parts, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq,
                         sparse, _weight, _freeze, device, dtype)
        self.num_embeddings = num_embeddings
        self.n_parts = n_parts

        # Reshape the weight to [num_embeddings, n_parts, embedding_dim]
        self.weight = Parameter(self.weight.view(num_embeddings, n_parts, embedding_dim), 
                                requires_grad=self.weight.requires_grad)

        # Prepare indices offset for the parts
        self.register_buffer('_part_idx_offset', torch.arange(n_parts, device=device))

    def forward(self, input: Tensor) -> Tensor:
        # Flatten the indices and embeddings (adding part indices to shape indices)
        input = input.unsqueeze(-1) * self.n_parts + self._part_idx_offset
        weight = self.weight.view(-1, self.embedding_dim)

        # Retrieve the embeddings with an additional part dimension
        return  F.embedding(
            input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

    def extra_repr(self) -> str:
        s = '{num_embeddings}, {n_parts}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        return s.format(**self.__dict__)

    @classmethod
    def from_pretrained(cls, embeddings, freeze=True, padding_idx=None,
                        max_norm=None, norm_type=2., scale_grad_by_freq=False,
                        sparse=False):
        assert embeddings.dim() == 3, \
            'PartEmbeddings parameter is expected to be 3-dimensional'
        rows, parts, cols = embeddings.shape
        embedding = cls(
            num_embeddings=rows,
            n_parts=parts,
            embedding_dim=cols,
            _weight=embeddings,
            _freeze=freeze,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse)
        return embedding


# Poses
#######

def get_part_poses(n_shapes, n_parts, datasource=None, instances=None, rotation=True, translation=True, 
                   scale=True, freeze=False, device=None, fill_nans=False):
    """
    Create and initialize part poses, possibly from existing ones.

    A single pose-embedding will be of dimension (n_parts, R+t).
    """
    poses = PartPose(n_shapes, n_parts, rotation=rotation, translation=translation, 
                     scale=scale, freeze=freeze, device=device)
    if instances is not None and datasource is not None:
        rotation, translation, scale = load_poses(datasource, instances)
        poses.load_existing(rotation=rotation, translation=translation, scale=scale, freeze=freeze, fill_nans=fill_nans)
    return poses

def load_poses(paramdir, instances):
    """Load the poses of the instances from the parameter directory."""
    quaternions, translations, scales = [], [], []
    for instance in instances:
        quaternions.append(np.load(os.path.join(paramdir, instance, "quaternions.npy")))
        translations.append(np.load(os.path.join(paramdir, instance, "translations.npy")))
        scales.append(np.load(os.path.join(paramdir, instance, "scales.npy")))
    quaternions = np.stack(quaternions, axis=0)
    translations = np.stack(translations, axis=0)
    scales = np.stack(scales, axis=0)
    return torch.tensor(quaternions).float(), torch.tensor(translations).float(), torch.tensor(scales).float()

class PartPose(nn.Module):
    """
    Module storing the part poses (rotation, translation) as vectors:
     - rotation: quaternions (4D)
     - translation: 3D vectors

    It will store the pose of all shape and all parts together, but will retrieve
    all part embeddings for a given shape index.
    """
    
    def __init__(self, n_shapes, n_parts, rotation=True, translation=True, scale=True, freeze=False, device=None, dtype=None):
        super().__init__()
        self.n_shapes = n_shapes
        self.n_parts = n_parts
        self.rotation = rotation
        self.translation = translation
        self.scale = scale

        # Prepare the pose vectors
        self.pose_dim = 0
        if rotation:
            self.pose_dim += 4
        if translation:
            self.pose_dim += 3
        if scale:
            self.pose_dim += 3
        assert self.pose_dim > 0, "At least one of rotation, translation or scale must be True"
        self.weight = torch.zeros(n_shapes, n_parts, self.pose_dim, device=device, dtype=dtype)
        if rotation:
            self.weight[..., 0] = 1.  # initialize the quaternions with the identity
        if scale:
            self.weight[..., -3:] = 1.  # initialize the scales to 1
        self.weight = Parameter(self.weight, requires_grad=not freeze)

        # Prepare indices offset for the parts
        self.register_buffer('_part_idx_offset', torch.arange(n_parts, device=device))

    def forward(self, input: Tensor) -> Tensor:
        # Flatten the indices and poses (adding part indices to shape indices)
        input = input.unsqueeze(-1) * self.n_parts + self._part_idx_offset
        weight = self.weight.view(-1, self.pose_dim)

        # Retrieve the "embeddings" with an additional part dimension
        embeddings = F.embedding(input, weight)

        # Return the parameters
        output = []
        offset = 0
        if self.rotation:
            output.append(embeddings[..., offset:offset + 4])
            offset += 4
        if self.translation:
            output.append(embeddings[..., offset:offset + 3])
            offset += 3
        if self.scale:
            output.append(embeddings[..., offset:offset + 3])
            offset += 3
        return output if len(output) > 1 else output[0]

    def extra_repr(self) -> str:
        s = '{n_shapes}, {n_parts}, pose_dim={pose_dim}'
        return s.format(**self.__dict__)
    
    @torch.no_grad()
    def standardize_quaternion(self):
        if self.rotation:
            self.weight[..., :4] = standardize_quaternion(self.weight[..., :4])

    @torch.no_grad()
    def load_existing(self, rotation=None, translation=None, scale=None, freeze=False, fill_nans=False):
        # Fill in existing weights, so need to same rotation/translation existences
        offset = 0
        if self.rotation is not None:
            self.weight[..., offset:offset + 4] = rotation.clone().detach().to(self.weight.device)
            offset += 4
        if self.translation is not None:
            self.weight[..., offset:offset + 3] = translation.clone().detach().to(self.weight.device)
            offset += 3
        if scale is not None:
            self.weight[..., offset:offset + 3] = scale.clone().detach().to(self.weight.device)
            offset += 3
        if fill_nans:
            self.fill_nans_average()
        self.weight.requires_grad = not freeze
    
    @torch.no_grad()
    def fill_nans_average(self):
        """Replace invalid poses by the average valid pose."""
        for p in range(self.n_parts):
            index = torch.isnan(self.weight[:, p]).any(-1)
            self.weight[index, p] = self.weight[~index, p].mean(0)