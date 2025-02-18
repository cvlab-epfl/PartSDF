"""
Module for layers and models that predict the SDF of multiple parts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..activation import get_activation
from ..features import get_input_features, PositionalEncoding
from ..utils import ModulatedLinear, DemodulatedLinear
from ..utils import MLP, LatentMLP
from ...primitives import quaternion_to_matrix, inv_transform_parts


############
# Template #
############

class SDFNetBase(nn.Module):
    """
    Base class for the SDFNet models.
    """

    def __init__(self):
        super().__init__()
        self.n_parts = None  # number of parts
        self.use_occ = False  # whether to use Occupancy
        self.out_softmin = None  # temperature for softmin
        self.out_softmax = None  # temperature for softmax (in case of Occupancy)
        self.features = None  # input features (e.g., positional encoding)
        # self.output_scale = None  # output scaling   # messes up with buffer registration in children classes


    def get_sdf(self, part_lat, xyz_feats):
        """
        Get the SDF of the parts in their canonical space.
        ! Needs to be overwritten !
        
        Args:
            part_lat (torch.Tensor): part latent codes, shape (*, P, L).
            xyz_feats (torch.Tensor): query point features, shape (*, 1/P, D).
                -2 dim can be =1 for repeating the same coordinates for all parts.
        
        Returns:
            torch.Tensor: SDF of the parts, shape (*, P, 1).
        """
        raise NotImplementedError


    def forward(self, part_lat, xyz, R=None, t=None, s=None, return_parts=False, **kwargs):
        """
        Args:
            part_lat (torch.Tensor): part latent codes, shape (*, P, L).
            xyz (torch.Tensor): query points, shape (*, 3).
            R (torch.Tensor): rotation matrices or quaternions, shape (*, P, 3, 3) or (*, P, 4).
            t (torch.Tensor): translation vectors, shape (*, P, 3).
            s (torch.Tensor): scale factors, shape (*, P, 3).
            return_parts (bool): whether to return the SDFs for each part.
        """
        xyz = xyz.unsqueeze(-2)  # (*, 3) -> (*, 1, 3)
        # Transform the input points based on the (optional) rotations and translations
        xyz = inv_transform_parts(xyz, R, t, s)  # -> (*, P, 3) if (R, t, s) exists

        # Compute input features (*, 3) -> (*, D)
        feats = self.features(xyz) if self.features is not None else xyz

        # Compute the SDFs for each part ((*, P, L), (*, 1, D)) -> (*, P, 1)
        sdf = self.get_sdf(part_lat, feats, R=R, t=t, s=s)

        # Apply output scaling
        if self.output_scale is not None:
            sdf = sdf * self.output_scale

        if not return_parts:
            # Apply min or softmin
            sdf = self.combine_part(sdf)

        return sdf
    

    def combine_part(self, sdf, dim=-2):
        """Combine the parts' SDF or Occupancy."""
        if not self.use_occ:
            return self.combine_part_sdf(sdf, dim=dim)
        else:
            return self.combine_part_occ(sdf, dim=dim)


    def combine_part_sdf(self, sdf, dim=-2):
        """Combine the parts' SDF, usually through min()."""
        # Apply min or softmin
        if self.out_softmin is not None and self.training:
            temp = self.out_softmin if isinstance(self.out_softmin, float) else 1.
            sdf = (sdf * F.softmin(sdf / temp, dim=dim)).sum(dim=dim)
        else:
            sdf = sdf.min(dim=dim)[0]
        return sdf


    def combine_part_occ(self, occ, dim=-2):
        """Combine the parts' Occupancy, usually through max()."""
        # Apply max or softmax
        if self.out_softmax is not None and self.training:
            temp = self.out_softmax if isinstance(self.out_softmax, float) else 1.
            occ = (occ * F.softmax(occ / temp, dim=dim)).sum(dim=dim)
        else:
            occ = occ.max(dim=dim)[0]
        return occ


##########
# Models #
##########

class PartSDF(SDFNetBase):
    """A single MLP for all parts, which then belong in the same latent space."""

    def __init__(self, n_parts=16, part_dim=128, input_dim=3, output_dim=1,
                 activation="relu", features=None, use_occ=False,
                 layer_type="modulated", n_layers=4, hidden_dim=512, dropout=0., weight_norm=True,
                 part_bias=False, part_norm=False, part_conv1d=False, input_pose=False,
                 output_scale=None, out_softmin=None,
                 **kwargs):
        """
        Args:
            n_parts (int): number of parts / local latents.
            part_dim (int): dimension of each local latent.
            input_dim (int): dimension of the input.
            output_dim (int): dimension of the output.
            activation (str): type of activation function.
            features (str, optional): type of input features.
            use_occ (bool): whether to use Occupancy instead of SDF.
            layer_type (str): type of layers in ("linear", "modulated", or "demodulated").
            n_layers (int): number of layers.
            hidden_dim (int): dimension of the hidden layers.
            dropout (float): dropout rate.
            weight_norm (bool): whether to apply weight normalization.
            part_bias (bool): whether to learn a bias for each part.
            part_norm (bool): whether to add a conditional layer norm for each part.
            part_conv1d (bool): whether to add 1x1 conv1d between layers for inter-part communication.
            input_pose (bool): whether to concatenate the pose to the input features.
            output_scale (float): scale to apply to the output.
            out_softmin (float>0, optional): if given, temperature for softmin
                during training on the output.
        """
        super().__init__()
        self.n_parts = n_parts
        self.part_dim = part_dim
        self.use_occ = use_occ
        self.input_pose = input_pose
        self.register_buffer("output_scale", torch.tensor(output_scale) if output_scale is not None else None)
        self.out_softmin = out_softmin

        # Input feature transformation
        if features is None or features == "none":
            self.features = None
            feats_dim = input_dim
        else:
            self.features = get_input_features(features)
            feats_dim = self.features.outdim
        if self.input_pose:
            feats_dim += 10
        
        # Main network
        part_bias = n_parts if part_bias else None
        part_norm = n_parts if part_norm else None
        part_conv1d = n_parts if part_conv1d else None
        self.sdfnet = LatentMLP(part_dim, feats_dim, output_dim, layer_type, n_layers, 
                                hidden_dim, activation, dropout, weight_norm,
                                part_bias=part_bias, part_norm=part_norm, n_parts=n_parts,
                                part_conv1d=part_conv1d)


    def get_sdf(self, part_lat, xyz_feats, R=None, t=None, s=None):
        """
        Get the SDF of the parts in their canonical space.
        
        Args:
            part_lat (torch.Tensor): part latent codes, shape (*, P, L).
            xyz_feats (torch.Tensor): query point features, shape (*, 1/P, D).
                -2 dim can be =1 for repeating the same coordinates for all parts.
            R (torch.Tensor): (unused) rotation matrices or quaternions, shape (*, P, 3, 3) or (*, P, 4).
            t (torch.Tensor): (unused) translation vectors, shape (*, P, 3).
            s (torch.Tensor): (unused) scale factors, shape (*, P, 3).
        
        Returns:
            torch.Tensor: SDF of the parts, shape (*, P, 1).
        """
        if self.input_pose:
            if R is None:
                R = torch.tensor([1., 0., 0., 0.], device=xyz_feats.device)
            assert R.shape[-1] == 4, "PartSDF only works with quaternions as input."
            if t is None:
                t = torch.zeros(3, device=xyz_feats.device)
            if s is None:
                s = torch.ones(3, device=xyz_feats.device)
            R = R.expand(xyz_feats.shape[:-1] + (4,))
            t = t.expand(xyz_feats.shape[:-1] + (3,))
            s = s.expand(xyz_feats.shape[:-1] + (3,))
            poses = torch.cat([R, t, s], dim=-1)
            xyz_feats = torch.cat([xyz_feats, poses], dim=-1)
        return self.sdfnet(part_lat, xyz_feats)
