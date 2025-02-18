"""
Module used for defining losses.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss_recon(loss='l1', reduction='mean'):
    """Return the reconstruction loss to apply on the generator's output."""
    if loss is None or loss.lower() == 'none':
        return None
    elif loss.lower() in ['l1', 'mae']:
        return nn.L1Loss(reduction=reduction)
    elif loss.lower() in ['l2', 'mse']:
        return nn.MSELoss(reduction=reduction)
    elif loss.lower() == 'l1-hard':
        return LossHard(loss='l1', reduction=reduction)
    elif loss.lower() == 'l1-hard-linear':
        return LossHard(loss='l1', reduction=reduction, linear_weight=True)
    elif loss.lower() == 'l2-hard':
        return LossHard(loss='l2', reduction=reduction)
    elif loss.lower() == 'l2-hard-linear':
        return LossHard(loss='l2', reduction=reduction, linear_weight=True)
    # Occupancy
    elif loss.lower() in ['occ', 'bce']:
        return nn.BCEWithLogitsLoss(reduction=reduction)
    elif loss.lower().startswith(('occtemp', 'bcetemp')):
        temperature = float(loss.split('-')[-1])
        return BCEWithLogitsTempLoss(temperature, reduction=reduction)
    else:
        raise NotImplementedError(f"Unknown reconstruction loss \"{loss}\".")


class LossHard(nn.Module):
    """L1 loss with hard samples re-weighting, from Duan et al., ECCV 2020."""

    def __init__(self, loss='l1', lambda_=0.5, reduction='mean', linear_weight=False) -> None:
        super().__init__()
        self.reduction = reduction
        self.lambda_ = lambda_
        self.linear_weight = linear_weight
        self.loss_fn = F.l1_loss if loss.lower() in ['l1', 'mae'] else F.mse_loss

    def forward(self, input, target):
        loss = self.loss_fn(input, target, reduction="none")
        if self.linear_weight:
            weights = 1 + self.lambda_ * F.relu(torch.sign(target) * (target - input)).detach() * 100
        else:
            weights = 1 + self.lambda_ * torch.sign(target) * torch.sign(target - input)
        loss = loss * weights

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss


class BCEWithLogitsTempLoss(nn.Module):
    """
    BCEWithLogitsLoss with temperature scaling of predictions.

    NOTE: an equivalent but *better* option is to use the temperature scaling
    at meshing time only, on a model trained with normal BCEWithLogitsLoss.
    Or to mesh the logits directly.
    """

    def __init__(self, temperature=1., reduction='mean') -> None:
        super().__init__()
        self.reduction = reduction
        self.temperature = temperature

    def forward(self, input, target):
        loss = F.binary_cross_entropy_with_logits(input / self.temperature, target, reduction=self.reduction)
        return loss
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(temperature={self.temperature})"


class IntersectionLoss(nn.Module):
    """Loss to reduce the intersection between parts."""

    def __init__(self, delta=0., tau=1., loss_fn=F.l1_loss, reduction='mean', use_occ=False):
        """
        Loss to reduce the intersection between parts.

        For points where more than a single part has SDF<delta,
        these SDFs are pushed toward >=delta, weighted by a softmax.

        Args:
            delta (float): threshold for the SDFs.
            tau (float): temperature for the softmax.
            loss_fn (callable): loss function to call on the SDFs.
                Must take an argument 'reduction'.
            reduction (str, default='mean'): reduction method.
            use_occ (bool, default=False): whether to use occupancy instead of SDF.
        """
        super().__init__()
        self.reduction = reduction
        self.delta = delta
        self.tau = tau
        self.loss_fn = loss_fn
        self.use_occ = use_occ
    
    def forward(self, sdfs):
        """
        Args:
            sdfs (torch.Tensor): SDFs for the parts, of shape (*, P, 1).
                P are the number of parts that we impose the loss on.
                If use_occ is True, the SDFs are assumed to be occupancy logit values.
        """
        sdfs = sdfs.squeeze(-1)
        n_points = sdfs.numel() // sdfs.shape[-1]

        if self.use_occ:
            loss = self.forward_occ(sdfs)
        else:
            loss = self.forward_sdf(sdfs)

        if self.reduction == 'mean':
            return loss.sum() / n_points
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
    
    def forward_sdf(self, sdfs):
        """
        Args:
            sdfs (torch.Tensor): SDFs for the parts, of shape (*, P, 1).
                P are the number of parts that we impose the loss on.
        """
        # Find intersection points
        mask = (sdfs < self.delta).sum(-1, keepdim=True) > 1
        if not mask.any():
            if self.reduction == 'none':
                return torch.zeros_like(sdfs, requires_grad=True)
            else:
                return torch.tensor(0., device=sdfs.device, requires_grad=True)

        # Compute softmax weights on masked SDF
        weights = F.softmax(
            torch.where(sdfs < self.delta, sdfs, float('-inf')) / self.tau, 
        dim=-1).detach()
        # Mask the weights based on intersection points
        weights = torch.nan_to_num(weights) * mask

        # Loss on the weighted sum of SDFs
        loss = self.loss_fn(weights * sdfs, torch.full_like(sdfs, self.delta), reduction='none')
        # Filter actual intersection points with the mask
        # loss = torch.where(mask, loss, 0.)

        return loss
    
    def forward_occ(self, occs):
        """
        Args:
            occs (torch.Tensor): Occupancy logits for the parts, of shape (*, P, 1).
                P are the number of parts that we impose the loss on.
        """
        inverse_delta = np.log((0.5 + self.delta) / (0.5 - self.delta))  # inverse sigmoid

        # Find intersection points
        mask = (occs > inverse_delta).sum(-1, keepdim=True) > 1
        if not mask.any():
            if self.reduction == 'none':
                return torch.zeros_like(occs, requires_grad=True)
            else:
                return torch.tensor(0., device=occs.device, requires_grad=True)

        # Compute softmax weights on masked occupancy
        weights = F.softmax(
            torch.where(occs.sigmoid() > 0.5 + self.delta, -occs, float('-inf')) / self.tau, 
        dim=-1).detach()
        # Mask the weights based on intersection points
        weights = torch.nan_to_num(weights) * mask

        # Weighted loss on the occupancies
        loss = weights * self.loss_fn(occs, torch.full_like(occs, inverse_delta), reduction='none')
        # Filter actual intersection points with the mask
        # loss = torch.where(mask, loss, 0.)

        return loss