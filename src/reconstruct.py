"""
Module for test and inference reconstruction.
"""

import time

import torch
from torch import optim
from torch.nn import functional as F

from .data import remove_nans, samples_from_tensor
from .loss import get_loss_recon, IntersectionLoss
from .primitives import standardize_quaternion
from .utils import clamp_sdf, get_device, index_extract


def reconstruct(model, sdf_data, n_iters, n_samples, lr, loss_fn_recon="l1", 
                latent_reg=None, clampD=None, latent_init=None, latent_size=256,
                n_parts=None, is_part_sdfnet=False, inter_lambda=None, inter_temp=0.02,
                rotations=None, translations=None, scales=None, lr_pose=None, spherical_lats=False,
                max_norm=None, verbose=False, device=get_device(), return_history=False):
    """Reconstruct the shape by optimizing the latent wrt to SDF data."""
    rotations = [rotations] if rotations is not None else None
    translations = [translations] if translations is not None else None
    scales = [scales] if scales is not None else None
    return reconstruct_batch(
        model, [sdf_data], n_iters, n_samples, lr, loss_fn_recon=loss_fn_recon, 
        latent_reg=latent_reg, clampD=clampD, latent_init=latent_init, latent_size=latent_size,
        n_parts=n_parts, is_part_sdfnet=is_part_sdfnet, inter_lambda=inter_lambda, inter_temp=inter_temp,
        rotations=rotations, translations=translations, scales=scales, lr_pose=lr_pose, spherical_lats=spherical_lats,
        max_norm=max_norm, verbose=verbose, device=device, return_history=return_history
    )

def reconstruct_batch(model, sdf_data, n_iters, n_samples, lr, loss_fn_recon="l1", 
                      latent_reg=None, clampD=None, latent_init=None, latent_size=256,
                      n_parts=None, is_part_sdfnet=False, inter_lambda=None, inter_temp=0.02,
                      rotations=None, translations=None, scales=None, lr_pose=None, spherical_lats=False,
                      max_norm=None, verbose=False, device=get_device(), return_history=False):
    """Reconstruct the batch of shapes by optimizing their latents wrt to SDF data."""
    if verbose:
        start_time = time.time()
    use_occ = hasattr(model, 'use_occ') and model.use_occ  # whether to use occupancy instead of SDF

    # Data
    n_shapes = len(sdf_data)
    sdf_pos = [remove_nans(sdf['pos']) for sdf in sdf_data]
    sdf_neg = [remove_nans(sdf['neg']) for sdf in sdf_data]
    sdf_pos = [torch.from_numpy(sdf).float().to(device) for sdf in sdf_pos]
    sdf_neg = [torch.from_numpy(sdf).float().to(device) for sdf in sdf_neg]

    # Initialize the latent
    if is_part_sdfnet and n_parts is not None:
        if latent_init is None:
            latent = torch.ones(n_shapes, n_parts, latent_size).normal_(0, 0.01).to(device)
        elif isinstance(latent_init, float):
            latent = torch.ones(n_shapes, n_parts, latent_size).normal_(0, latent_init).to(device)
        elif isinstance(latent_init, torch.Tensor):
            latent = latent_init.clone().detach()
        latent = latent.view(n_shapes, n_parts, -1)
        # and the poses (if applicable)
        if rotations is not None:
            rotations = torch.stack([rot.float().to(device) for rot in rotations]).view(n_shapes, n_parts, -1)
            rotations.requires_grad_(True)
        if translations is not None:
            translations = torch.stack([t.float().to(device) for t in translations]).view(n_shapes, n_parts, -1)
            translations.requires_grad_(True)
        if scales is not None:
            scales = torch.stack([s.float().to(device) for s in scales]).view(n_shapes, n_parts, -1)
            scales.requires_grad_(True)
    else:
        if latent_init is None:
            latent = torch.ones(n_shapes, latent_size).normal_(0, 0.01).to(device)
        elif isinstance(latent_init, float):
            latent = torch.ones(n_shapes, latent_size).normal_(0, latent_init).to(device)
        elif isinstance(latent_init, torch.Tensor):
            latent = latent_init.clone().detach()
        latent = latent.view(n_shapes, -1)
    if spherical_lats:
        latent /= latent.norm(dim=-1, keepdim=True)
    latent.requires_grad_(True)

    # Optimizer and scheduler
    if isinstance(loss_fn_recon, str):
        loss_fn_recon = get_loss_recon(loss_fn_recon, 'none').to(device)
    if inter_lambda is not None:
        _loss_fn = F.binary_cross_entropy_with_logits if use_occ else F.l1_loss
        loss_intersection = IntersectionLoss(delta=0., tau=inter_temp, loss_fn=_loss_fn, reduction='mean', use_occ=use_occ).to(device)
    optimizer = optim.Adam([latent], lr=lr)
    lr_pose = lr if lr_pose is None else lr_pose
    if rotations is not None:
        optimizer.add_param_group({"params": [rotations], "lr": lr_pose})
    if translations is not None:
        optimizer.add_param_group({"params": [translations], "lr": lr_pose})
    if scales is not None:
        optimizer.add_param_group({"params": [scales], "lr": lr_pose})
    scheduler = optim.lr_scheduler.StepLR(optimizer, n_iters//2, 0.1)

    # Model in evaluation mode and frozen
    model.eval()
    p_state = []
    if return_history:
        history = {"loss": [], "latent": [], "rotations": [], "translations": [], "scales": []}
    for p in model.parameters():
        p_state.append(p.requires_grad)
        p.requires_grad = False
    
    for _ in range(n_iters):
        # Sample SDF
        xyz, sdf_gt = [], []
        for pos, neg in zip (sdf_pos, sdf_neg):
            _xyz, _sdf_gt = samples_from_tensor(pos, neg, n_samples)
            xyz.append(_xyz)
            sdf_gt.append(_sdf_gt)
        xyz, sdf_gt = torch.stack(xyz), torch.stack(sdf_gt)

        # Forward pass
        _R = rotations.unsqueeze(1) if rotations is not None else None
        _t = translations.unsqueeze(1) if translations is not None else None
        _s = scales.unsqueeze(1) if scales is not None else None
        if inter_lambda is not None:
            preds = model(latent.unsqueeze(1), xyz, R=_R, t=_t, s=_s, return_parts=True)
        else:
            preds = model(latent.unsqueeze(1), xyz, R=_R, t=_t, s=_s)
        preds_noclamp = preds
        if use_occ:
            sdf_gt = (sdf_gt <= 0).float()
        elif clampD is not None and clampD > 0:
            if inter_lambda is not None:
                preds = clamp_sdf(preds, clampD, ref=sdf_gt.unsqueeze(-2).expand_as(preds))
            else:
                preds = clamp_sdf(preds, clampD)
            sdf_gt = clamp_sdf(sdf_gt, clampD)

        # Full shape reconstruction loss
        if inter_lambda is not None:
            loss = loss_fn_recon(model.combine_part(preds), sdf_gt).mean()
        else:
            loss = loss_fn_recon(preds, sdf_gt).mean()
        # Latent regularization
        if latent_reg is not None:
            loss = loss + latent_reg * latent.square().sum()
        # Part intersection loss
        if inter_lambda is not None:
            loss = loss + inter_lambda * loss_intersection(preds_noclamp)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            # Latent max norm
            if max_norm is not None:
                    latent_norm = latent.norm(dim=-1, keepdim=True)
                    latent *= latent_norm.clamp(max=max_norm) / latent_norm
            if spherical_lats:
                latent /= latent.norm(dim=-1, keepdim=True)
        
            # Rotation quaternion
            if rotations is not None:
                rotations = standardize_quaternion(rotations)
    
        if return_history:
            history['loss'].append(loss.detach().cpu().item())
            history['latent'].append(latent.detach().cpu().clone())
            if rotations is not None:
                history['rotations'].append(rotations.detach().cpu().clone())
            if translations is not None:
                history['translations'].append(translations.detach().cpu().clone())
            if scales is not None:
                history['scales'].append(scales.detach().cpu().clone())

    # Restore model's parameter state
    for p, state in zip(model.parameters(), p_state):
        p.requires_grad = state
    
    if verbose:
        print(f"reconstruction took {time.time() - start_time:.3f}s.") 
    if return_history:
        out = history['loss'], torch.stack(history['latent'], dim=0)
        if rotations is not None:
            out += (torch.stack(history['rotations'], dim=0),)
        if translations is not None:
            out += (torch.stack(history['translations'], dim=0),)
        if scales is not None:
            out += (torch.stack(history['scales'], dim=0),)
    else:
        out = loss.detach().cpu().numpy(), latent.detach().clone()
        if rotations is not None:
            out += (rotations.detach().clone(),)
        if translations is not None:
            out += (translations.detach().clone(),)
        if scales is not None:
            out += (scales.detach().clone(),)
    return out


#########
# Parts #
#########

def reconstruct_parts(model, sdf_data, label_data, n_iters, n_samples, lr, loss_fn_recon="l1-hard", 
                      recon_lambda=0.5, loss_fn_parts="l1-hard", parts_lambda=1.0,
                      latent_reg=None, clampD=None, latent_init=None, latent_size=256,
                      n_parts=None, is_part_sdfnet=False, inter_lambda=None, inter_temp=0.02,
                      rotations=None, translations=None, scales=None, spherical_lats=False,
                      max_norm=None, verbose=False, device=get_device(), return_history=False):
    """Reconstruct the shape by optimizing the latent wrt to SDF data with part supervision."""
    rotations = [rotations] if rotations is not None else None
    translations = [translations] if translations is not None else None
    scales = [scales] if scales is not None else None
    return reconstruct_parts_batch(
        model, [sdf_data], [label_data], n_iters, n_samples, lr, loss_fn_recon=loss_fn_recon,
        recon_lambda=recon_lambda, loss_fn_parts=loss_fn_parts, parts_lambda=parts_lambda, 
        latent_reg=latent_reg, clampD=clampD, latent_init=latent_init, latent_size=latent_size,
        n_parts=n_parts, is_part_sdfnet=is_part_sdfnet, inter_lambda=inter_lambda, inter_temp=inter_temp,
        rotations=rotations, translations=translations, scales=scales, spherical_lats=spherical_lats,
        max_norm=max_norm, verbose=verbose, device=device, return_history=return_history
    )

def reconstruct_parts_batch(model, sdf_data, label_data, n_iters, n_samples, lr, loss_fn_recon="l1-hard", 
                            recon_lambda=0.5, loss_fn_parts="l1-hard", parts_lambda=1.0,
                            latent_reg=None, clampD=None, latent_init=None, latent_size=256,
                            n_parts=None, is_part_sdfnet=False, inter_lambda=None, inter_temp=0.02,
                            rotations=None, translations=None, scales=None, spherical_lats=False,
                            max_norm=None, verbose=False, device=get_device(), return_history=False):
    """Reconstruct the batch of shapes by optimizing their latents wrt to SDF data with part supervision."""
    if verbose:
        start_time = time.time()
    use_occ = hasattr(model, 'use_occ') and model.use_occ  # whether to use occupancy instead of SDF

    # Data
    n_shapes = len(sdf_data)
    # sdf_pos = [remove_nans(sdf['pos']) for sdf in sdf_data]
    # sdf_neg = [remove_nans(sdf['neg']) for sdf in sdf_data]
    sdf_pos = [torch.from_numpy(sdf['pos']).float().to(device) for sdf in sdf_data]
    sdf_neg = [torch.from_numpy(sdf['neg']).float().to(device) for sdf in sdf_data]
    label_pos = [torch.from_numpy(label['pos']).float().to(device) for label in label_data]
    label_neg = [torch.from_numpy(label['neg']).float().to(device) for label in label_data]
    data_pos = [torch.cat([pos, l_pos[:, None]], dim=-1) for pos, l_pos in zip(sdf_pos, label_pos)]
    data_neg = [torch.cat([neg, l_neg[:, None]], dim=-1) for neg, l_neg in zip(sdf_neg, label_neg)]

    # Initialize the latent
    if is_part_sdfnet and n_parts is not None:
        if latent_init is None:
            latent = torch.ones(n_shapes, n_parts, latent_size).normal_(0, 0.01).to(device)
        elif isinstance(latent_init, float):
            latent = torch.ones(n_shapes, n_parts, latent_size).normal_(0, latent_init).to(device)
        elif isinstance(latent_init, torch.Tensor):
            latent = latent_init.clone().detach()
        latent = latent.view(n_shapes, n_parts, -1)
        # and the poses (if applicable)
        if rotations is not None:
            rotations = torch.stack([rot.float().to(device) for rot in rotations]).view(n_shapes, n_parts, -1)
        if translations is not None:
            translations = torch.stack([t.float().to(device) for t in translations]).view(n_shapes, n_parts, -1)
        if scales is not None:
            scales = torch.stack([s.float().to(device) for s in scales]).view(n_shapes, n_parts, -1)
    else:
        if latent_init is None:
            latent = torch.ones(n_shapes, latent_size).normal_(0, 0.01).to(device)
        elif isinstance(latent_init, float):
            latent = torch.ones(n_shapes, latent_size).normal_(0, latent_init).to(device)
        elif isinstance(latent_init, torch.Tensor):
            latent = latent_init.clone().detach()
        latent = latent.view(n_shapes, -1)
    if spherical_lats:
        latent /= latent.norm(dim=-1, keepdim=True)
    latent.requires_grad_(True)

    # Optimizer and scheduler
    if isinstance(loss_fn_recon, str):
        loss_fn_recon = get_loss_recon(loss_fn_recon, 'none').to(device)
    if loss_fn_parts is None:
        loss_fn_parts = loss_fn_recon
    elif isinstance(loss_fn_parts, str):
        loss_fn_parts = get_loss_recon(loss_fn_parts, 'none').to(device)
    if inter_lambda is not None:
        _loss_fn = F.binary_cross_entropy_with_logits if use_occ else F.l1_loss
        loss_intersection = IntersectionLoss(delta=0., tau=inter_temp, loss_fn=_loss_fn, reduction='mean', use_occ=use_occ).to(device)
    optimizer = optim.Adam([latent], lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, n_iters//2, 0.1)

    # Model in evaluation mode and frozen
    model.eval()
    p_state = []
    if return_history:
        history = {"loss": [], "latent": [], "rotations": [], "translations": [], "scales": []}
    for p in model.parameters():
        p_state.append(p.requires_grad)
        p.requires_grad = False
    
    for _ in range(n_iters):
        # Sample SDF
        xyz, sdf_gt, part_labels = [], [], []
        for pos, neg in zip(data_pos, data_neg):
            samples = samples_from_tensor(pos, neg, n_samples, full_samples=True)
            xyz.append(samples[..., :3].float())
            sdf_gt.append(samples[..., 3:4].float())
            part_labels.append(samples[..., 4:5].int())
        xyz, sdf_gt, part_labels = torch.stack(xyz), torch.stack(sdf_gt), torch.stack(part_labels)

        # Forward pass
        _R = rotations.unsqueeze(1) if rotations is not None else None
        _t = translations.unsqueeze(1) if translations is not None else None
        _s = scales.unsqueeze(1) if scales is not None else None
        preds = model(latent.unsqueeze(1), xyz, R=_R, t=_t, s=_s, return_parts=True)
        preds_noclamp = preds
        if use_occ:
            sdf_gt = (sdf_gt <= 0).float()
        elif clampD is not None and clampD > 0:
            preds = clamp_sdf(preds, clampD, ref=sdf_gt.unsqueeze(-2).expand_as(preds))
            sdf_gt = clamp_sdf(sdf_gt, clampD)

        # Full shape reconstruction loss
        loss = loss_fn_recon(model.combine_part(preds), sdf_gt).mean()
        loss = recon_lambda * loss * (_ > n_iters//2)
        # Part reconstruction loss
        sdf_part = index_extract(preds, part_labels)  # BxNx1
        part_loss = loss_fn_parts(sdf_part, sdf_gt).mean()
        loss = loss + parts_lambda * part_loss
        # Latent regularization
        if latent_reg is not None:
            loss = loss + latent_reg * latent.square().sum()
        # Part intersection loss
        if inter_lambda is not None:
            loss = loss + inter_lambda * loss_intersection(preds_noclamp)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Latent max norm
        with torch.no_grad():
            if max_norm is not None:
                latent_norm = latent.norm(dim=-1, keepdim=True)
                latent *= latent_norm.clamp(max=max_norm) / latent_norm
            if spherical_lats:
                latent /= latent.norm(dim=-1, keepdim=True)
    
        if return_history:
            history['loss'].append(loss.detach().cpu().item())
            history['latent'].append(latent.detach().cpu().clone())
            if rotations is not None:
                history['rotations'].append(rotations.detach().cpu().clone())
            if translations is not None:
                history['translations'].append(translations.detach().cpu().clone())
            if scales is not None:
                history['scales'].append(scales.detach().cpu().clone())

    # Restore model's parameter state
    for p, state in zip(model.parameters(), p_state):
        p.requires_grad = state
    
    if verbose:
        print(f"reconstruction took {time.time() - start_time:.3f}s.") 
    if return_history:
        out = history['loss'], torch.stack(history['latent'], dim=0)
        if rotations is not None:
            out += (torch.stack(history['rotations'], dim=0),)
        if translations is not None:
            out += (torch.stack(history['translations'], dim=0),)
        if scales is not None:
            out += (torch.stack(history['scales'], dim=0),)
    else:
        out = loss.detach().cpu().numpy(), latent.detach().clone()
        if rotations is not None:
            out += (rotations.detach().clone(),)
        if translations is not None:
            out += (translations.detach().clone(),)
        if scales is not None:
            out += (scales.detach().clone(),)
    return out