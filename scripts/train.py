"""
Main training script for PartSDF.

Train INR models that predict parts SDFs given parts latents and poses.
"""

import os, os.path
import sys
import argparse
import logging
from multiprocessing import cpu_count
import time
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import workspace as ws
from src.data import PartSdfDataset
from src.loss import get_loss_recon, IntersectionLoss
from src.mesh import create_mesh, create_parts, SdfGridFiller
from src.metric import chamfer_distance
from src.model import get_model, get_part_latents, get_part_poses
from src.optimizer import get_optimizer, get_scheduler
from src.primitives import standardize_quaternion
from src.reconstruct import reconstruct_batch, reconstruct
from src.utils import (configure_logging, set_seed, get_device, clamp_sdf, get_gradient, 
                       index_extract, get_color_parts)
from src import visualization as viz


def parser(argv=None):
    """Parse the arguments."""
    parser = argparse.ArgumentParser(description="Train a deep implicit representation network with part supervision.")

    parser.add_argument("experiment", help="path to the experiment directory. If existing, will try to resume it (see --no-resume)")

    parser.add_argument('--debug', action='store_true', help="increase verbosity to print debugging messages")
    parser.add_argument('--load-epoch', type=int, default=None, help="specific epoch to resume from (will throw an error if not possible)")
    parser.add_argument('--no-resume', action='store_true', help="do not resume the experiment if existing and start from epoch 0")
    parser.add_argument('--no-test', action='store_true', help="do not perform final test reconstruction")
    parser.add_argument('-q', '--quiet', dest="verbose", action='store_false',  help="disable verbosity and run in quiet mode")
    parser.add_argument('--reset', action='store_true', help="force a reset of the experimental directory")
    parser.add_argument('--seed', type=int, default=0, help="initial seed for the RNGs (default=0)")
    parser.add_argument('--workers', type=int, default=16, help="number of worker subprocesses preparing batches (use 0 to load in the main process)")

    args = parser.parse_args(argv)

    return args


@torch.no_grad()
def to_spherical(latents):
    """Normalize the latents to the unit sphere."""
    latents.weight.data = torch.nn.functional.normalize(latents.weight.data, dim=-1)


def main(args=None):
    # Initialization
    if args is None:
        args = parser()
    args.workers = min(args.workers, cpu_count())
    set_seed(args.seed)
    start_time = time.time()

    expdir = args.experiment
    specs = ws.load_specs(expdir)
    device = specs.get("Device", get_device())
    if args.reset:
        ws.reset_experiment_dir(expdir)
    else:
        ws.build_experiment_dir(expdir)
    configure_logging(args, os.path.join(ws.get_log_dir(expdir), "trainlog.txt"))

    logging.info(f"Command:  python {' '.join(sys.argv)}")
    logging.info(f"Date: " + time.strftime("%d %B %Y at %H:%M:%S (%Y%m%d-%H%M%S)"))
    logging.info(f"Running experiment in {expdir}. (on {device})")
    logging.info(f"arguments = {args}")
    
    # Data
    use_poses = specs["Parts"].get("UsePoses", False)
    paramdir = specs["Parts"].get("ParametersDir", None) if use_poses else None
    dataset = PartSdfDataset(specs["DataSource"], specs["TrainSplit"], specs["SamplesPerScene"], 
                             specs["SamplesDir"], specs["SamplesFile"], specs["Parts"]["Type"], 
                             specs["Parts"]["SamplesDir"], specs["Parts"].get("SamplesFile", None))
    dataloader = DataLoader(dataset, batch_size=specs["ScenesPerBatch"], shuffle=True, num_workers=args.workers, pin_memory=True)
    len_dataset = len(dataset)
    # Validation data
    valid_frequency = specs.get("ValidFrequency", None)
    valid_split = specs.get("ValidSplit", specs["TestSplit"])
    if valid_frequency is not None and valid_split is not None:
        with open(valid_split) as f:
            valid_split = json.load(f)
    else:
        valid_frequency, valid_split = None, None

    logging.info(f"{len_dataset} shapes in training dataset.")
    if valid_frequency is not None:
        logging.info(f"{len(valid_split)} shapes in validation dataset.")
    logging.info(f"{args.workers} worker processes created.")

    # Model and latent vectors
    # latent_dim = specs["LatentDim"]
    n_parts = specs["Parts"]["NumParts"]
    part_dim = specs["Parts"]["LatentDim"]
    use_occ = True if specs.get("ImplicitField", "SDF").lower() in ["occ", "occupancy"] else False
    model = get_model(
        specs.get("Network", "DeepSDF"),
        **specs.get("NetworkSpecs", {}),
        n_parts=n_parts,
        part_dim=part_dim,
        use_occ=use_occ
    ).to(device)
    
    spherical_lats = specs.get("LatentSpherical", False)
    latents = get_part_latents(len(dataset), n_parts, part_dim, specs.get("LatentBound", None), device=device, spherical=spherical_lats)
    if use_poses:
        # Load the primitives poses and freeze them
        poses = get_part_poses(len(dataset), n_parts, os.path.join(specs["DataSource"], paramdir), 
                               dataset.instances, freeze=True, device=device, fill_nans=True)
        ws.save_poses(expdir, poses)
        # Get average pose for the reconstruction init
        pose_init = poses.weight.detach().clone().mean(0, keepdim=True)
        rotations_init = standardize_quaternion(pose_init[..., :4])
        translations_init = pose_init[..., 4:7]
        scales_init = pose_init[..., 7:10]

    # If using pre-trained network and latents, load them (note: will get overwritten by existing checkpoints!)
    model_pretrain = specs.get("NetworkPretrained", None)
    if model_pretrain is not None:
        model.load_state_dict(torch.load(model_pretrain))
    latent_pretrain = specs.get("LatentPretrained", None)
    if latent_pretrain is not None:
        latents.load_state_dict(torch.load(latent_pretrain))

    logging.info(f"Model has {sum([x.nelement() for x in model.parameters()]):,} parameters." + \
                 (" (pretrained)" if model_pretrain is not None else ""))
    logging.info(f"{latents.num_embeddings}x{latents.n_parts} latent vectors of size {latents.embedding_dim}." + \
                 (" (pretrained)" if latent_pretrain is not None else ""))
    if use_poses:
        logging.info(f"Using part poses (pre-computed).")

    # Loss and optimizer
    loss_recon = get_loss_recon(specs["ReconLoss"], reduction='none').to(device)
    recon_lambda = specs["ReconLossLambda"]
    part_loss_recon = get_loss_recon(specs["Parts"]["ReconLoss"], reduction='none').to(device)
    part_recon_lambda = specs["Parts"]["ReconLossLambda"]
    latent_reg = specs["LatentRegLambda"]
    eikonal_lambda = specs.get("EikonalLossLambda", None)
    intersection_lambda = specs["Parts"].get("IntersectionLambda", None)
    intersection_temp = specs["Parts"].get("IntersectionTemp", 1.)
    if intersection_lambda is not None:
        _loss_fn = F.binary_cross_entropy_with_logits if use_occ else F.l1_loss
        loss_intersection = IntersectionLoss(delta=0., tau=intersection_temp, loss_fn=_loss_fn, reduction='mean', use_occ=use_occ).to(device)
    part_latent_reg = specs["Parts"].get("LatentRegLambda", None)
    
    optimizer = get_optimizer([model, latents], type=specs["Optimizer"].pop("Type"),
                              lrs=specs["Optimizer"].pop("LearningRates"),
                              kwargs=specs["Optimizer"])
    scheduler = get_scheduler(optimizer, **specs["LearningRateSchedule"])

    # Resume from checkpoint
    history = {'epoch': 0}
    if not args.no_resume:
        if args.load_epoch is None and os.path.isfile(ws.get_checkpoint_filename(expdir)):
            checkpoint = ws.load_checkpoint(expdir)
            ws.use_checkpoint(checkpoint, model, latents, optimizer, scheduler)
            history = ws.load_history(expdir, checkpoint['epoch'])
            del checkpoint

        elif args.load_epoch is not None and os.path.isfile(os.path.join(expdir, ws.HISTORY_FILE)):
            ws.load_experiment(expdir, args.load_epoch, model, latents, optimizer, scheduler)
            history = ws.load_history(expdir, args.load_epoch)

        if history['epoch'] > 0:
            logging.info(f"Loaded checkpoint from epoch={history['epoch']}.")
    
    # Prepare checkpointing and logging
    log_frequency = specs.get("LogFrequency", 10)
    snapshot_epochs = set(range(
        specs["SnapshotFrequency"],
        specs["NumEpochs"] + 1,
        specs["SnapshotFrequency"]
    ))
    for cp in specs["AdditionalSnapshots"]:
        snapshot_epochs.add(cp)
    render_frequency = specs.get("RenderFrequency", None)

    # Training parameters
    n_epochs = specs['NumEpochs']
    clampD = specs["ClampingDistance"]

    # Training
    loss_names = ['loss', 'loss_part']
    if valid_frequency is not None:
        loss_names += ['loss_val']
    if intersection_lambda is not None and intersection_lambda > 0.:
        loss_names += ['loss_inter']
    if eikonal_lambda is not None and eikonal_lambda > 0.:
        loss_names += ['loss_eik']
    if part_latent_reg is not None and part_latent_reg > 0.:
        loss_names += ['loss_reg_part']
    for key in loss_names + ['lr', 'lr_lat', 'lat_norm']:
        if key not in history:
            history[key] = []
    for epoch in range(history['epoch']+1, n_epochs+1):
        time_epoch = time.time()
        running_losses = {name: 0. for name in loss_names if not name.endswith('val')}
        model.train()
        optimizer.zero_grad()

        for i, batch in enumerate(dataloader):
            indices, xyz, sdf_gt, part_labels = batch[0:4]
            batch_size = xyz.shape[0]
            xyz = xyz.to(device).requires_grad_(eikonal_lambda is not None and eikonal_lambda > 0.)  # BxNx3
            sdf_gt = sdf_gt.to(device)  # BxNx1
            part_labels = part_labels.to(device)  # BxNx1
            indices = indices.to(device).unsqueeze(-1)  # Bx1
            batch_latents = latents(indices)  # Bx1xPxL

            if use_poses:
                quaternions, translations, scales = poses(indices)  # Bx1xPx4, Bx1xPx3, Bx1xPx3
                sdf_pred = model(batch_latents, xyz, R=quaternions, t=translations, s=scales, return_parts=True)  # BxNxPx1
            else:
                sdf_pred = model(batch_latents, xyz, return_parts=True)  # BxNxPx1
            sdf_pred_noclamp = sdf_pred
            if use_occ:
                sdf_gt = (sdf_gt <= 0.).float()  # BxNx1
            elif clampD is not None and clampD > 0.:
                sdf_pred = clamp_sdf(sdf_pred, clampD, ref=sdf_gt.unsqueeze(-2).expand_as(sdf_pred))
                sdf_gt = clamp_sdf(sdf_gt, clampD)
            
            # Full shape reconstruction loss
            loss = loss_recon(model.combine_part(sdf_pred), sdf_gt).mean()
            running_losses['loss'] += loss.detach() * batch_size
            loss = recon_lambda * loss
            # Part reconstruction loss
            sdf_part = index_extract(sdf_pred, part_labels)  # BxNx1
            part_loss = part_loss_recon(sdf_part, sdf_gt).mean()
            loss = loss + part_recon_lambda * part_loss
            running_losses['loss_part'] += part_loss.detach() * batch_size
            # Part intersection loss
            if intersection_lambda is not None and intersection_lambda > 0.:
                inter_loss = loss_intersection(sdf_pred_noclamp)
                loss = loss + intersection_lambda * inter_loss
                running_losses['loss_inter'] += inter_loss.detach() * batch_size
            # Eikonal loss
            if eikonal_lambda is not None and eikonal_lambda > 0.:
                grads = get_gradient(xyz, sdf_pred_noclamp)
                loss_eikonal = (grads.norm(dim=-1) - 1.).square().mean()
                loss = loss + eikonal_lambda * loss_eikonal
                running_losses['loss_eik'] += loss_eikonal.detach() * batch_size
            # Part latents regularization
            if part_latent_reg is not None and part_latent_reg > 0.:
                loss_reg_p = min(1, epoch / 100) * batch_latents.square().sum()
                loss = loss + part_latent_reg * loss_reg_p
                running_losses['loss_reg_part'] += loss_reg_p.detach() / batch_size / n_parts
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if spherical_lats:
                to_spherical(latents)
        
        # Validation
        if valid_frequency is not None and epoch % valid_frequency == 0:
            valid_metrics = {'loss': [], 'CD': []}
            valid_meshes, valid_parts = [], []
            grid_filler = SdfGridFiller(256, device=device) if not use_occ else None
            model.eval()
            # Reconstruct and evaluate each validation shape
            for i, instance in enumerate(valid_split):
                # Load sdf data
                filename = os.path.join(specs["DataSource"], specs["SamplesDir"], instance, specs["SamplesFile"])
                npz = np.load(filename)
                if use_poses:
                    rotations, translations, scales = rotations_init.clone(), translations_init.clone(), scales_init.clone()
                else:
                    rotations, translations, scales = None, None, None
                # Optimize latent and reconstruct the mesh
                out = reconstruct(model, npz, 800, 8000, 5e-3, loss_recon, latent_reg=part_latent_reg, clampD=clampD, 
                                  latent_size=part_dim, n_parts=n_parts, is_part_sdfnet=True, spherical_lats=spherical_lats,
                                  inter_lambda=intersection_lambda, inter_temp=intersection_temp,
                                  rotations=rotations, translations=translations, scales=scales, verbose=False, device=device)
                err, latent = out[0], out[1]
                if use_poses:
                    rotations, translations, scales = out[2], out[3], out[4]
                valid_metrics['loss'].append(err)
                mesh = create_mesh(model, latent, grid_filler=grid_filler, R=rotations, t=translations, s=scales)
                parts = create_parts(model, latent, grid_filler=grid_filler, R=rotations, t=translations, s=scales)
                # Save mesh and metrics
                if i < 8:
                    valid_meshes.append(mesh)
                    valid_parts.append(trimesh.util.concatenate(get_color_parts(parts)))
                if mesh.is_empty:
                    continue
                # Chamfer-distance
                gt_samples = np.load(os.path.join(specs["DataSource"], specs["SamplesDir"], instance, 'surface.npy'))[:, :3]
                valid_metrics['CD'].append(chamfer_distance(np.random.permutation(gt_samples)[:30000, :3], mesh.sample(30000)))
            for k in valid_metrics:
                valid_metrics[k] = np.mean(valid_metrics[k]) if len(valid_metrics[k]) > 0 else -1
            del grid_filler
            optimizer.zero_grad()  # remove gradients that were computed during validation
        
        history['epoch'] += 1
        for name in running_losses:
            history[name].append(running_losses[name].item() / len_dataset)
        history["lr"].append(optimizer.state_dict()["param_groups"][0]["lr"])
        history["lr_lat"].append(optimizer.state_dict()["param_groups"][1]["lr"])
        lat_norms = torch.norm(latents.weight.data.detach(), dim=1).cpu()
        history["lat_norm"].append(lat_norms.mean())
        if valid_frequency is not None and epoch % valid_frequency == 0:
            history["loss_val"].append(valid_metrics['loss'])

        # Apply lr-schedule
        if scheduler is not None:
            scheduler.step()
        
        # Renders, snapshot, log and checkpoint
        if render_frequency is not None and epoch % render_frequency == 0:
            idx = torch.cat([torch.arange(8)[:latents.num_embeddings],  # 8 first training shapes
                             torch.randperm(max(0, latents.num_embeddings - 8))[:8] + 8])  # 8 random training shapes
            render_lats = latents(idx.to(device))
            if use_poses:
                quats, trans, scales = poses(idx.to(device))
                meshes = [create_mesh(model, lat, grid_filler=True, R=R, t=t, s=s) 
                          for lat, R, t, s in zip(render_lats, quats, trans, scales)]
                parts = [create_parts(model, lat, grid_filler=True, R=R, t=t, s=s) 
                         for lat, R, t, s in zip(render_lats, quats, trans, scales)]
            else:
                meshes = [create_mesh(model, lat, grid_filler=True) for lat in render_lats]
                parts = [create_parts(model, lat, grid_filler=True) for lat in render_lats]
            parts = [trimesh.util.concatenate(get_color_parts(part)) for part in parts]
            renders = viz.render_meshes(meshes, size=224, aa_factor=2)
            renders_p = viz.render_meshes(parts, size=224, use_texture=True, aa_factor=2)
            ws.save_renders(expdir, renders, epoch)
            # Validation renders
            if valid_frequency is not None and epoch % valid_frequency == 0:
                renders = viz.render_meshes(valid_meshes, size=224, aa_factor=2)
                renders_p = viz.render_meshes(valid_parts, size=224, use_texture=True, aa_factor=2)
                ws.save_renders(expdir, renders, f"valid_{epoch}")
        if epoch in snapshot_epochs:
            ws.save_experiment(expdir, epoch, model, latents, optimizer, scheduler)
        if epoch % log_frequency == 0:
            ws.save_history(expdir, history)
            checkpoint = ws.build_checkpoint(epoch, model, latents, optimizer, scheduler)
            ws.save_checkpoint(expdir, checkpoint)
            del checkpoint
        
        msg = f"Epoch {epoch}/{n_epochs}:"
        for name in running_losses:
            msg += f"{name}={history[name][-1]:.6f} - "
        if valid_frequency is not None and epoch % valid_frequency == 0:
            for k in valid_metrics:
                msg += f"{k}-val={valid_metrics[k]:.6f} - "
        msg = msg[:-3] + f" ({time.time() - time_epoch:.1f}s/epoch)"
        logging.info(msg)

    # End of training
    last_epoch = history['epoch']
    checkpoint = ws.build_checkpoint(last_epoch, model, latents, optimizer, scheduler)
    ws.save_checkpoint(expdir, checkpoint)
    ws.save_history(expdir, history)
    ws.save_experiment(expdir, last_epoch, model, latents, optimizer, scheduler)
    torch.cuda.empty_cache()  # release unused GPU memory
    
    # Final test reconstruction
    if not args.no_test:
        model.eval()

        for sname, split in zip(["valid", "test"], [specs["ValidSplit"], specs["TestSplit"]]):
            # Load the data
            if split is None:
                continue
            with open(split) as f:
                test_instances = json.load(f)
            # First 8 + random 8 test shapes
            idx = list(range(8))[:len(test_instances)] + (torch.randperm(max(0, len(test_instances) - 8))[:8] + 8).tolist()
            test_instances = [test_instances[i] for i in idx]
            npz = []
            for instance in test_instances:
                filename = os.path.join(specs["DataSource"], specs["SamplesDir"], instance, specs["SamplesFile"])
                npz.append(np.load(filename))
            if use_poses:
                rotations = [rotations_init.clone() for _ in range(len(test_instances))]
                translations = [translations_init.clone() for _ in range(len(test_instances))]
                scales = [scales_init.clone() for _ in range(len(test_instances))]
            else:
                rotations, translations, scales = None, None, None
            
            # Reconstruct the shapes (optimize the latents)
            time_test = time.time()
            out = []
            for i in range(len(npz)):  # loop to avoid memory issues on batched reconstruction
                R, t, s = (rotations[i], translations[i], scales[i]) if use_poses else (None, None, None)
                out.append(reconstruct(model, npz[i], 800, 8000, 5e-3, loss_recon, latent_reg=part_latent_reg, clampD=clampD,
                                       latent_size=part_dim, n_parts=n_parts, is_part_sdfnet=True, spherical_lats=spherical_lats,
                                       inter_lambda=intersection_lambda, inter_temp=intersection_temp, 
                                       rotations=R, translations=t, scales=s, verbose=False, device=device))
            err = np.mean([out[i][0] for i in range(len(out))])
            latent = torch.cat([out[i][1] for i in range(len(out))], dim=0)
            if use_poses:
                rotations = torch.cat([out[i][2] for i in range(len(out))], dim=0)
                translations = torch.cat([out[i][3] for i in range(len(out))], dim=0)
                scales = torch.cat([out[i][4] for i in range(len(out))], dim=0)
            logging.info(f"{sname.capitalize()} reconstruction ({len(idx)} shapes, {time.time() - time_test:.0f}s): final error={err:.6f}")
            
            # Render and save
            if use_poses:
                meshes = [create_mesh(model, lat, grid_filler=True, R=R, t=t, s=s) 
                          for lat, R, t, s in zip(latent, rotations, translations, scales)]
                parts = [create_parts(model, lat, grid_filler=True, R=R, t=t, s=s) 
                         for lat, R, t, s in zip(latent, rotations, translations, scales)]
            else:
                meshes = [create_mesh(model, lat, grid_filler=True) for lat in latent]
                parts = [create_parts(model, lat, grid_filler=True) for lat in latent]
            parts = [trimesh.util.concatenate(get_color_parts(part)) for part in parts]
            renders = viz.render_meshes(meshes, size=224, aa_factor=2)
            renders_p = viz.render_meshes(parts, size=224, use_texture=True, aa_factor=2)
            ws.save_renders(expdir, renders, str(history['epoch'])+"_"+sname)
            # Save latents and meshes
            latent_subdir = ws.get_recon_latent_subdir(expdir, history['epoch'])
            mesh_subdir = ws.get_recon_mesh_subdir(expdir, history['epoch'])
            os.makedirs(latent_subdir, exist_ok=True)
            os.makedirs(mesh_subdir, exist_ok=True)
            for i, instance in enumerate(test_instances):
                torch.save(latent[i:i+1], os.path.join(latent_subdir, instance + ".pth"))
                meshes[i].export(os.path.join(mesh_subdir, instance + ".obj"))
            if use_poses:
                pose_subdir = ws.get_recon_poses_subdir(expdir, history['epoch'])
                os.makedirs(pose_subdir, exist_ok=True)
                for i, instance in enumerate(test_instances):
                    torch.save(torch.cat([rotations[i:i+1], translations[i:i+1], scales[i:i+1]], dim=-1), 
                               os.path.join(pose_subdir, instance + ".pth"))

    torch.cuda.empty_cache()  # release unused GPU memory

    duration = time.time() - start_time
    duration_msg = "{:.0f}h {:02.0f}min {:02.0f}s".format(duration // 3600, (duration // 60) % 60, duration % 60)
    logging.info(f"End of training after {duration_msg}.")
    logging.info(f"Results saved in {expdir}.")


if __name__ == "__main__":
    args = parser()
    main(args)