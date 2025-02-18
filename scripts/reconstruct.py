"""
Main reconstruction script.

Reconstruct a set of shapes with a deep implicit model.
"""

import os, os.path
import sys
import argparse
import logging
import json
import time

import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import workspace as ws
from src.loss import get_loss_recon
from src.mesh import create_mesh, create_parts
from src.model import get_model, get_part_latents, get_part_poses
from src.primitives import standardize_quaternion
from src.reconstruct import reconstruct, reconstruct_parts
from src.utils import configure_logging, set_seed, get_device, get_color_parts


def parser(argv=None):
    parser = argparse.ArgumentParser(description="Reconstruct shapes with an implicit model.")

    parser.add_argument("experiment", help="path to the experiment directory")

    # Reconstruction arguments
    parser.add_argument('--debug', action='store_true', help="increase verbosity to print debugging messages")
    parser.add_argument('-i', '--iters', type=int, default=800, help="number of iterations for latent optimization")
    parser.add_argument('--load-epoch', default='latest', help="epoch to load, default to latest available")
    parser.add_argument('--lr', type=float, default=5e-3, help="learning rate for latent optimization")
    parser.add_argument('--max-norm', action='store_true', help="use the max norm from specs to bound the reconstructed latent (otherwise is unbounded)")
    parser.add_argument('-n', '--n-samples', type=int, default=8000, help="number of sdf samples used per iteration")
    parser.add_argument('--overwrite', action='store_true', help="overwrite shapes that are already reconstructed")
    parser.add_argument('-q', '--quiet', dest="verbose", action='store_false',  help="disable verbosity and run in quiet mode")
    parser.add_argument('-r', '--resolution', type=int, default=256, help="resolution for the reconstruction with marching cubes")
    parser.add_argument('--seed', type=int, default=0, help="initial seed for the RNGs (default=0)")
    parser.add_argument('-s', '--split', help="split to reconstruct, default to \"TestSplit\" in specs file")
    parser.add_argument('--suffix', type=str, default="", help="suffix to add to the reconstruction directories")
    parser.add_argument('-t', '--test', action='store_true', help="reconstruct the test set, otherwise reconstruct the validation set (--split override this)")

    # Parts arguments
    parser.add_argument('--no-intersection-loss', action='store_false', dest='intersection_loss', help="disable intersection loss for part reconstruction (default to the value from specs.json!)")
    parser.add_argument('--parts', action='store_true', help="use part reconstruction losses")
    parser.add_argument('--partsdf', action='store_true', help="the model takes part latents as input and computes their SDFs")

    args = parser.parse_args(argv)

    return args


@torch.no_grad()
def fill_nans_average(tensor, average):
    """Replace invalid values (any along dim=-1) by the average."""
    index = torch.isnan(tensor).any(-1)
    tensor[index] = average[index].detach().clone().to(tensor.dtype).to(tensor.device)


def main(args=None):
    # Initialization
    if args is None:
        args = parser()
    set_seed(args.seed)
    device = get_device()
    start_time = time.time()

    expdir = args.experiment
    specs = ws.load_specs(expdir)
    device = specs.get("Device", get_device())
    configure_logging(args, os.path.join(ws.get_log_dir(expdir), "reconlog.txt"))

    logging.info(f"Command:  python {' '.join(sys.argv)}")
    logging.info(f"Date: " + time.strftime("%d %B %Y at %H:%M:%S (%Y%m%d-%H%M%S)"))
    logging.info(f"Reconstructing shapes in {expdir}. (on {device})")
    logging.info(f"arguments = {args}")
    
    # Data
    if args.split is None:
        args.split = specs["TestSplit"] if args.test or specs["ValidSplit"] is None else specs["ValidSplit"]
    with open(args.split) as f:
        instances = json.load(f)
    n_shapes = len(instances)
    n_samples = args.n_samples
    datasource = os.path.join(specs["DataSource"], specs["SamplesDir"])
    samplefile = specs["SamplesFile"]
    # Does the model use part decomposition? (and parts poses)
    use_parts = specs["Network"].split("-")[0].lower().startswith("part")
    use_poses = specs["Parts"].get("UsePoses", False) if use_parts else False
    paramdir = specs["Parts"].get("ParametersDir", None) if use_poses else None

    logging.info(f"{n_shapes} shapes in {args.split} to reconstruct.")

    # Model
    latent_dim = specs["LatentDim"]
    if use_parts:
        n_parts = specs["NetworkSpecs"].pop("n_parts") if "n_parts" in specs["NetworkSpecs"] else specs["Parts"]["NumParts"]
        part_dim = specs["NetworkSpecs"].pop("part_dim") if "part_dim" in specs["NetworkSpecs"] else specs["Parts"]["LatentDim"]
    else:
        n_parts, part_dim = None, None
    use_occ = True if specs.get("ImplicitField", "SDF").lower() in ["occ", "occupancy"] else False
    spherical_lats = specs.get("LatentSpherical", False)
    model = get_model(
        specs.get("Network", "DeepSDF"),
        **specs.get("NetworkSpecs", {}),
        latent_dim=latent_dim,
        n_parts=n_parts,
        part_dim=part_dim,
        use_occ=use_occ
    ).to(device)
    # Evaluation mode with frozen model
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    
    # Get training latents and poses to get average values (to init the reconstruction)
    if use_poses:
        with open(specs["TrainSplit"]) as f:
            train_instances = json.load(f)
        latents = get_part_latents(len(train_instances), n_parts, part_dim, specs.get("LatentBound", None), device=device)
        poses = get_part_poses(len(train_instances), n_parts, freeze=True, device=device, fill_nans=True)

    logging.info(f"Model has {sum([x.nelement() for x in model.parameters()]):,} parameters.")
    if use_poses:
        logging.info(f"Using part poses.")

    # Loss
    loss_recon = get_loss_recon(specs["ReconLoss"], reduction='none').to(device)
    if args.parts:
        recon_lambda = specs["ReconLossLambda"]
        part_loss_recon = get_loss_recon(specs["Parts"]["ReconLoss"], reduction='none').to(device)
        part_recon_lambda = specs["Parts"]["ReconLossLambda"]
    latent_reg = specs["LatentRegLambda"]
    if use_parts:
        if args.partsdf:
            latent_reg = specs["Parts"].get("LatentRegLambda", None)
        inter_lambda = specs["Parts"].get("IntersectionLambda", None) if args.intersection_loss else None
        inter_temp = specs["Parts"].get("IntersectionTemp", 1.)
        if inter_lambda is not None:
            logging.info(f"Using intersection loss with lambda={inter_lambda} and temperature={inter_temp}.")
    else:
        inter_lambda, inter_temp = None, None

    # Resume from checkpoint
    if args.load_epoch == 'latest':
        args.load_epoch = ws.load_history(expdir)['epoch']
    ws.load_model(expdir, model, args.load_epoch)
    if use_poses:
        ws.load_latents(expdir, latents, args.load_epoch)
        ws.load_poses(expdir, poses)
        latents_init = latents.weight.detach().clone().mean(0, keepdim=True)
        poses_init = poses.weight.detach().clone().mean(0, keepdim=True)
        rotations_init = standardize_quaternion(poses_init[..., :4])
        translations_init = poses_init[..., 4:7]
        scales_init = poses_init[..., 7:10]

    logging.info(f"Loaded checkpoint from epoch={args.load_epoch}.")

    # Parameters and directories
    if args.suffix:
        args.suffix = "_" + args.suffix
    clampD = specs["ClampingDistance"]
    max_norm = specs["LatentBound"] if args.max_norm else None
    load_epoch = str(args.load_epoch) + ("_parts" if args.parts else "") + args.suffix
    latent_subdir = ws.get_recon_latent_subdir(expdir, load_epoch)
    mesh_subdir = ws.get_recon_mesh_subdir(expdir, load_epoch)
    os.makedirs(latent_subdir, exist_ok=True)
    os.makedirs(mesh_subdir, exist_ok=True)
    if use_parts:
        parts_subdir = ws.get_recon_parts_subdir(expdir, load_epoch)
        os.makedirs(parts_subdir, exist_ok=True)
    if use_poses:
        pose_subdir = ws.get_recon_poses_subdir(expdir, load_epoch)
        os.makedirs(pose_subdir, exist_ok=True)

    # Reconstruction
    for i, instance in enumerate(instances):
        logging.info(f"Shape {i+1}/{n_shapes} ({instance})")
        if not args.overwrite and \
           os.path.isfile(os.path.join(latent_subdir, instance + ".pth")) and \
           os.path.isfile(os.path.join(mesh_subdir, instance + ".obj")) and \
           (not use_parts or os.path.isfile(os.path.join(parts_subdir, instance + ".obj"))) and \
           (not use_poses or os.path.isfile(os.path.join(pose_subdir, instance + ".pth"))):
            logging.info(f"already existing, skipping...")
            continue
        
        # Load sdf data
        filename = os.path.join(datasource, instance, samplefile)
        npz = np.load(filename)
        if args.parts:
            filename = os.path.join(specs["DataSource"], specs["Parts"]["SamplesDir"], instance, samplefile)
            npz_parts = np.load(filename)
        if use_poses:
            if args.parts:
                rotations = torch.tensor(np.load(os.path.join(specs["DataSource"], paramdir, instance, "quaternions.npy")))
                translations = torch.tensor(np.load(os.path.join(specs["DataSource"], paramdir, instance, "translations.npy")))
                scales = torch.tensor(np.load(os.path.join(specs["DataSource"], paramdir, instance, "scales.npy")))
                _latent_init = None  # random init
                # Replace potential NaNs by the average pose (NaNs: part inexisting in dataset)
                fill_nans_average(rotations, rotations_init.squeeze(0))
                fill_nans_average(translations, translations_init.squeeze(0))
                fill_nans_average(scales, scales_init.squeeze(0))
            else:
                rotations, translations, scales = (rotations_init.clone(), translations_init.clone(), scales_init.clone())
                _latent_init = latents_init.clone()
        else:
            rotations, translations, scales, _latent_init = None, None, None, None

        # Optimize latent
        if args.parts:
            out = reconstruct_parts(model, npz, npz_parts, args.iters, n_samples, args.lr, loss_recon,
                                    recon_lambda, part_loss_recon, part_recon_lambda, latent_reg, clampD, 
                                    latent_init=_latent_init, latent_size=(part_dim if args.partsdf else latent_dim),
                                    n_parts=n_parts, is_part_sdfnet=args.partsdf, inter_lambda=inter_lambda, 
                                    inter_temp=inter_temp, rotations=rotations, translations=translations, 
                                    scales=scales, max_norm=max_norm, spherical_lats=spherical_lats,
                                    verbose=args.verbose, device=device)
        else:
            out = reconstruct(model, npz, args.iters, n_samples, args.lr, loss_recon, latent_reg, clampD,
                              latent_init=_latent_init, latent_size=(part_dim if args.partsdf else latent_dim),
                              n_parts=n_parts, is_part_sdfnet=args.partsdf, inter_lambda=inter_lambda, 
                              inter_temp=inter_temp, rotations=rotations, translations=translations, 
                              scales=scales, max_norm=max_norm, spherical_lats=spherical_lats,
                              verbose=args.verbose, device=device)
        err, latent = out[0], out[1]
        if use_poses:
            rotations, translations, scales = out[2], out[3], out[4]
        logging.info(f"Final error={err:.6f}, latent norm = {latent.norm():.4f}")

        # Reconstruct the mesh
        mesh = create_mesh(model, latent, N=args.resolution, max_batch=32**3, verbose=args.verbose, 
                           device=device, R=rotations, t=translations, s=scales)

        # Save results
        torch.save(latent, os.path.join(latent_subdir, instance + ".pth"))
        mesh.export(os.path.join(mesh_subdir, instance + ".obj"))

        # Reconstruct parts if applicable
        if use_parts:
            parts = create_parts(model, latent, N=args.resolution, max_batch=32**3, verbose=args.verbose, 
                                 device=device, R=rotations, t=translations, s=scales)
            trimesh.util.concatenate(get_color_parts(parts)).export(os.path.join(parts_subdir, instance + ".obj"))
            for i, part in enumerate(parts):
                if not part.is_empty:
                    part.export(os.path.join(parts_subdir, instance + f"_{i}.obj"))
        if use_poses:
            torch.save(torch.cat([rotations, translations, scales], dim=-1), os.path.join(pose_subdir, instance + ".pth"))
    
    torch.cuda.empty_cache()  # release unused GPU memory

    duration = time.time() - start_time
    duration_msg = "{:.0f}h {:02.0f}min {:02.0f}s".format(duration // 3600, (duration // 60) % 60, duration % 60)
    logging.info(f"End of reconstruction after {duration_msg}.")
    logging.info(f"Results saved in {expdir}.")


if __name__ == "__main__":
    args = parser()
    main(args)