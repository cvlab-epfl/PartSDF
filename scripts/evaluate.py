"""
Main evaluation script.

Evaluate a set of shapes reconstructed with a deep implicit model.
"""

import os, os.path
import sys
import argparse
import logging
import json
import time

import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import workspace as ws
from src.metric import chamfer_distance, mesh_iou, image_consistency, part_iou
from src.utils import configure_logging, set_seed


def parser(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate shapes reconstructed with a deep implicit model.")

    parser.add_argument("experiment", help="path to the experiment directory")

    # Evaluation arguments
    parser.add_argument('--debug', action='store_true', help="increase verbosity to print debugging messages")
    parser.add_argument('--load-epoch', default='latest', help="epoch to load, default to latest available")
    parser.add_argument("--mesh-to-sdf", action='store_true', help="use the mesh-to-sdf package to compute the SDF and surface samples (useful for dirty meshes, e.g., unprocessed ShapeNet)")
    parser.add_argument('--overwrite', action='store_true', help="overwrite shapes that are already evaluated")
    parser.add_argument('-q', '--quiet', dest="verbose", action='store_false',  help="disable verbosity and run in quiet mode")
    parser.add_argument('--seed', type=int, default=0, help="initial seed for the RNGs (default=0)")
    parser.add_argument('-s', '--split', help="split to evaluate, default to \"ValidSplit\" in specs file")
    parser.add_argument('--suffix', type=str, default="", help="suffix to add to the reconstruction and evaluation directories")
    parser.add_argument('-t', '--test', action='store_true', help="reconstruct the test set, otherwise reconstruct the validation set (--split override this)")

    # Parts arguments
    parser.add_argument('--parts', action='store_true', help="use reconstruction made with part supervision")

    # Chamfer-Distance
    parser.add_argument('--cd-samples', type=int, default=30000, help="number of surface samples for chamfer distance (default=30000)")
    parser.add_argument('--cd-no-square-dist', action='store_false', dest="cd_square_dist", help="do not use square distances for the chamfer-distance")

    # Mesh Intersection-over-Union
    parser.add_argument('--iou-resolution', type=int, default=128, help="resolution of the grid for the mesh IoU (default=128)")
    parser.add_argument('--iou-skip', action='store_true', help="skip the IoU computation, e.g., if meshes are dirty and need fast results")

    # Image Consistency
    parser.add_argument('--ic-skip', action='store_true', help="skip the Image Consistency computation")

    # Part Intersection-over-Union
    parser.add_argument('--piou-resolution', type=int, default=128, help="resolution of the grid for the part IoU (default=128)")
    parser.add_argument('--piou-skip', action='store_true', help="skip the part IoU computation, e.g., if meshes are dirty and need fast results")

    args = parser.parse_args(argv)

    if args.mesh_to_sdf:
        raise NotImplementedError("Would be too slow, probably better to pre-compute them.")
        # from mesh_to_sdf import mesh_to_sdf, get_surface_point_cloud
        os.environ['PYOPENGL_PLATFORM'] = 'egl'

    return args


def smart_load(filename):
    """Load the mesh if it exists, else return an empty mesh."""
    if os.path.isfile(filename):
        return trimesh.load(filename)
    return trimesh.Trimesh()


def main(args=None):
    # Initialization
    if args is None:
        args = parser()
    set_seed(args.seed)
    start_time = time.time()

    expdir = args.experiment
    specs = ws.load_specs(expdir)
    configure_logging(args, os.path.join(ws.get_log_dir(expdir), "evallog.txt"))

    logging.info(f"Command:  python {' '.join(sys.argv)}")
    logging.info(f"Date: " + time.strftime("%d %B %Y at %H:%M:%S (%Y%m%d-%H%M%S)"))
    logging.info(f"Evaluating shapes in {expdir}.")
    logging.info(f"arguments = {args}")
    
    # Data
    if args.split is None:
        args.split = specs["TestSplit"] if args.test or specs["ValidSplit"] is None else specs["ValidSplit"]
    with open(args.split) as f:
        instances = json.load(f)
    n_shapes = len(instances)
    datasource = os.path.join(specs["DataSource"], specs["SamplesDir"])
    use_parts = specs["Network"].lower().startswith("part")
    if use_parts:
        n_parts = specs["Parts"]["NumParts"]
        partdir = specs["Parts"]["ParametersDir"].split("/")[0]

    logging.info(f"{n_shapes} shapes in {args.split} to evaluate.")

    # Resume from checkpoint
    if args.load_epoch == 'latest':
        args.load_epoch = ws.load_history(expdir)['epoch']

    # Parameters and directories
    if args.suffix:
        args.suffix = "_" + args.suffix
    load_epoch = str(args.load_epoch) + ("_parts" if args.parts else "") + args.suffix
    eval_dir = ws.get_eval_dir(expdir, load_epoch)
    mesh_subdir = ws.get_recon_mesh_subdir(expdir, load_epoch)
    if use_parts:
        parts_subdir = ws.get_recon_parts_subdir(expdir, load_epoch)
    os.makedirs(eval_dir, exist_ok=True)

    logging.info(f"Evaluating checkpoint from epoch={load_epoch}.")

    ## Evaluation metrics
    results = {}
    filenames = {}
    metrics = [
        "chamfer",  # Chamfer-Distance
        "iou",      # Mesh Intersection-over-Union
        "ic",       # Image Consistency
    ]
    if use_parts:
        metrics.append("piou")  # Part Intersection-over-Union
    for metric in metrics:
        results[metric] = {}
        filenames[metric] = os.path.join(eval_dir, metric+".json")
        if os.path.isfile(filenames[metric]):
            with open(filenames[metric]) as f:
                results[metric].update(json.load(f))

    # Evaluation
    for i, instance in enumerate(instances):
        logging.info(f"Shape {i+1}/{n_shapes} ({instance})")

        # Reconstruction
        recon_mesh_filename = os.path.join(mesh_subdir, instance + ".obj")
        if not os.path.isfile(recon_mesh_filename):
            logging.warning(f"no mesh found under {recon_mesh_filename}! Skipping...")
            continue
        try:
            recon_mesh = trimesh.load(recon_mesh_filename)
        except ValueError:  # empty mesh
            recon_mesh = trimesh.Trimesh()

        # Chamfer-Distance
        if not args.overwrite and instance in results["chamfer"]:
            logging.info(f"chamfer = {results['chamfer'][instance]} (existing)")
        else:
            # Load GT surface samples
            gt_samples = np.load(os.path.join(datasource, instance, "surface.npy"))[:, :3]
            gt_samples = np.random.permutation(gt_samples)[:args.cd_samples, :3]

            # Reconstruction surface samples
            if not recon_mesh.is_empty:
                recon_samples = recon_mesh.sample(args.cd_samples)
            else:
                recon_samples = np.zeros((0, 3))

            # Chamfer-Distance
            chamfer_val = chamfer_distance(gt_samples, recon_samples, square_dist=args.cd_square_dist)
            results["chamfer"][instance] = float(chamfer_val)
            logging.info(f"chamfer = {chamfer_val}")
        
        # Mesh Intersection-over-Union
        if not args.overwrite and instance in results["iou"]:
            logging.info(f"iou = {results['iou'][instance]} (existing)")
        elif not args.iou_skip:
            # Load GT mesh
            gt_mesh = trimesh.load(os.path.join(specs['DataSource'], "meshes", instance+".obj"))

            # Mesh Intersection-over-Union
            iou_val = mesh_iou(gt_mesh, recon_mesh, args.iou_resolution)
            results["iou"][instance] = float(iou_val)
            logging.info(f"iou = {iou_val}")
        
        # Image Consistency
        if not args.overwrite and instance in results["ic"]:
            logging.info(f"image consistency = {results['ic'][instance]} (existing)")
        elif not args.ic_skip:
            # Load GT mesh
            gt_mesh = trimesh.load(os.path.join(specs['DataSource'], "meshes", instance+".obj"))

            # Compute Image Consistency
            ic_val = image_consistency(gt_mesh, recon_mesh)
            results["ic"][instance] = float(ic_val)
            logging.info(f"image consistency = {ic_val}")
        
        # Part Intersection-over-Union
        if use_parts:
            if not args.overwrite and instance in results["piou"]:
                logging.info(f"piou = {results['piou'][instance]} (existing)")
            elif not args.piou_skip:
                # Load parts
                gt_parts = [smart_load(os.path.join(specs['DataSource'], partdir, "meshes", instance, f"part{i}.obj"))
                            for i in range(n_parts)]
                parts = [smart_load(os.path.join(parts_subdir, instance + f"_{i}.obj"))
                         for i in range(n_parts)]

                # Part Intersection-over-Union
                piou_val = part_iou(gt_parts, parts, args.piou_resolution, process1=True)
                results["piou"][instance] = float(piou_val)
                logging.info(f"part iou = {piou_val}")
    
    # Save results
    for metric in metrics:
        with open(filenames[metric], "w") as f:
            json.dump(results[metric], f, indent=2)
    
    # Average and median metrics
    for metric in metrics:
        all_values = list(results[metric].values())
        all_values = [v for v in all_values if not np.isnan(v)]
        if len(all_values) == 0:
            logging.info(f"Empty metric {metric}  ({len(all_values)}/{len(instances)} shapes)")
            continue
        logging.info(f"Average {metric} = {np.mean(all_values)}  ({len(all_values)}/{len(instances)} shapes)")
        logging.info(f"Median  {metric} = {np.median(all_values)}  ({len(all_values)}/{len(instances)} shapes)")

    duration = time.time() - start_time
    duration_msg = "{:.0f}h {:02.0f}min {:02.0f}s".format(duration // 3600, (duration // 60) % 60, duration % 60)
    logging.info(f"End of evaluation after {duration_msg}.")
    logging.info(f"Results saved in {expdir}.")


if __name__ == "__main__":
    args = parser()
    main(args)