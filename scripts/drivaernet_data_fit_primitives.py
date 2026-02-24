"""
Fit primitives to the parts.
For DrivAerNet++ dataset.

Note: use the bounding boxes instead of fitting to the mesh.

The following settings are used:
- primitives: cuboids for body and back, cylinders for wheels
- reference quaternions: identity for all
- Use bouding boxes directly (no fitting)
"""

import os, os.path
import sys
import argparse
import json
from math import ceil
from multiprocessing import Process

import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.primitives import standardize_quaternion, mesh_cuboid, mesh_cylinder


# Utils
#######

def load_parts(partdir, instance):
    parts = []
    for i in range(6):  # 6 parts
        fn = os.path.join(partdir, "meshes", instance, f"part{i}.obj")
        if os.path.isfile(fn):
            parts.append(trimesh.load(fn))
        else:
            parts.append(trimesh.Trimesh())
    return parts


# Cuboid
########

def init_cuboid(part, init_quaternion=np.array([1., 0., 0., 0.])):
    """Get the convex hull of the part and its initial cuboid parameters."""
    scale = part.extents
    t = part.bounds.mean(0)
    quat = init_quaternion.copy()
    return scale, t, quat


# Cylinder
##########

def init_cylinder(part, init_quaternion=np.array([1., 0., 0., 0.])):
    """Get the convex hull of the part and its initial cylinder parameters."""
    radius = part.extents[:2].mean() / 2
    height = part.extents[2]
    t = part.bounds.mean(0)
    quat = init_quaternion.copy()
    return radius, height, t, quat


########
# Main #
########

def fit_primitives(args, partdir, instances, pid=None):
    """Fit primitives to the given shape instances."""
    if pid is not None:  # print with process id
        def iprint(*args, **kwargs):
            print(f"P{pid}: ", sep="", end="", flush=True)
            return print(*args, **kwargs)
    else:
        iprint = print
        
    # Reference quaternion/orientation
    ref_quaternions = standardize_quaternion(np.array(args.ref_quaternion))

    n_shapes = len(instances)
    iprint(f"{n_shapes} shapes to process:")
    for i, instance in enumerate(instances):
        if (i+1) % max(1, n_shapes//5) == 0:
            iprint(f"Generating for shape {i+1}/{n_shapes}...")

        # Load the parts
        parts = load_parts(partdir, instance)
        paramdir = os.path.join(partdir, "parameters", instance)
        primdir = os.path.join(partdir, "primitives", instance)
        os.makedirs(paramdir, exist_ok=True)
        os.makedirs(primdir, exist_ok=True)

        # Verify if already computed
        if not args.overwrite and \
           os.path.exists(os.path.join(paramdir, "quaternions.npy")) and \
           os.path.exists(os.path.join(paramdir, "translations.npy")) and \
           os.path.exists(os.path.join(paramdir, "scales.npy")):
            continue

        # All parts' parameters
        quaternions, translations, scales = [], [], []

        # Fit primitives to the parts
        for j, part in enumerate(parts):
            ref_quaternion = ref_quaternions[j]
            if part.is_empty:
                quaternions.append(np.full(4, np.nan))
                translations.append(np.full(3, np.nan))
                scales.append(np.full(3, np.nan))
                continue
            if args.primitive[j] == "cuboid":
                scale, translation, quaternion = init_cuboid(part, ref_quaternion)
                cuboid = mesh_cuboid(scale, translation, quaternion)
                # Save the results
                np.savez(os.path.join(paramdir, f"part{j}_cuboid.npz"), scale=scale,
                         translation=translation, quaternion=quaternion)
                cuboid.export(os.path.join(primdir, f"part{j}_cuboid.obj"))
                scale = scale / 2  # half-lengths
            elif args.primitive[j] == "cylinder":
                radius, height, translation, quaternion = init_cylinder(part, ref_quaternion)
                cylinder = mesh_cylinder(radius, height, translation, quaternion)
                # Save the results
                np.savez(os.path.join(paramdir, f"part{j}_cylinder.npz"), radius=radius, height=height, 
                         translation=translation, quaternion=quaternion)
                cylinder.export(os.path.join(primdir, f"part{j}_cylinder.obj"))
                scale = np.array([radius, radius, height / 2])  # radius and half-length
            quaternions.append(quaternion)
            translations.append(translation)
            scales.append(scale)
        
        # Save the global parameters
        quaternions = np.stack(quaternions, axis=0)
        translations = np.stack(translations, axis=0)
        scales = np.stack(scales, axis=0)
        np.save(os.path.join(paramdir, "quaternions.npy"), quaternions)
        np.save(os.path.join(paramdir, "translations.npy"), translations)
        np.save(os.path.join(paramdir, "scales.npy"), scales)
            
        # Delete variables as there seems to be a weird memory leak?
        del quaternions, translations, scales, parts

    iprint("Done.")


def main(args):
    """Launch the (parallel) fitting of primitives."""
    np.random.seed(args.seed)

    datadir, partdir = args.datadir, args.partdir
    with open(os.path.join(datadir, "splits", args.splitfile + ".json"), "r") as f:
        instances = json.load(f)
    partdir = os.path.join(datadir, partdir)
    os.makedirs(partdir, exist_ok=True)

    if args.nproc == 0:
        fit_primitives(args, partdir, instances)
    else:
        # Divide shapes into chunks
        instance_chunks = []
        len_chunks = ceil(len(instances) / args.nproc)
        i = -1
        for i in range(args.nproc - 1):
            instance_chunks.append(instances[i * len_chunks: (i + 1) * len_chunks])
        instance_chunks.append(instances[(i + 1) * len_chunks:])

        # Create sub-processes
        processes = []
        for i, chunk in enumerate(instance_chunks):
            p = Process(target=fit_primitives, args=(args, partdir, chunk), kwargs={"pid": i})
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    print(f"Results saved in {partdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Primitive fitting for parts.")
    parser.add_argument("datadir", type=str, help="directory of the dataset")
    parser.add_argument("splitfile", type=str, help="split file listing the instances to process (without the .json)")
    parser.add_argument("partdir", type=str, help="name of the subdir where the part decomposition is saved")
    
    parser.add_argument("--nproc", default=0, type=int, help="number of processes to create to compute in parallel, give 0 to use main process (default: 0)")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing results, if any")
    parser.add_argument("--seed", default=0, type=int, help="seed for the RNGs")

    args = parser.parse_args()

    # Duplicate single arguments for each part
    args.primitive = ["cuboid"] * 2 + ["cylinder"] * 4
    args.ref_quaternion = [[1., 0., 0., 0.]] * 6

    main(args)