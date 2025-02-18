"""
Normalize the parts accordingly to the meshes normalizations.

Uses normalization data found under ./normalization/ in the dataset directory,
and saves the normalized parts under ./<partdir>/meshes/.
"""

import os, os.path
import argparse
from math import ceil
from multiprocessing import Process

import numpy as np
import trimesh, trimesh.visual
from trimesh.transformations import rotation_matrix


SCALE = 0.9  # in [-0.9, 0.9]^3 cube


def load_part(datadir, instance, part_num, no_texture=True):
    filename = os.path.join(datadir, instance, f"part{part_num}.obj")
    if not os.path.isfile(filename):
        return trimesh.Trimesh()
    part = trimesh.load(filename, force='mesh')
    if no_texture:
        part.visual = trimesh.visual.color.ColorVisuals()
    return part


def normalize_part(part, T, R, S):
    part = part.copy()
    part.apply_translation(T)
    part.apply_transform(R)
    part.apply_scale(S)
    return part


def normalize_parts(args, source, dest, partdir, instances, pid=None):
    """Copy and normalize all meshes in the given list of instances."""
    if pid is not None:  # print with process id
        _print = print
        def print(*args, **kwargs):
            _print(f"P{pid}: ", sep="", end="", flush=True)
            return _print(*args, **kwargs)

    normdir = os.path.join(dest, "normalization")
    
    n_shapes = len(instances)
    print(f"{n_shapes} shapes to process:")
    for i, instance in enumerate(instances):
        if (i+1) % max(1, n_shapes//5) == 0:
            print(f"Normalizing shape {i+1}/{n_shapes}...")

        # Load all parts
        parts = [load_part(source, instance, i) for i in range(args.n_parts)]

        # Verify if normalized parts already exist
        skip = args.overwrite
        for j, part in enumerate(parts):
            if not part.is_empty:
                skip = skip or not os.path.isfile(os.path.join(dest, partdir, "meshes", instance, f"part{j}.obj"))
        if skip:
            continue

        # Load the normalization of the corresponding mesh
        norm = np.load(os.path.join(normdir, instance + ".npz"))
        T, R, S = norm["T"], norm["R"], norm["S"]

        # Normalize and save the parts
        os.makedirs(os.path.join(dest, partdir, "meshes", instance), exist_ok=True)
        for j, part in enumerate(parts):
            if part.is_empty:
                continue
            part = normalize_part(part, T, R, S)
            part.export(os.path.join(dest, partdir, "meshes", instance, f"part{j}.obj"))

    print("Done.")


def main(args):
    """Launch the (parallel) normalization of meshes."""
    source, dest, partdir = args.source, args.dest, args.partdir
    instances = sorted(os.listdir(source))
    instances = [i for i in instances if not i.startswith(('.', '_'))]
    instances.sort()
    os.makedirs(dest, exist_ok=True)
    os.makedirs(os.path.join(dest, partdir, "meshes"), exist_ok=True)

    if args.nproc == 0:
        normalize_parts(args, source, dest, partdir, instances)
    else:
        # Divide shapes into chunks
        filename_chunks = []
        len_chunks = ceil(len(instances) / args.nproc)
        i = -1
        for i in range(args.nproc - 1):
            filename_chunks.append(instances[i * len_chunks: (i + 1) * len_chunks])
        filename_chunks.append(instances[(i + 1) * len_chunks:])

        # Create sub-processes
        processes = []
        for i, chunk in enumerate(filename_chunks):
            p = Process(target=normalize_parts, args=(args, source, dest, partdir, chunk), kwargs={"pid": i})
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    print(f"Results saved in {dest}/{partdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize instance parts of a dataset.")
    parser.add_argument("source", type=str, help="source directory containing the parts to normalize (one subdir per instance)")
    parser.add_argument("dest", type=str, help="destination directory of the dataset")
    
    parser.add_argument("--partdir", default="parts", type=str, help="name of the subdir where the part decomposition will be saved")
    
    parser.add_argument("--nproc", default=0, type=int, help="number of processes to create to compute in parallel, give 0 to use main process (default: 0)")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing results, if any")

    args = parser.parse_args()

    main(args)