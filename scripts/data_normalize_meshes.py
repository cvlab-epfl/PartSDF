"""
Normalize all meshes so that their bounding box is centered and 
contained within [-0.9 0.9]^3. Additionally, the shapes should face
toward +x, with up being +y and right +z.

In the dataset directory, it saves the meshes in under ./meshes/,
and normalization translation, rotation and scaling under ./normalization/.
"""

import os, os.path
import argparse
from math import ceil
from multiprocessing import Process

import numpy as np
import trimesh, trimesh.visual
from trimesh.transformations import rotation_matrix


SCALE = 0.9  # in [-0.9, 0.9]^3 cube


def load_mesh(datadir, instance, no_texture=True):
    mesh = trimesh.load(os.path.join(datadir, instance + ".obj"), force='mesh')
    if no_texture:
        mesh.visual = trimesh.visual.color.ColorVisuals()
    return mesh


def normalize_mesh(mesh):
    mesh = mesh.copy()
    # 1. Center the mesh
    T = -mesh.bounds.mean(0)
    mesh.apply_translation(T)
    # 2. Rotate it toward +x (!depends on the dataset!)
    R = np.eye(4)
    # R = rotation_matrix(-np.pi / 2, [0, 1, 0])  # example
    mesh.apply_transform(R)
    # 3. Normalize it to [-0.9, +0.9] cube
    S = 2. * SCALE / np.max(mesh.extents)
    mesh.apply_scale(S)
    return mesh, (T, R, S)


def normalize_meshes(args, source, dest, instances, pid=None):
    """Copy and normalize all meshes in the given list of instances."""
    if pid is not None:  # print with process id
        def iprint(*args, **kwargs):
            print(f"P{pid}: ", sep="", end="", flush=True)
            return print(*args, **kwargs)
    else:
        iprint = print
    
    n_shapes = len(instances)
    iprint(f"{n_shapes} shapes to process:")
    for i, instance in enumerate(instances):
        if (i+1) % max(1, n_shapes//5) == 0:
            iprint(f"Normalizing shape {i+1}/{n_shapes}...")

        if not args.overwrite and os.path.isfile(os.path.join(dest, "meshes", instance + ".obj")):
            continue

        mesh = load_mesh(source, instance)
        mesh, (T, R, S) = normalize_mesh(mesh)
        mesh.export(os.path.join(dest, "meshes", instance + ".obj"))
        np.savez(os.path.join(dest, "normalization", instance + ".npz"), T=T, R=R, S=S)

    iprint("Done.")


def main(args):
    """Launch the (parallel) normalization of meshes."""
    source, dest = args.source, args.dest
    filenames = sorted(os.listdir(source))
    filenames = [fn for fn in filenames if not fn.startswith(('.', '_'))]
    filenames = [os.path.splitext(fn)[0] for fn in filenames]
    filenames.sort()
    os.makedirs(dest, exist_ok=True)
    os.makedirs(os.path.join(dest, "meshes"), exist_ok=True)
    os.makedirs(os.path.join(dest, "normalization"), exist_ok=True)

    if args.nproc == 0:
        normalize_meshes(args, source, dest, filenames)
    else:
        # Divide shapes into chunks
        filename_chunks = []
        len_chunks = ceil(len(filenames) / args.nproc)
        i = -1
        for i in range(args.nproc - 1):
            filename_chunks.append(filenames[i * len_chunks: (i + 1) * len_chunks])
        filename_chunks.append(filenames[(i + 1) * len_chunks:])

        # Create sub-processes
        processes = []
        for i, chunk in enumerate(filename_chunks):
            p = Process(target=normalize_meshes, args=(args, source, dest, chunk), kwargs={"pid": i})
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    print(f"Results saved in {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize instance meshes of a dataset.")
    parser.add_argument("source", type=str, help="source directory containing the instances to normalize")
    parser.add_argument("dest", type=str, help="destination directory of the dataset")
    
    parser.add_argument("--nproc", default=0, type=int, help="number of processes to create to compute in parallel, give 0 to use main process (default: 0)")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing results, if any")

    args = parser.parse_args()

    main(args)