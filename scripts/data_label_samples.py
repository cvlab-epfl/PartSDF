"""
Label 3D data samples with parts.
"""

import os, os.path
import argparse
from math import ceil
from multiprocessing import Process

import numpy as np
import trimesh
import igl


def distance_to_mesh(points, mesh):
    """Return the squared distance of the points to the mesh. If empty, return +inf."""
    if not mesh.is_empty and points.size > 0:
        return igl.point_mesh_squared_distance(points, mesh.vertices, mesh.faces)[0].astype(np.float32)
    else:
        return np.full(points.shape[0], float('inf')).astype(np.float32)
    

def load_parts(partdir, instance, n_parts):
    parts = []
    for i in range(n_parts):
        fn = os.path.join(partdir, "meshes", instance, f"part{i}.obj")
        if os.path.isfile(fn):
            parts.append(trimesh.load(fn))
        else:
            parts.append(trimesh.Trimesh())
    return parts


# Main
######

def label_samples(args, datadir, outdir, instances, pid=None):
    """Generate data samples for the given meshes."""
    if pid is not None:  # print with process id
        _print = print
        def print(*args, **kwargs):
            _print(f"P{pid}: ", sep="", end="", flush=True)
            return _print(*args, **kwargs)

    n_shapes = len(instances)
    print(f"{n_shapes} shapes to process:")
    for i, instance in enumerate(instances):
        if (i+1) % max(1, n_shapes//5) == 0:
            print(f"Generating for shape {i+1}/{n_shapes}...")
        
        ## Load the parts
        parts = load_parts(outdir, instance, args.n_parts)
        
        ## Label each SDF sample based on its closest part
        os.makedirs(os.path.join(outdir, "sample_labels", instance), exist_ok=True)
        for sample_type in ['uniform', 'nearsurface', 'deepsdf']:
            filename_sample = os.path.join(datadir, "samples", instance, sample_type+".npz")
            if not os.path.exists(filename_sample):
                continue
            filename = os.path.join(outdir, "sample_labels", instance, sample_type+".npz")
            if args.overwrite or not os.path.isfile(filename):
                samples = np.load(filename_sample)
                # Find closest parts
                try:
                    dists2 = {k: np.stack([distance_to_mesh(samples[k], parts[i]) for i in range(5)], axis=0)
                            for k in ['pos', 'neg']}
                    closest = {k: np.argmin(dists2[k], axis=0) for k in ['pos', 'neg']}
                    np.savez(filename, **closest)
                except ValueError as err:
                    print(f"Error for {instance} ({sample_type}): {err}")
            
    print("Done.")


def main(args):
    """Launch the (parallel) labeling of data samples."""
    np.random.seed(args.seed)

    datadir, outdir = args.datadir, args.partdir
    filenames = sorted(os.listdir(os.path.join(datadir, "meshes")))
    filenames = [fn for fn in filenames if not fn.startswith(('.', '_'))]
    instances = [os.path.splitext(fn)[0] for fn in filenames]
    instances.sort()
    outdir = os.path.join(datadir, outdir)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "sample_labels"), exist_ok=True)

    if args.nproc == 0:
        label_samples(args, datadir, outdir, instances)
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
            p = Process(target=label_samples, args=(args, datadir, outdir, chunk), kwargs={"pid": i})
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    print(f"Results saved in {outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="label data samples for meshes.")
    parser.add_argument("datadir", type=str, help="directory of the dataset")
    parser.add_argument("n_parts", type=int, help="max number of parts used")

    parser.add_argument("--partdir", default="parts", type=str, help="name of the subdir where the part decomposition will be saved")

    parser.add_argument("--nproc", default=0, type=int, help="number of processes to create to compute in parallel, give 0 to use main process (default: 0)")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing results, if any")
    parser.add_argument("--seed", default=0, help="seed for the RNGs")

    args = parser.parse_args()

    main(args)