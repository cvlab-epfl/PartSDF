"""
Generate 3D data samples with SDF values.
"""

import os, os.path
import sys
import argparse
from math import sqrt, ceil
from multiprocessing import Process

import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import remove_nans
from src.utils import make_grid2d, get_sdf_mesh


# 3D samples
############

def sample_surface(mesh, n_samples, dirty_mesh=False):
    """Sample points (with normals) on the surface of the given mesh."""
    if dirty_mesh:
        pc = get_surface_point_cloud(mesh)
        xyz, normals = pc.points, pc.normals
        indices = np.random.choice(xyz.shape[0], n_samples)
        xyz, normals = xyz[indices], normals[indices]
    else:
        xyz, fid = mesh.sample(n_samples, return_index=True)
        # Interpolate vertex normals from barycentric coordinates
        bary = trimesh.triangles.points_to_barycentric(triangles=mesh.triangles[fid], points=xyz)
        normals = (mesh.vertex_normals[mesh.faces[fid]] * bary.reshape((-1, 3, 1))).sum(axis=1)
    return np.concatenate((xyz, normals), axis=1).astype(np.float32)


# SDF samples
#############

def _prepare_sdf_results(mesh, xyz, dirty_mesh=False):
    """Compute the SDF, filter out bad samples, and prepare results for saving."""
    # Compute SDF values
    if dirty_mesh:
        sdf = mesh_to_sdf(mesh, xyz, surface_point_method='scan', sign_method='depth')
    else:
        sdf = get_sdf_mesh(mesh, xyz)
    # Remove NaNs
    samples = np.concatenate([xyz, sdf[:, None]], axis=1)
    samples = remove_nans(samples)
    # Prepare results
    pos_idx = sdf >= 0.
    pos = samples[pos_idx].astype(np.float32)
    neg = samples[~pos_idx].astype(np.float32)
    return {
        "pos": pos,
        "neg": neg
    }


def sample_sdf_nearsurface(mesh, n_samples, var=0.005, dirty_mesh=False):
    """Sample points near the surface of the given mesh."""
    # Nearsurface samples, computed by adding Gaussian noise to surface samples
    xyz = mesh.sample(n_samples)
    xyz = np.concatenate([
        xyz + np.random.normal(scale=sqrt(var), size=xyz.shape),
        xyz + np.random.normal(scale=sqrt(var/10.), size=xyz.shape)
    ], axis=0)
    return _prepare_sdf_results(mesh, xyz, dirty_mesh=dirty_mesh)


def sample_sdf_uniform(mesh, n_samples, dirty_mesh=False):
    """Sample points uniformly in the full domain."""
    # Uniform samples in the full domain
    xyz = np.random.uniform(low=-1, high=1, size=(n_samples, 3))
    return _prepare_sdf_results(mesh, xyz, dirty_mesh=dirty_mesh)


def make_deepsdf_samples(nearsurface_fn, uniform_fn, n_uniform=25000):
    """Sample data based on DeepSDF, Park et al. 2019.
    
    Samples among (with pos/neg balance):
        - 500K near-surface points
        - 25K uniform points
    """
    # Load the samples files
    surf_npz = np.load(nearsurface_fn)
    unif_npz = np.load(uniform_fn)

    # Sample 25K from uniform
    uniform = np.concatenate([unif_npz['pos'], unif_npz['neg']], 0)
    unif_idx = np.random.permutation(len(uniform))[:n_uniform]
    uniform = uniform[unif_idx]
    pos_idx = uniform[:, 3] > 0.
    pos = uniform[pos_idx]
    neg = uniform[~pos_idx]
    # Add the 500K samples from nearsurface
    pos = np.concatenate([surf_npz['pos'], pos])
    neg = np.concatenate([surf_npz['neg'], neg])

    results = {
        "pos": pos,
        "neg": neg
    }
    return results


# Main
######

def generate_samples(args, datadir, dest, instances, pid=None):
    """Generate data samples for the given meshes."""
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
            iprint(f"Generating for shape {i+1}/{n_shapes}...")

        mesh = trimesh.load(os.path.join(datadir, "meshes", instance + ".obj"))
        destdir = os.path.join(dest, instance)
        os.makedirs(destdir, exist_ok=True)

        ## 3D samples
        # Surface (with normals)
        if not args.skip_surf:
            sample_fn = os.path.join(destdir, "surface.npy")
            if args.overwrite or not os.path.isfile(sample_fn):
                samples = sample_surface(mesh, args.n_samples, dirty_mesh=args.mesh_to_sdf)
                np.save(sample_fn, samples)
        
        ## SDF samples
        # Near-surface
        sample_fn = os.path.join(destdir, "nearsurface.npz")
        if args.overwrite or not os.path.isfile(sample_fn):
            results = sample_sdf_nearsurface(mesh, args.n_samples, dirty_mesh=args.mesh_to_sdf)
            np.savez(sample_fn, **results)

        # Uniform
        sample_fn = os.path.join(destdir, "uniform.npz")
        if args.overwrite or not os.path.isfile(sample_fn):
            results = sample_sdf_uniform(mesh, args.n_samples, dirty_mesh=args.mesh_to_sdf)
            np.savez(sample_fn, **results)
        
        # DeepSDF-like samples (~95% near-surface, 5% uniform)
        sample_fn = os.path.join(destdir, "deepsdf.npz")
        if args.overwrite or not os.path.isfile(sample_fn):
            results = make_deepsdf_samples(os.path.join(destdir, "nearsurface.npz"),
                                           os.path.join(destdir, "uniform.npz"),
                                           n_uniform=args.n_samples // 10)
            np.savez(sample_fn, **results)
            
    iprint("Done.")


def main(args):
    """Launch the (parallel) generation of data samples."""
    np.random.seed(args.seed)

    datadir = args.datadir
    filenames = sorted(os.listdir(os.path.join(datadir, "meshes")))
    filenames = [fn for fn in filenames if not fn.startswith(('.', '_'))]
    instances = [os.path.splitext(fn)[0] for fn in filenames]
    instances.sort()
    dest = os.path.join(datadir, "samples")
    os.makedirs(dest, exist_ok=True)

    if args.nproc == 0:
        generate_samples(args, datadir, dest, instances)
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
            p = Process(target=generate_samples, args=(args, datadir, dest, chunk), kwargs={"pid": i})
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    print(f"Results saved in {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data samples for meshes.")
    parser.add_argument("datadir", type=str, help="directory of the dataset")

    parser.add_argument("--mesh-to-sdf", action='store_true', help="use the mesh-to-sdf package to compute the SDF and surface samples (useful for dirty meshes, e.g., unprocessed ShapeNet)")
    parser.add_argument("-n", "--n-samples", default=250000, type=int, help="number of samples to generate per sampling type")
    parser.add_argument("--nproc", default=0, type=int, help="number of processes to create to compute in parallel, give 0 to use main process (default: 0)")
    parser.add_argument("--skip-surf", action='store_true', help="do not compute surface samples")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing results, if any")
    parser.add_argument("--seed", default=0, help="seed for the RNGs")

    args = parser.parse_args()

    if args.mesh_to_sdf:
        from mesh_to_sdf import mesh_to_sdf, get_surface_point_cloud
        os.environ['PYOPENGL_PLATFORM'] = 'egl'

    main(args)