"""
Fit primitives to the parts.

Note: can use the convex hull of the parts to fit the primitives.
"""

import os, os.path
import sys
import argparse
import json
from math import ceil
from multiprocessing import Process

import numpy as np
from scipy import optimize as opt
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
import trimesh, trimesh.creation, trimesh.util

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.primitives import (
    matrix_to_quaternion, quaternion_to_matrix, standardize_quaternion,
    mesh_cuboid, mesh_cylinder, sample_unit_cuboid, sample_unit_cylinder
)


# Utils
#######

def load_parts(partdir, instance, n_parts):
    parts = []
    for i in range(n_parts):
        fn = os.path.join(partdir, "meshes", instance, f"part{i}.obj")
        if os.path.isfile(fn):
            parts.append(trimesh.load(fn))
        else:
            parts.append(trimesh.Trimesh())
    return parts

# Copying the function from src/metric.py to avoid any import that may need GPU
def chamfer_distance(pc1, pc2, square_dist=True):
    """
    Compute the symmetric L2-Chamfer Distance between the two point clouds.
    
    Args:
    -----
    pc1, pc2: (N, 3) arrays
        The point clouds to compare.
    square_dist: bool (default=True)
        If True, compute the squared distance.
    
    Returns:
    --------
    chamfer: float
        The symmetric L2-Chamfer distance.
    """
    tree1 = KDTree(pc1)
    dist1, _ = tree1.query(pc2)
    if square_dist:
        dist1 = np.square(dist1)
    chamfer2to1 = np.mean(dist1)

    tree2 = KDTree(pc2)
    dist2, _ = tree2.query(pc1)
    if square_dist:
        dist2 = np.square(dist2)
    chamfer1to2 = np.mean(dist2)

    return chamfer2to1 + chamfer1to2


# Primitive
###########

def fit_primitive(dist_fn, param0, mesh, n_samples=10_000, constraints=[], **optimize_kwargs):
    """Fit a primitive shape to a mesh."""
    # Sample points
    samples = mesh.sample(n_samples)
    # Optimize primitive
    result = opt.minimize(lambda x: dist_fn(samples, x).mean(), param0, method='SLSQP',
                          constraints=constraints, **optimize_kwargs)
    if not result.success:
        print(result.message)
    return result.x


# Cuboid
########

def init_cuboid(part, init_quaternion=np.array([1., 0., 0., 0.])):
    """Get the convex hull of the part and its initial cuboid parameters."""
    convex_hull = part.convex_hull
    scale = convex_hull.extents
    t = convex_hull.bounds.mean(0)
    quat = init_quaternion.copy()
    return convex_hull, scale, t, quat

def chamfer_distance_cuboid(samples, cuboid_samples, scale, translation, quaternion):
    """Compute the chamfer distance between the samples and cuboid samples."""
    # Transform the cuboid samples
    cuboid_samples = scale / 2 * cuboid_samples @ quaternion_to_matrix(quaternion).T + translation
    return chamfer_distance(samples, cuboid_samples)

def fit_cuboid(convex_hull, scale, translation, quaternion, n_samples=10_000, fixed_rotation=False, 
               **optimize_kwargs):
    """Fit a cuboid to a part."""
    cuboid_samples = sample_unit_cuboid(n_samples)
    param0 = np.concatenate([scale, translation, quaternion])
    if fixed_rotation:
        constraints = [  # fixed quaternion
            {'type': 'eq', 'fun': lambda x: x[6] - quaternion[0]},
            {'type': 'eq', 'fun': lambda x: x[7] - quaternion[1]},
            {'type': 'eq', 'fun': lambda x: x[8] - quaternion[2]},
            {'type': 'eq', 'fun': lambda x: x[9] - quaternion[3]},
        ]
    else:
        constraints = [
            {'type': 'eq', 'fun': lambda x: (x[6:10] ** 2).sum() - 1},  # unit quaternion
        ]
    param1 = fit_primitive(lambda x, p: chamfer_distance_cuboid(x, cuboid_samples, p[0:3], p[3:6], p[6:10]),
                           param0, convex_hull, n_samples, constraints=constraints, **optimize_kwargs)
    scale, translation, quaternion = param1[0:3], param1[3:6], param1[6:10]
    return scale, translation, quaternion

# Re-orient the cuboid

# First, 6 possible orientation for the first axis
__matrices1 = [Rotation.from_euler('z', angle, degrees=True).as_matrix() for angle in (0, 90, 180, -90)] + \
              [Rotation.from_euler('y', angle, degrees=True).as_matrix() for angle in (-90, 90)]
# Then, 4 possible orientations for the second axis
__matrices2 = [Rotation.from_euler('x', angle, degrees=True).as_matrix() for angle in (0, 90, 180, -90)]
def possible_quaternions(quaternion):
    """Return the 24 possible quaternions representating all possible orientations of the cuboid."""
    matrix = quaternion_to_matrix(quaternion)
    # Combine all of them
    matrix = np.array([matrix @ m1 @ m2 for m1 in __matrices1 for m2 in __matrices2])
    return standardize_quaternion(matrix_to_quaternion(matrix))

def arg_closest_quat(ref, quats):
    """Return the index of the closest quaternion to the reference quaternion."""
    dist = np.arccos(np.dot(ref, quats.T))
    dist = np.where(dist > np.pi / 2, np.pi - dist, dist)
    return np.argmin(dist)

# Scale permutation due to re-orientations
__scale_perm1 = np.array([
    (0, 1, 2), (0, 2, 1), (0, 1, 2), (0, 2, 1),  # 0, 90, 180, -90 around z
    (2, 1, 0), (2, 1, 0),                        # -90, 90 around y
])
__scale_perm2 = np.array([
    (0, 1, 2), (0, 2, 1), (0, 1, 2), (0, 2, 1),  # 0, 90, 180, -90 around x
])
__scale_perm = np.stack([s1[s2] for s1 in __scale_perm1 for s2 in __scale_perm2])
def permute_scale(scale, idx):
    """Return the permuted scale according to the orientation index."""
    return scale[__scale_perm[idx]]

# All together
def reorient_cuboid(ref_quaternion, quaternion, scale):
    """Re-orient the cuboid to the reference quaternion."""
    all_quats = possible_quaternions(quaternion)
    idx = arg_closest_quat(ref_quaternion, all_quats)
    quaternion = all_quats[idx]
    scale = permute_scale(scale, idx)
    return quaternion, scale


# Cylinder
##########

def init_cylinder(part, init_quaternion=np.array([1., 0., 0., 0.])):
    """Get the convex hull of the part and its initial cylinder parameters."""
    convex_hull = part.convex_hull
    radius = convex_hull.extents[::2].mean() / 2
    height = convex_hull.extents[1]
    t = convex_hull.bounds.mean(0)
    quat = init_quaternion.copy()
    return convex_hull, radius, height, t, quat

def chamfer_distance_cylinder(samples, cylinder_samples, radius, height, translation, quaternion):
    """Compute the chamfer distance between the samples and cylinder samples."""
    # Transform the cuboid samples
    cylinder_samples = [radius, radius, height/2] * cylinder_samples @ quaternion_to_matrix(quaternion).T + translation
    return chamfer_distance(samples, cylinder_samples)

def fit_cylinder(convex_hull, radius, height, translation, quaternion, n_samples=10_000, fixed_rotation=False,
                 **optimize_kwargs):
    """Fit a cylinder to a part."""
    cylinder_samples = sample_unit_cylinder(n_samples)
    param0 = np.concatenate([np.array([radius, height]), translation, quaternion])
    if fixed_rotation:
        constraints = [  # fixed quaternion
            {'type': 'eq', 'fun': lambda x: x[5] - quaternion[0]},
            {'type': 'eq', 'fun': lambda x: x[6] - quaternion[1]},
            {'type': 'eq', 'fun': lambda x: x[7] - quaternion[2]},
            {'type': 'eq', 'fun': lambda x: x[8] - quaternion[3]},
        ]
    else:
        constraints = [
            {'type': 'eq', 'fun': lambda x: (x[5:9] ** 2).sum() - 1},  # unit quaternion
        ]
    param1 = fit_primitive(lambda x, p: chamfer_distance_cylinder(x, cylinder_samples, p[0], p[1], p[2:5], p[5:9]),
                           param0, convex_hull, n_samples, constraints=constraints, **optimize_kwargs)
    radius, height, translation, quaternion = param1[0], param1[1], param1[2:5], param1[5:9]
    return radius, height, translation, quaternion

# Re-orient the cylinder
def reorient_cylinder(ref_quaternion, quaternion):
    """Re-orient the cylinder to the reference quaternion."""
    ref_matrix = quaternion_to_matrix(ref_quaternion)
    matrix = quaternion_to_matrix(quaternion)
    # Re-orient the flat faces of the cylinder (scalar product of the z-axes should be >=0)
    if np.dot(ref_matrix[:, 2], matrix[:, 2]) < 0:
        matrix = matrix @ Rotation.from_euler('x', 180, degrees=True).as_matrix()
    # Re-orient the annular face of the cylinder (scalar product of the x-axes should be maximum)
    ref_xaxis = ref_matrix[:, 0]
    xaxis = matrix[:, 0]
    zaxis = matrix[:, 2]
    def x_scalar_product(theta):
        _xaxis = Rotation.from_rotvec(theta * zaxis, degrees=True).as_matrix() @ xaxis
        return - np.dot(ref_xaxis.reshape(-1), _xaxis.reshape(-1))
    result = opt.minimize(x_scalar_product, 0., bounds=[(-180, 180)])
    if not result.success:
        print(result.message)
    theta = result.x[0]
    matrix = matrix @ Rotation.from_euler('z', theta, degrees=True).as_matrix()
    return standardize_quaternion(matrix_to_quaternion(matrix))


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
        parts = load_parts(partdir, instance, args.n_parts)
        paramdir = os.path.join(partdir, "parameters", instance)
        primdir = os.path.join(partdir, "primitives", instance)
        os.makedirs(paramdir, exist_ok=True)
        os.makedirs(primdir, exist_ok=True)

        # Verify if already computed
        if not args.overwrite and \
           os.path.exists(os.path.join(paramdir, instance, "quaternions.npy")) and \
           os.path.exists(os.path.join(paramdir, instance, "translations.npy")) and \
           os.path.exists(os.path.join(paramdir, instance, "scales.npy")):
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
                convex_hull, scale, translation, quaternion = init_cuboid(part, ref_quaternion)
                scale, translation, quaternion = fit_cuboid(convex_hull, scale, translation, quaternion, 
                                                            n_samples=args.pc_size, fixed_rotation=args.fixed_rotation[j])
                if args.reorient[j]:
                    quaternion, scale = reorient_cuboid(ref_quaternion, quaternion, scale)
                cuboid = mesh_cuboid(scale, translation, quaternion)
                # Save the results
                np.savez(os.path.join(paramdir, f"part{j}_cuboid.npz"), scale=scale,
                         translation=translation, quaternion=quaternion)
                cuboid.export(os.path.join(primdir, f"part{j}_cuboid.obj"))
                scale /= 2  # half-lengths
            elif args.primitive[j] == "cylinder":
                convex_hull, radius, height, translation, quaternion = init_cylinder(part, ref_quaternion)
                radius, height, translation, quaternion = fit_cylinder(convex_hull, radius, height, translation, quaternion, 
                                                                       n_samples=args.pc_size, fixed_rotation=args.fixed_rotation[j])
                if args.reorient[j]:
                    quaternion = reorient_cylinder(ref_quaternion, quaternion)
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
    filenames = sorted(os.listdir(os.path.join(datadir, "meshes")))
    filenames = [fn for fn in filenames if not fn.startswith(('.', '_'))]
    instances = [os.path.splitext(fn)[0] for fn in filenames]
    instances.sort()
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
    parser.add_argument("n_parts", type=int, help="max number of parts used")
    
    parser.add_argument("--partdir", default="parts", type=str, help="name of the subdir where the part decomposition will be saved")

    # Spec file describing the primitives
    parser.add_argument("--specs", default=None, type=str, help="specs file with primitives argument, if used, will overwrite command line arguments")
    # OR manually
    parser.add_argument("--fixed-rotation", action="store_true", help="fix the rotation of the primitive to the initial orientation (:=the reference)")
    parser.add_argument("--no-reorient", action="store_false", dest="reorient", help="do not re-orient the primitives to a reference to break symmetries")
    parser.add_argument("--primitive", default="cuboid", choices=["cuboid", "cylinder"], help="primitive to use (default: cuboid)")
    parser.add_argument("--ref-quaternion", default=[1., 0., 0., 0.], type=float, nargs=4, help="reference quaternion for the orientation (default: [1., 0., 0., 0.])")
    
    parser.add_argument("--pc-size", default=10_000, type=int, help="size of the part point cloud to fit primitives (default: 10_000)")
    parser.add_argument("--nproc", default=0, type=int, help="number of processes to create to compute in parallel, give 0 to use main process (default: 0)")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing results, if any")
    parser.add_argument("--seed", default=0, type=int, help="seed for the RNGs")

    args = parser.parse_args()

    # Load specs, if given
    if args.specs is not None:
        with open(args.specs, 'r') as f:
            specs = json.load(f)
        args.fixed_rotation = specs.get("fixed_rotation", args.fixed_rotation)
        args.reorient = specs.get("reorient", args.reorient)
        args.primitive = specs.get("primitive", args.primitive)
        args.ref_quaternion = specs.get("ref_quaternion", args.ref_quaternion)
    # Duplicate single arguments for each part
    if not isinstance(args.fixed_rotation, list):
        args.fixed_rotation = [args.fixed_rotation] * args.n_parts
    if not isinstance(args.reorient, list):
        args.reorient = [args.reorient] * args.n_parts
    if not isinstance(args.primitive, list):
        args.primitive = [args.primitive] * args.n_parts
    if not isinstance(args.ref_quaternion[0], list):
        args.ref_quaternion = [args.ref_quaternion] * args.n_parts

    main(args)