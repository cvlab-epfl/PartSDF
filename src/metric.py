"""
Evaluation metrics for reconstructed shapes.
"""

import numpy as np
from scipy.spatial import KDTree
import trimesh
import igl

from .image_concistency import image_consistency
from .utils import make_grid, get_winding_number_mesh


def chamfer_distance(pc1, pc2, square_dist=True, return_idx=False,
                     val1=None, val2=None, val_fn=np.abs):
    """
    Compute the symmetric L2-Chamfer Distance between the two point clouds.

    Can optionally gives other values to be compared on the points.
    
    Args:
    -----
    pc1, pc2: (N, 3) arrays
        The point clouds to compare.
    square_dist: bool (default=True)
        If True, compute the squared distance.
    return_idx: bool (default=False)
        If True, also return the indices of the closest points.
    val1, val2: (N, D) arrays (optional)
        If given, compare these values on the closest points.
    val_fn: callable (default=np.abs)
        The function to compare the values. Should take the difference as input.
    
    Returns:
    --------
    chamfer: float
        The symmetric L2-Chamfer distance.
    idx1, idx2: (N,) arrays (optional)
        The indices of the closest points.
    val_diff: float (optional)
        The average difference between the values on the closest points.
    """
    tree1 = KDTree(pc1)
    dist1, idx1 = tree1.query(pc2)
    if square_dist:
        dist1 = np.square(dist1)
    chamfer2to1 = np.mean(dist1)

    tree2 = KDTree(pc2)
    dist2, idx2 = tree2.query(pc1)
    if square_dist:
        dist2 = np.square(dist2)
    chamfer1to2 = np.mean(dist2)

    results = (chamfer2to1 + chamfer1to2,)
    if return_idx:
        results += (idx1, idx2)

    # Compare values on the points
    if val1 is not None and val2 is not None:
        val2to1 = np.mean(val_fn(val1[idx1] - val2))
        val1to2 = np.mean(val_fn(val2[idx2] - val1))
        results = results + (val2to1 + val1to2,)

    return results[0] if len(results) == 1 else results


def mesh_iou(mesh1, mesh2, N=256, bbox=None):
    """
    Compute volumetric IoU between the meshes using occupancy samples.

    Args:
    -----
    mesh1, mesh2: trimesh.base.Trimesh
        The meshes to compare.
    N: int or tuple of int (default=256)
        The resolution of the grid to compute occupancy on.
    bbox: (2, 3) array (optional)
        The bounding box to use for the grid. If None, take the bbox of both meshes.
    
    Returns:
    --------
    iou: float
        The volumetric IoU.
    """
    if mesh1.is_empty and mesh2.is_empty:
        return 1.0
    elif mesh1.is_empty or mesh2.is_empty:
        return 0.0

    if bbox is None:  # take a bbox including both meshes
        bbox = np.stack([np.minimum(mesh1.bounds[0], mesh2.bounds[0]), 
                         np.maximum(mesh1.bounds[1], mesh2.bounds[1])])
    
    # Compute occupancy on a regular grid
    xyz = make_grid(bbox, N).numpy()
    wn1, wn2 = get_winding_number_mesh(mesh1, xyz), get_winding_number_mesh(mesh2, xyz)
    occ1, occ2 = (wn1 >= 0.5), (wn2 >= 0.5)
    
    intersection = np.sum(occ1 & occ2)
    union = np.sum(occ1 | occ2)
    return intersection / union


# Part IoU
##########

def part_iou(parts1, parts2, N=256, bbox=None, mean=True, process1=False):
    """
    Compute volumetric IoU between the parts using occupancy samples.

    Args:
    -----
    parts1, parts2: list of trimesh.base.Trimesh
        The list of parts to compare. They need to be the same size, possibly with
        empty meshes!
    N: int or tuple of int (default=256)
        The resolution of the grid to compute occupancy on.
    bbox: (2, 3) array (optional)
        The bounding box to use for the grid. If None, take the bbox of both parts.
    mean: bool (default=True)
        If True, return the mean IoU over all parts.
    process1: bool (default=False)
        If True, process `parts1` as they might be non-watertight. This will compute
        the full shape occupancy, then label each interior point with the closest part.
    
    Returns:
    --------
    pIoU: float or np.array
        The volumetric part IoU.
    """
    if not process1:
        # Assumes watertight parts, directly compute their IoU
        piou = []
        for part1, part2 in zip(parts1, parts2):
            piou.append(mesh_iou(part1, part2, N, bbox))

    else:  # parts1 may not be watertight
        # Consider first full shape
        mesh1, mesh2 = trimesh.util.concatenate(parts1), trimesh.util.concatenate(parts2)
        if mesh1.is_empty and mesh2.is_empty:
            return 1.0
        elif mesh1.is_empty or mesh2.is_empty:
            return 0.0
        if bbox is None:  # take a bbox including both meshes
            bbox = np.stack([np.minimum(mesh1.bounds[0], mesh2.bounds[0]), 
                             np.maximum(mesh1.bounds[1], mesh2.bounds[1])])
        xyz = make_grid(bbox, N).numpy()
        
        # Get the (robust) per-part occupancy
        occ1 = robust_part_occ(mesh1, parts1, xyz)
        
        # Finally, get the per-part IoU
        piou = []
        for i, part2 in enumerate(parts2):
            occ1_empty = np.all(~occ1[i])
            if occ1_empty and part2.is_empty:
                piou.append(1.0)
            elif occ1_empty or part2.is_empty:
                piou.append(0.0)
            else:
                occ2 = (get_winding_number_mesh(part2, xyz) >= 0.5)
                intersection = np.sum(occ1[i] & occ2)
                union = np.sum(occ1[i] | occ2)
                piou.append(intersection / union)
    
    piou = np.array(piou)
    return piou.mean() if mean else piou

def _distance_to_mesh(points, mesh):
    """Return the squared distance of the points to the mesh. If empty, return +inf."""
    if not mesh.is_empty and points.size > 0:
        return igl.point_mesh_squared_distance(points, mesh.vertices, mesh.faces)[0].astype(np.float32)
    else:
        return np.full(points.shape[0], float('inf')).astype(np.float32)
    
def robust_part_occ(mesh, parts, xyz):
    """Compute the occupancy of each parts based of the full mesh."""
    # Get the full occupancy
    occ = (get_winding_number_mesh(mesh, xyz) >= 0.5)

    # Label each interior point with the closest part
    # Occupied points
    coord_idx = np.nonzero(occ)
    coords = xyz[coord_idx]
    # Find closest parts
    dists2 = np.stack([_distance_to_mesh(coords, parts[i]) for i in range(len(parts))], axis=0)
    closest = np.argmin(dists2, axis=0)
    # Per-part occupancy
    occ = np.zeros((len(parts),) + occ.shape, dtype=bool)
    for i in range(len(parts)):
        occ[i][tuple(coord_idx[j][closest == i] for j in range(3))] = True
    return occ

def _occ_iou(occ1, occ2):
    """Compute the IoU between two occupancy grids."""
    intersection = np.sum(occ1 & occ2)
    union = np.sum(occ1 | occ2)
    if union == 0:
        return 1.0
    return intersection / union

def part_iou_matching(parts1, parts2, N=256, bbox=None, mean=True, process1=False):
    """
    Compute volumetric IoU between the parts using occupancy samples.
    Assumes different part numbers, and tries to match them greedily (parts2 to parts1).

    Args:
    -----
    parts1, parts2: list of trimesh.base.Trimesh
        The list of parts to compare.
    N: int or tuple of int (default=256)
        The resolution of the grid to compute occupancy on.
    bbox: (2, 3) array (optional)
        The bounding box to use for the grid. If None, take the bbox of both parts.
    mean: bool (default=True)
        If True, return the mean IoU over all parts.
    process1: bool (default=False)
        If True, process `parts1` as they might be non-watertight. This will compute
        the full shape occupancy, then label each interior point with the closest part.
    
    Returns:
    --------
    pIoU: float or np.array
        The volumetric part IoU.
    """
    # Consider first full shape
    mesh1, mesh2 = trimesh.util.concatenate(parts1), trimesh.util.concatenate(parts2)
    if mesh1.is_empty and mesh2.is_empty:
        return 1.0
    elif mesh1.is_empty or mesh2.is_empty:
        return 0.0
    if bbox is None:  # take a bbox including both meshes
        bbox = np.stack([np.minimum(mesh1.bounds[0], mesh2.bounds[0]), 
                         np.maximum(mesh1.bounds[1], mesh2.bounds[1])])
    xyz = make_grid(bbox, N).numpy()

    # Get the (robust) per-part occupancy for parts1
    if not process1:
        # Assumes watertight parts, directly compute their occupancy
        occ1 = [(get_winding_number_mesh(part, xyz) >= 0.5) if not part.is_empty else np.zeros(xyz.shape[:-1], dtype=bool)
                for part in parts1]
        occ1 = np.stack(occ1)
    else:  # parts1 may not be watertight
        occ1 = robust_part_occ(mesh1, parts1, xyz)
    
    # Get the per-part occupancy for parts2
    occ2 = [(get_winding_number_mesh(part, xyz) >= 0.5) if not part.is_empty else np.zeros(xyz.shape[:-1], dtype=bool)
            for part in parts2]
    occ2 = np.stack(occ2)

    # Greedily match parts2 to parts1 based on maximal part-IoU
    occ2_matched = np.zeros_like(occ1)
    for i in range(len(occ2)):
        if occ2[i].sum() == 0:
            continue
        # Find the index of part in occ1 to match, based on maximum IoU
        idx = np.argmax([_occ_iou(occ1[j], occ2[i]) for j in range(len(occ1))])
        occ2_matched[idx] |= occ2[i]

    # Finally, get the per-part IoU
    piou = [_occ_iou(occ1[i], occ2_matched[i]) for i in range(len(occ1))]
    piou = np.array(piou)
    return piou.mean() if mean else piou


def mmd(distances):
    """Minimum Matching Distance between generated shapes (rows) and GT shapes (columns)."""
    return distances.min(0).mean()

def coverage(distances):
    """Coverage score between generated shapes (rows) and GT shapes (columns)."""
    cov = np.unique(distances.argmin(1))  # unique GT shapes matched
    return len(cov) / distances.shape[1]