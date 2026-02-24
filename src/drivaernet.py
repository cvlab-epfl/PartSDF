"""
Module for DrivaerNet++ related utilities.

See https://github.com/Mohamedelrefaie/DrivAerNet
"""

import numpy as np
import trimesh


# Normalization from DrivAerNet++ space to [-0.9, 0.9]^3 cube
# with x' = s * (R @ x + T)
T = np.array([(4.1 - 1.05) / 2, 
              -0.7, 
              0.0])
R = np.array([[-1, 0, 0], 
              [ 0, 0, 1], 
              [ 0, 1, 0]])
S = 2 * 0.9 / (4.1 + 1.05)

# Name of the annotation parts corresponding to the back of the car for all 3 car types
BACK_ANNOTS = {
    "E": ["body_2", "windows_1", "body_roof", "body_tail_1"],
    "F": ["body_1", "windows_1", "body_roof_1", "body_tail_1"],
    "N": ["body_2", "windows_1", "body_roof_1", "body_tail_1"],
}


def load_clean(path):
    """
    Load a DrivaerNet++ mesh and clean it (normalization + face reorientation).
    """
    mesh = trimesh.load(path)
    mesh = normalize_drivaernet(mesh)
    mesh = reorient(mesh, drivaernet_world=False)
    return mesh

def load_clean_bodyback(mesh_path, annot_path, back_type, return_mesh=False):
    """
    Load a DrivaerNet++ example by parts (body / back separation) and normalize it.

    Args:
        mesh_path: path to the DrivaerNet++ mesh file.
        annot_path: path to the DrivaerNet++ annotation file.
        back_type: car type, one of "E", "F", or "N". (or the full instance name)
        return_mesh: whether to also return the full mesh.
    """
    mesh = trimesh.load(mesh_path)
    annot = trimesh.load(annot_path)
    parts_fit = fit_annot_to_mesh(annot.geometry, mesh)
    # Normalize / clean the meshes into our world space
    mesh = reorient(normalize_drivaernet(mesh), drivaernet_world=False)
    parts_fit = {k: normalize_drivaernet(p) for k, p in parts_fit.items()}
    # Separate body and back
    body, back = split_body_back(back_type, parts_fit, drivaernet_world=False)
    wheels = split_body_wheels(mesh, drivaernet_world=False)[1:]
    parts = [body, back] + wheels
    if return_mesh:
        return parts, mesh
    else:
        return parts


# Normalization
###############

_R_transform = np.concatenate([R, np.zeros((3, 1))], axis=1)
_R_transform = np.concatenate([_R_transform, np.array([[0, 0, 0, 1]])], axis=0)
def normalize_drivaernet(mesh):
    """Normalize mesh/points from DrivaerNet++ space to [-0.9, 0.9]^3 cube.

    Args:
        mesh: a trimesh object or a (N, 3) array of points.

    Returns:
        Normalized trimesh object or points.
    """
    if isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.copy()
        mesh.apply_transform(_R_transform)
        mesh.apply_translation(T)
        mesh.apply_scale(S)
        return mesh
    else:  # assume it's a (N, 3) array of points
        points = mesh @ R.T
        points = points + T
        points = points * S
        return points

def denormalize_drivaernet(mesh):
    """Denormalize mesh/points from [-0.9, 0.9]^3 cube to DrivaerNet++ space.

    Args:
        mesh: a trimesh object or a (N, 3) array of points.
    
    Returns:
        Denormalized trimesh object or points.
    """
    if isinstance(mesh, trimesh.Trimesh):
        mesh.apply_scale(1 / S)
        mesh.apply_translation(-T)
        mesh.apply_transform(_R_transform.T)
        return mesh
    else:  # assume it's a (N, 3) array of points
        points = mesh * (1 / S)
        points = points - T
        points = points @ R
        return points

def fit_annot_to_mesh(annot, mesh):
    """Rescale and translate annotation mesh to fit the DrivaerNet++ mesh.
    Must be in the original DrivaerNet++ space!

    Args:
        annot: dict of trimesh objects of annotation.
        mesh: a trimesh object of DrivaerNet++.
    
    Returns:
        Rescaled and translated annotation mesh.
    """
    # Annotations are the half body without wheels
    body = split_body_wheels(mesh)[0]
    # Rescale the annotations from mm to m
    annot = {k: a.copy().apply_scale(1e-3) for k, a in annot.items()}
    # Rescale like the (half-)body
    scale = body.extents / trimesh.util.concatenate(annot.values()).extents
    scale[1] /= 2  # annots are only half a car along the y-axis
    annot = {k: a.apply_scale(scale) for k, a in annot.items()}
    # Translate to the body center
    translation = body.bounds.mean(0) - trimesh.util.concatenate(annot.values()).bounds.mean(0)
    translation[1] = 0 - trimesh.util.concatenate(annot.values()).bounds[1, 1]  # annots are only half a car
    annot = {k: a.apply_translation(translation) for k, a in annot.items()}
    return annot
    

# Face orientation
##################

def reorient(mesh, drivaernet_world=True):
    """
    Reorient faces of DrivaerNet++ mesh to have outward normals.
    Only the left wheels are correctly oriented in the original meshes.
    """
    parts = split_body_wheels(mesh, drivaernet_world=drivaernet_world)
    parts = reorient_parts(parts)
    return trimesh.util.concatenate(parts)

def reorient_parts(parts):
    """
    Reorient faces of DrivaerNet++ mesh parts to have outward normals.
    Only the left wheels are correctly oriented in the original meshes.

    The part order is [body, left front wheel, right front wheel, left rear wheel, right rear wheel].
    """
    for i, part in enumerate(parts):
        if i in [1, 3]:  # left wheels
            continue
        part.invert()
    return parts


# Mesh parts
############

def split_body_wheels(mesh, drivaernet_world=True):
    """
    Split DrivaerNet++ mesh into body and wheels parts.
    Wheel order: FL, FR, RL, RR.

    Args:
        mesh: a trimesh object of DrivaerNet++.
        drivaernet_world: whether the mesh is in DrivaerNet++ world space,
            or normalized space.
    
    Returns:
        list of 5 meshes.
    """
    parts = mesh.split()
    assert len(parts) == 5, "Expected 5 parts (body + 4 wheels)."

    # Identify the body as the largest part (bbox-wise)
    body_idx = np.argmax([np.prod(p.extents) for p in parts])
    body = parts[body_idx]
    wheels = [parts[i] for i in range(5) if i != body_idx]

    # Sort wheels by position (x, y) for drivaernet world, (-x, z) otherwise
    # Front then rear wheels
    wheels.sort(key=lambda w: w.bounds.mean(0)[0], 
                reverse=not drivaernet_world)
    # Left then right wheels in each axle
    ax = 1 if drivaernet_world else 2
    if wheels[0].bounds.mean(0)[ax] > wheels[1].bounds.mean(0)[ax]:
        wheels[0], wheels[1] = wheels[1], wheels[0]
    if wheels[2].bounds.mean(0)[ax] > wheels[3].bounds.mean(0)[ax]:
        wheels[2], wheels[3] = wheels[3], wheels[2]
    
    return [body] + wheels

def split_body_back(back_type, annot, drivaernet_world=True):
    """
    Split DrivaerNet++ annotation parts into body and back parts.

    Args:
        back_type: car type, one of "E", "F", or "N". (or the full instance name)
        annot: dict of trimesh objects, named as in the annotation STL file.
        drivaernet_world: whether the annots are in DrivaerNet++ or normalized space.
    
    Returns:
        body and back meshes.
    """
    back_type = back_type[0]  # in case full instance name is given
    body, back = [], []
    for key, part in annot.items():
        if key in BACK_ANNOTS[back_type]:
            back.append(part)
        else:
            body.append(part)
    body = trimesh.util.concatenate(body)
    back = trimesh.util.concatenate(back)
    # Mirror them as the annotations are given for the left half of the car
    body2 = body.copy()
    back2 = back.copy()
    ax = 1 if drivaernet_world else 2
    body2.vertices[:, ax] = -body2.vertices[:, ax]
    back2.vertices[:, ax] = -back2.vertices[:, ax]
    body2.invert()
    back2.invert()
    body = trimesh.util.concatenate([body, body2])
    back = trimesh.util.concatenate([back, back2])
    return body, back