"""
Module related to geometric primitives.
"""

import numpy as np
import torch
import torch.nn.functional as F
import trimesh, trimesh.creation


##############
# Transforms #
##############
# Quaternion <-> Matrix is taken from pytorch3d. The code is copied to avoid needing the whole package.

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    is_ndarray = False
    if isinstance(quaternions, np.ndarray):
        quaternions = torch.from_numpy(quaternions)
        is_ndarray = True
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    o = o.reshape(quaternions.shape[:-1] + (3, 3))
    if is_ndarray:
        o = o.numpy()
    return o


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a quaternion to a standard form: one in which the real
    part is non negative and the norm is unit.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    if isinstance(quaternions, np.ndarray):
        quaternions = quaternions / np.linalg.norm(quaternions, axis=-1, keepdims=True)
        return np.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)
    quaternions = quaternions / torch.norm(quaternions, dim=-1, keepdim=True)
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    is_ndarray = False
    if isinstance(matrix, np.ndarray):
        matrix = torch.from_numpy(matrix)
        is_ndarray = True
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    out = standardize_quaternion(out)
    if is_ndarray:
        out = out.numpy()
    return out


def slerp_quaternion(q0, q1, t):
    """Spherical linear interpolation between two quaternions."""
    cos_theta = (q0 * q1).sum(-1, keepdims=True)
    theta = torch.acos(cos_theta)
    q = torch.where(cos_theta > 0.995,
        (1 - t) * q0 + t * q1,  # linear interpolation when the quaternions are close
        (torch.sin((1 - t) * theta) * q0 + torch.sin(t * theta) * q1) / torch.sin(theta)
    )
    return q


##############
# Primitives #
##############

def sqrd_dist_to_cuboid(xyz, scale, translation, quaternion, _sdf=False):
    """Return the squared distance of the points in xyz to the cuboid defined by the parameters."""
    # NOTE: currently only numpy based
    # Rotation quaternion to matrix
    R = quaternion_to_matrix(quaternion)
    # Transform the points into the cuboid's frame
    xyz = (xyz - translation) @ R
    # Compute the distances to the 3 faces and take the minimum (using symmetries and projections)
    xyz = np.abs(xyz)
    _xyz = np.concatenate([np.minimum(xyz[..., 0:1], scale[0] / 2), 
                           np.minimum(xyz[..., 1:2], scale[1] / 2), 
                           np.minimum(xyz[..., 2:3], scale[2] / 2)], axis=-1)
    p_x, p_y, p_z = _xyz.copy(), _xyz.copy(), _xyz.copy()
    p_x[..., 0] = scale[0] / 2
    p_y[..., 1] = scale[1] / 2
    p_z[..., 2] = scale[2] / 2
    d2_x = ((xyz - p_x) ** 2).sum(-1)
    d2_y = ((xyz - p_y) ** 2).sum(-1)
    d2_z = ((xyz - p_z) ** 2).sum(-1)
    d2 = np.minimum(d2_x, np.minimum(d2_y, d2_z))
    if _sdf:
        d2 = d2 ** 0.5
        # Inside/outside sign
        inside = np.all(xyz < scale / 2, axis=-1)
        d2 = np.where(inside, -d2, d2)
    return d2

def sqrd_dist_to_cylinder(xyz, radius, height, translation, quaternion, _sdf=False):
    """Return the squared distance of the points in xyz to the cylinder defined by the parameters."""
    # NOTE: currently only numpy based
    # Rotation quaternion to matrix
    R = quaternion_to_matrix(quaternion)
    # Transform the points into the cylinder's frame
    xyz = (xyz - translation) @ R
    # Compute the distances to both faces and take the minimum (using symmetries and projections)
    xyz = np.abs(xyz)
    xy_norm = np.linalg.norm(xyz[..., :2], axis=-1)
    p_ring = np.concatenate([xyz[..., :2] / xy_norm[..., None] * radius, 
                             np.minimum(xyz[..., 2:3], height/2)], axis=-1)
    p_disk = np.concatenate([xyz[..., :2] * np.minimum(radius / xy_norm, 1)[..., None], 
                             np.full_like(xyz[..., 2:3], height/2)], axis=-1)
    d2_ring = ((xyz - p_ring) ** 2).sum(-1)
    d2_disk = ((xyz - p_disk) ** 2).sum(-1)
    d2 = np.minimum(d2_ring, d2_disk)
    if _sdf:
        d2 = d2 ** 0.5
        # Inside/outside sign
        inside = (xy_norm < radius) & (xyz[..., 2] < height / 2)
        d2 = np.where(inside, -d2, d2)
    return d2


def sdf_cuboid(xyz, scale, translation, quaternion):
    """Return the signed distance of the points in xyz to the cuboid defined by the parameters."""
    return sqrd_dist_to_cuboid(xyz, scale, translation, quaternion, _sdf=True)

def sdf_cylinder(xyz, radius, height, translation, quaternion):
    """Return the signed distance of the points in xyz to the cylinder defined by the parameters."""
    return sqrd_dist_to_cylinder(xyz, radius, height, translation, quaternion, _sdf=True)


def mesh_cuboid(scale, translation=[0, 0, 0], quaternion=[1, 0, 0, 0], **kwargs):
    """"Create a cuboid mesh from the parameters."""
    transform = np.eye(4)
    transform[:3, :3] = quaternion_to_matrix(quaternion)
    transform[:3, 3] = translation
    return trimesh.creation.box(extents=scale, transform=transform, **kwargs)

def mesh_cylinder(radius, height, translation=[0, 0, 0], quaternion=[1, 0, 0, 0], **kwargs):
    """"Create a cylinder mesh from the parameters."""
    transform = np.eye(4)
    transform[:3, :3] = quaternion_to_matrix(quaternion)
    transform[:3, 3] = translation
    return trimesh.creation.cylinder(radius=radius, height=height, transform=transform, **kwargs)


#########
# Misc. #
#########

def inv_transform(xyz, R=None, t=None, s=None):
    """Apply the inverse of the transformation to the points."""
    if t is not None:
        xyz = xyz - t
    if R is not None:
        if R.shape[-1] == 4:
            R = quaternion_to_matrix(R)
        xyz = torch.einsum("...i,...ij->...j", xyz, R)
    if s is not None:
        xyz = xyz / s
    return xyz

def inv_transform_parts(xyz, R=None, t=None, s=None):
    """Apply the inverse of the parts transformation to the points."""
    if t is not None:
        xyz = xyz - t
    if R is not None:
        if R.shape[-1] == 4:
            R = quaternion_to_matrix(R)
        if xyz.shape[-2] == 1:  # duplicate the points if needed
            xyz.expand(R.shape[:-1])
        xyz = torch.einsum("...pi,...pij->...pj", xyz, R)
    if s is not None:
        xyz = xyz / s
    return xyz


def sample_unit_cuboid(n_samples):
    """Sample points on a unit cuboid [-1, 1]^3."""
    # Sample the faces (order: -x, +x, -y, +y, -z, +z)
    faces_idx = np.random.choice(6, n_samples, replace=True)
    # Sample points on a [-1, 1]^2 square
    samples = np.random.rand(n_samples, 2) * 2 - 1
    # Create offsets for each faces (their "origin", the center of the face)
    offsets = np.array([
        [-1,  0,  0],  # -x
        [ 1,  0,  0],  # +x
        [ 0, -1,  0],  # -y
        [ 0,  1,  0],  # +y
        [ 0,  0, -1],  # -z
        [ 0,  0,  1],  # +z
    ])
    # Direction of the sampling variable
    dirs = np.array([
        [[0, 1, 0], [0, 0, 1]],  # -x
        [[0, 1, 0], [0, 0, 1]],  # +x
        [[1, 0, 0], [0, 0, 1]],  # -y
        [[1, 0, 0], [0, 0, 1]],  # +y
        [[1, 0, 0], [0, 1, 0]],  # -z
        [[1, 0, 0], [0, 1, 0]],  # +z
    ])
    # Compute the points
    points = offsets[faces_idx] + (samples[..., None] * dirs[faces_idx]).sum(1)
    return points

def sample_unit_cylinder(n_samples):
    """Sample points on a unit cylinder ~[-1, 1]^3."""
    # Sample the faces (order: top, bottom, side)
    faces_idx = np.random.choice(3, n_samples, replace=True)
    # Sample points on a [-1, 1]^2 square
    samples = np.random.rand(n_samples, 2)
    thetas = samples[:, 0] * 2 * np.pi
    # Compute the points
    points = np.zeros((n_samples, 3))
    # Top and bottom faces
    radius = np.sqrt(samples[:, 1])
    height = np.where(faces_idx == 0, 1, -1)
    points = np.where(faces_idx[..., None] == 2, points,
                      np.stack([radius * np.cos(thetas), radius * np.sin(thetas), height], axis=1))
    # Side face
    points = np.where(faces_idx[..., None] != 2, points,
                      np.stack([np.cos(thetas), np.sin(thetas), samples[:, 1] * 2 - 1], axis=1))
    return points