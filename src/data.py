"""
Data module to load SDF samples.
"""

import os.path
import json
from math import sqrt

import numpy as np
import igl
import torch
from torch.utils.data import Dataset


def remove_nans(samples):
    """Return samples with non-NaN SDF."""
    # Note: samples should be Nx4
    if isinstance(samples, np.ndarray):
        nan_indices = np.isnan(samples[:, 3])
    elif isinstance(samples, torch.Tensor):
        nan_indices = torch.isnan(samples[:, 3])
    return samples[~nan_indices, :]


def samples_from_array(pos, neg, n_samples, balance=True, full_samples=False):
    """Extract SDF samples from array of values."""
    if balance:
        neg = np.random.permutation(neg)[:n_samples//2]
        pos = np.random.permutation(pos)[:n_samples - len(neg)]  # have exactly n_samples
        samples = np.concatenate([pos, neg], 0)
    else:
        samples = np.concatenate([pos, neg], 0)
        samples = np.random.permutation(samples)[:n_samples]
    
    if full_samples:
        return samples

    # Break into input position and target sdf
    xyz = samples[:, 0:3]
    sdf = samples[:, 3:4]

    return xyz, sdf


def samples_from_tensor(pos, neg, n_samples, balance=True, full_samples=False):
    """Extract SDF samples from tensor of values."""
    if balance:
        neg_idx = torch.randperm(len(neg), device=neg.device)[:n_samples//2]
        pos_idx = torch.randperm(len(pos), device=pos.device)[:n_samples - len(neg_idx)]  # have exactly n_samples
        samples = torch.cat([pos[pos_idx], neg[neg_idx]], 0)
    else:
        samples = torch.cat([pos, neg], 0)
        samples_idx = torch.randperm(len(samples), device=samples.device)[:n_samples]
        samples = samples[samples_idx]
    
    if full_samples:
        return samples

    # Break into input position and target sdf
    xyz = samples[:, 0:3]
    sdf = samples[:, 3:4]

    return xyz, sdf


def samples_from_file(filename, n_samples, balance=True, full_samples=False):
    """Extract SDF samples from a file."""
    # Load the samples file
    npz = np.load(filename)

    return samples_from_array(npz['pos'], npz['neg'], n_samples, balance=balance, full_samples=full_samples)


class SampleDataset(Dataset):
    """
    Base class to implement a dataset of samples.
    """

    def __init__(self, datadir, split, n_samples, sampledir, samplefile):
        """
        Initialize the Dataset with a split of shapes.

        Parameters
        ----------
        datadir: str
            Path to the data source directory.
        split: list or str
            List of the shape names, or optionally the according JSON file.
        n_samples: int
            Number of samples per shape.
        sampledir: str
            Name of the sub-directory where the SDF samples resides.
        samplefile: str
            Name of the sample file to load.
        """
        super().__init__()
        self.datadir = datadir
        self.n_samples = n_samples
        self.sampledir = sampledir
        self.samplefile = samplefile

        if isinstance(split, str):
            with open(split) as f:
                split = json.load(f)
        # List the instances with dataset / shape name
        self.instances = split

        # List the filenames
        self.filenames = []
        for instance in self.instances:
            self.filenames.append(os.path.join(self.datadir, self.sampledir, instance, self.samplefile))
    

    def __len__(self):
        return len(self.instances)
    

    def __getitem__(self, idx):
        raise NotImplementedError("__getitem__() must be rewritten in children classes!")


class SdfDataset(SampleDataset):
    """
    Dataset of SDF samples.
    """

    def __init__(self, datadir, split, n_samples, sampledir="samples", samplefile="deepsdf.npz",
                 balance=True):
        """
        Initialize the Dataset with a split of shapes.

        Parameters
        ----------
        datadir: str
            Path to the data source directory.
        split: list or str
            List of the shape names, or optionally the according JSON file.
        n_samples: int
            Number of samples per shape.
        sampledir: str (default="samples")
            Name of the sub-directory where the SDF samples resides.
        samplefile: str (default="deepsdf.npz")
            Name of the sample file to load.
        balance: bool (default=True)
            If True, will balance the number of samples inside and outside.
        """
        super().__init__(datadir, split, n_samples, sampledir, samplefile)
        self.balance = balance
    

    def __getitem__(self, idx):
        xyz, sdf = samples_from_file(self.filenames[idx], self.n_samples, balance=self.balance)
        return idx, xyz, sdf


class SurfaceDataset(SampleDataset):
    """
    Dataset of surface samples.
    """

    def __init__(self, datadir, split, n_samples, sampledir="samples", samplefile="surface.npy",
                 return_idx=False, return_normals=False):
        """
        Initialize the Dataset with a split of shapes.

        Parameters
        ----------
        datadir: str
            Path to the data source directory.
        split: list or str
            List of the shape names, or optionally the according JSON file.
        n_samples: int
            Number of samples per shape.
        sampledir: str (default="samples")
            Name of the sub-directory where the samples resides.
        samplefile: str (default="surface.npy")
            Name of the sample file to load.
        return_idx: bool (default=False)
            If True, __getitem__() will also return the idx.
        return_normals: bool (default=False)
            If True, __getitem__() will also return the normals.
        """
        super().__init__(datadir, split, n_samples, sampledir, samplefile)
        self.return_idx = return_idx
        self.return_normals = return_normals


    def __getitem__(self, idx):
        samples = np.load(self.filenames[idx])
        samples = np.random.permutation(samples)[:self.n_samples]
        xyz = samples[:, 0:3]
        if self.return_normals:
            normals = samples[:, 3:6]
            if self.return_idx:
                return idx, xyz, normals
            return xyz, normals
        if self.return_idx:
            return idx, xyz
        return xyz


class UniformDataset(SampleDataset):
    """
    Dataset of uniform samples.
    """

    def __init__(self, datadir, split, n_samples, sampledir="samples", samplefile="uniform.npz",
                 return_idx=False, balance=False, only_pos=False, only_neg=False):
        """
        Initialize the Dataset with a split of shapes.

        Parameters
        ----------
        datadir: str
            Path to the data source directory.
        split: list or str
            List of the shape names, or optionally the according JSON file.
        n_samples: int
            Number of samples per shape.
        sampledir: str (default="samples")
            Name of the sub-directory where the samples resides.
        samplefile: str (default="uniform.npz")
            Name of the sample file to load.
        return_idx: bool (default=False)
            If True, __getitem__() will also return the idx.
        balance: bool (default=False)
            If True, will balance the number of samples inside and outside.
        only_pos, only_neg: bool (default=False)
            If True, will only return the positive/negative samples (outside/inside the shape).
            Both are exclusive.
        """
        super().__init__(datadir, split, n_samples, sampledir, samplefile)
        self.return_idx = return_idx
        self.balance = balance
        self.only_pos = only_pos
        self.only_neg = only_neg

        assert not (only_pos and only_neg), "only_pos and only_neg are exclusive!"


    def __getitem__(self, idx):
        npz = np.load(self.filenames[idx])
        if self.only_pos:
            samples = npz['pos']
            samples = np.random.permutation(samples)[:self.n_samples]
            xyz, sdf = samples[:, 0:3], samples[:, 3:4]
        elif self.only_neg:
            samples = npz['neg']
            samples = np.random.permutation(samples)[:self.n_samples]
            xyz, sdf = samples[:, 0:3], samples[:, 3:4]
        else:
            xyz, sdf = samples_from_array(npz['pos'], npz['neg'], self.n_samples, balance=self.balance)
        if self.return_idx:
            return idx, xyz, sdf
        return xyz, sdf


class PartSdfDataset(SampleDataset):
    """
    Dataset of part SDF samples.
    """

    def __init__(self, datadir, split, n_samples, sampledir="samples", samplefile="deepsdf.npz",
                 part_sampledir="parts_cuboid20/sample_labels", part_samplefile=None, balance=True):
        """
        Initialize the Dataset with a split of shapes.

        Parameters
        ----------
        datadir: str
            Path to the data source directory.
        split: list or str
            List of the shape names, or optionally the according JSON file.
        n_samples: int
            Number of samples per shape.
        sampledir: str (default="samples")
            Name of the sub-directory where the SDF samples resides.
        samplefile: str (default="deepsdf.npz")
            Name of the sample file to load.
        part_sampledir: str (default="parts_cuboid20/sample_labels")
            Name of the sub-directory where the part samples resides.
        part_samplefile: str (default=None)
            Name of the part sample file to load, if different from `samplefile`.
        balance: bool (default=True)
            If True, will balance the number of samples inside and outside.
        """
        super().__init__(datadir, split, n_samples, sampledir, samplefile)
        self.balance = balance

        # List part sample filenames
        self.part_sampledir = part_sampledir
        self.part_samplefile = part_samplefile if part_samplefile is not None else samplefile
        self.part_filenames = []
        for instance in self.instances:
            self.part_filenames.append(os.path.join(self.datadir, self.part_sampledir, instance, self.part_samplefile))
    

    def __getitem__(self, idx):
        npz = np.load(self.filenames[idx])
        part_npz = np.load(self.part_filenames[idx])

        samples = {k: np.concatenate([npz[k], part_npz[k][:, None]], axis=1) for k in ['pos', 'neg']}
        samples = samples_from_array(samples['pos'], samples['neg'], self.n_samples, balance=self.balance,
                                        full_samples=True)
        xyz = samples[:, 0:3].astype(np.float32)
        sdf = samples[:, 3:4].astype(np.float32)
        label = samples[:, 4:5].astype(int)
        return idx, xyz, sdf, label


class MultiDataset(torch.utils.data.Dataset):
    """
    Join multiple datasets into a single one and concatenate their outputs.
    
    This assumes that they are all aligned w.r.t. the shapes!
    """

    def __init__(self, datasets):
        self.datasets = datasets
        self._len = min([len(dataset) for dataset in datasets])
    

    def __len__(self):
        return self._len
    

    def __getitem__(self, idx):
        out = tuple()
        for dataset in self.datasets:
            x = dataset[idx]
            if not isinstance(x, tuple):
                x = (x,)
            out = out + x
        return out


##############
# Generation #
##############

def generate_deepsdf_samples(mesh, n_nearsurface=250_000, n_uniform=25_000, ns_var=0.005, 
                             only_xyz=False):
    """
    Generate SDF samples for the given watertight mesh, following DeepSDF, Park et al., CVPR2019.

    Args
    ----
    mesh: trimesh.Trimesh
        Triangular mesh, can be loaded from disk with trimesh.load(filename).
        /!\ Its vertices should be bounded in [-1, 1]^3 !
        It should also be watertight, otherwise the SDF computation might be wrong.
    n_nearsurface: int (default=250_000)
        Number of samples to generate near the surface of the mesh. (will be times 2)
    n_uniform: int (default=25_000)
        Number of samples to generate uniformly in the [-1, 1]^3 cube.
        Usually 5-10% of nearsurface should suffice.
    ns_var: float (default=0.005)
        Variance of the Gaussian noise to add to the nearsurface samples.
    only_xyz: bool (default=False)
        If True, will return the samples without the SDF values as a single array.
    
    Returns
    -------
    data: dict of np.ndarray (or np.ndarray if `only_xyz`)
        Samples with XYZ coords + SDF value (:= 4D), regrouped based on 
        the sign of their SDF.
        - data['pos']: array of shape (N1, 4), N1 samples with positive SDF.
        - data['neg']: array of shape (N2, 4), N2 samples with negative SDF.
        It can be saved with np.savez(filename, **data)
    """
    # Nearsurface samples, computed by adding Gaussian noise to surface samples
    surf_samples = mesh.sample(n_nearsurface)
    xyz_nearsurface = np.concatenate([
        surf_samples + np.random.normal(scale=sqrt(ns_var), size=surf_samples.shape),
        surf_samples + np.random.normal(scale=sqrt(ns_var/10.), size=surf_samples.shape)
    ], axis=0)

    # Uniform samples
    xyz_uniform = np.random.uniform(-1., 1., (n_uniform, 3))
    
    xyz = np.concatenate([xyz_nearsurface, xyz_uniform], axis=0)
    if only_xyz:
        return xyz
    
    # Compute their SDF
    sdf = igl.signed_distance(xyz, mesh.vertices, mesh.faces)[0]
    samples = np.concatenate([xyz, sdf[:, None]], axis=1)
    
    samples = samples.astype(np.float32)
    samples = remove_nans(samples)
    
    # Separate positive and negative samples
    pos_idx = samples[:,3] >= 0.
    pos = samples[pos_idx]
    neg = samples[~pos_idx]
    data = {
        "pos": pos,
        "neg": neg
    }
    return data