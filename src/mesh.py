"""
Module for creating and dealing with meshes.
"""

import time

import numpy as np
import trimesh
from skimage.measure import marching_cubes
import torch

from .utils import make_grid, compute_sdf, get_device


@torch.no_grad()
def create_mesh(model, latent, N=256, max_batch=32**3, verbose=False, grid_filler=False, device=get_device(),
                occ_tau=0.1, **model_kwargs):
    """
    Reconstruct the mesh from the latent vector(s).

    Args:
        model (nn.Module): INR model.
        latent (torch.Tensor): latent vector(s) of the shape to mesh.
        N (int): resolution of the grid.
        max_batch (int): maximum batch size for predictions.
        verbose (bool): enable verbosity.
        grid_filler (bool or SdfGridFiller): whether to use an SdfGridFiller.
        device (str or Device): device to use.
        occ_tau (float): temperature scaling for occupancy (if relevant).
        model_kwargs (dict): additional keyword arguments for the model, such as pose.
    """
    use_occ = hasattr(model, 'use_occ') and model.use_occ  # whether to use occupancy instead of SDF
    # Compute the SDF values on a grid (fast or exact)
    if not use_occ and isinstance(grid_filler, SdfGridFiller):
        sdf_grid = grid_filler.fill_grid(grid_filler.make_sdf_func(model, latent, max_batch, **model_kwargs), verbose=verbose)
        N = grid_filler.N_max
    elif not use_occ and grid_filler:
        gf = SdfGridFiller(N, device="cpu")
        sdf_grid = gf.fill_grid(gf.make_sdf_func(model, latent, max_batch, **model_kwargs), device="cpu", verbose=verbose)
    else:
        sdf_grid = compute_sdf_grid(model, latent, N=N, max_batch=max_batch, verbose=verbose, device=device, **model_kwargs)
    # Marching cubes
    if use_occ:
        level, gradient_direction = 0.5, 'ascent'
        sdf_grid = torch.sigmoid(sdf_grid * occ_tau)  # temperature scaling to smoothen Marching-Cubes results
    else:
        level, gradient_direction = 0., 'descent'
    mesh = convert_sdf_grid_to_mesh(sdf_grid, voxel_size=2. / (N - 1), level=level, gradient_direction=gradient_direction)
    return mesh


@torch.no_grad()
def create_parts(model, latent, N=256, max_batch=32**3, verbose=False, grid_filler=False, device=get_device(), 
                 occ_tau=0.1, **model_kwargs):
    """
    Reconstruct the individual parts from the latent vector(s).

    Args:
        model (nn.Module): INR model.
        latent (torch.Tensor): latent vector(s) of the shape to mesh.
        N (int): resolution of the grid.
        max_batch (int): maximum batch size for predictions.
        verbose (bool): enable verbosity.
        grid_filler (bool or SdfGridFiller): whether to use an SdfGridFiller.
        device (str or Device): device to use.
        occ_tau (float): temperature scaling for occupancy (if relevant).
        model_kwargs (dict): additional keyword arguments for the model, such as pose.
    """
    use_occ = hasattr(model, 'use_occ') and model.use_occ  # whether to use occupancy instead of SDF
    # Compute the SDF values on a grid (fast or exact)
    if use_occ or not grid_filler:
        sdf_grid = compute_sdf_grid(model, latent, N=N, return_parts=True, max_batch=max_batch, 
                                    verbose=verbose, device=device, **model_kwargs)
    else:
        if not isinstance(grid_filler, SdfGridFiller):
            grid_filler = SdfGridFiller(N, device="cpu")
        sdf_grid = torch.stack([
            grid_filler.fill_grid(grid_filler.make_sdf_func(model, latent, max_batch, i, **model_kwargs), device="cpu", verbose=verbose)
            for i in range(model.n_parts)
        ], dim=-1)
        N = grid_filler.N_max
    # Marching cubes
    if use_occ:
        level, gradient_direction = 0.5, 'ascent'
        sdf_grid = torch.sigmoid(sdf_grid * occ_tau)  # temperature scaling to smoothen Marching-Cubes results
    else:
        level, gradient_direction = 0., 'descent'
    parts = [convert_sdf_grid_to_mesh(sdf_grid[..., i], voxel_size=2. / (N - 1), level=level, gradient_direction=gradient_direction)
             for i in range(sdf_grid.shape[-1])]
    return parts


def compute_sdf_grid(model, latent, N=256, max_batch=32**3, bbox=[(-1., -1., -1.), (1., 1., 1.)], 
                     return_parts=False, verbose=False, device=get_device(), **model_kwargs):
    """
    Compute the SDF values over a grid in the given bounding box.

    Args:
        model (nn.Module): INR model.
        latent (torch.Tensor): latent vector(s) of the shape.
        N (int): resolution of the grid.
        max_batch (int): maximum batch size for predictions.
        bbox (list): bounding box of the grid, a list of the min and max coordinates.
        return_parts (bool): whether to return the parts SDFs.
        verbose (bool): enable verbosity.
        device (str or Device): device to use.
        model_kwargs (dict): additional keyword arguments for the model, such as pose.
    """
    # Create points on a grid
    xyz = make_grid(bbox, N, device=device)

    # Predict the SDF values on the grid
    sdf = compute_sdf(model, latent, xyz, return_parts, max_batch, verbose, device=device, **model_kwargs)  
    return sdf.squeeze(-1)


def convert_sdf_grid_to_mesh(sdf_grid, voxel_size=1., voxel_origin=[-1., -1., -1.], level=0.,
                             gradient_direction='descent', trimesh_process=True):
    """
    Convert the grid of SDF values to a mesh.

    Args:
        sdf_grid (torch.Tensor or np.ndarray): grid of SDF values.
        voxel_size (float or list): size of the voxels.
        voxel_origin (list): origin of the voxel grid.
        level (float): level to extract the mesh.
        gradient_direction (str): direction of the gradient ('descent' for SDF, 'ascent' for OCC).
        trimesh_process (bool): whether to process the mesh with trimesh.
    """
    if not isinstance(sdf_grid, np.ndarray):
        sdf_grid = sdf_grid.detach().cpu().numpy()
    if isinstance(voxel_size, (int, float)):
        voxel_size = [voxel_size] * 3
    
    try:
        verts, faces, _, _ = marching_cubes(sdf_grid, level=level, spacing=voxel_size, 
                                            gradient_direction=gradient_direction)
        verts += np.array(voxel_origin)
    except ValueError:  # no surface within range
        verts, faces = None, None

    return trimesh.Trimesh(verts, faces, process=trimesh_process)


class SdfGridFiller():
    """
    Smart grid filler that starts at low resolution and go up by querying coordinates
    where the SDF is low instead of the whole grid.

    Adapted from Benoit Guillard.
    """
    
    def __init__(self, N_max, device=get_device()):
        """
        Initialize the grid filler.

        Construct grid and precompute sparse masks, from 32 to N_max.
        
        Args:
        -----
        N_max : int
            Maximum resolution of the grid.
        device: str or Device (default=cuda if available)
            Device where the xyz coords are stored (trade-off memory vs computation).
        """
        # Save attributes
        self.N_max = N_max
        
        # Create the coords grid (N,N,N,3) where 3 = x,y,z
        xyz = make_grid([(-1., -1., -1.), (1., 1., 1.)], N_max)
        xyz.pin_memory()
        self.xyz = xyz.reshape(-1, 3).to(device)
        
        # Precompute binary masks for sparsely adressing the above grid
        self.N_levels = [32 * (2**i) for i in range(int(np.log2(N_max) - 4))] # Minimum level = 32
        mask = torch.zeros((N_max, N_max, N_max), dtype=bool, pin_memory=True)
        # Fill dictionaries with precomputed masks
        self.masks_coarse = {}
        self.masks_coarse_no_recompute = {}
        self.idxs_coarse_neighbors_blocks = {}
        for i, N in enumerate(self.N_levels):
            step = N_max // N
            #### 1: Subsample coarsely
            mask_coarse = mask.clone()
            mask_coarse[::step, ::step, ::step] = True
            # (N_max**3) array, with True only for indices of the coarse sampling (N**3 locations):
            mask_coarse = mask_coarse.reshape(-1) 
            self.masks_coarse[i] = mask_coarse.clone().to(self.xyz.device)

            #### 2: Compute the idxs of neighboring blocks
            neighbors_block_coarse = mask.clone()
            neighbors_block_coarse[:step, :step, :step] = True
            neighbors_block_coarse = neighbors_block_coarse.reshape(-1)
            # Shape (N_max**3 // N, N): idxs_coarse_neighbors_blocks[i] represents the (N_max // N)**3 indexes covered by coarse point i
            idxs_coarse_neighbors_blocks = torch.where(mask_coarse)[0].reshape(-1,1) + torch.where(neighbors_block_coarse)[0].reshape(1,-1)
            self.idxs_coarse_neighbors_blocks[i] = idxs_coarse_neighbors_blocks.clone().to(self.xyz.device)

            #### 3: If at a level >0, do not recompute already queried SDFs
            if i > 0:
                mask_coarse_no_recompute = mask_coarse.clone()
                mask_coarse_no_recompute[self.masks_coarse[i-1]] = False
                self.masks_coarse_no_recompute[i] = mask_coarse_no_recompute.clone().to(self.xyz.device)

                
    @torch.no_grad()
    def fill_grid(self, sdf_func, return_queries=False, device=get_device(), verbose=False):
        """
        Fill the SDF grid by increasing resolution.

        Args:
        -----
        sdf_func: callable (N, 3) -> (N)
            Function that takes a batch of coordinates and returns the SDF values.
        return_queries: bool (default=False)
            Also return the number of queried points.
        device: str (default="cuda:0" if available)
            Device where the SDF grid is stored.
        
        Returns:
        --------
        sdf_values: torch.Tensor, shape (N_max, N_max, N_max)
            The SDF values on the grid.
        queries_number: int (optional, if return_queries)
            The number of queries made to the network.
        """
        if verbose:
            start_time = time.time()
        if return_queries:
            queries_number = 0
        
        sdf_values = torch.zeros(len(self.xyz), device=device)
        close_surface_masks = {}
        idxs_coarse_neighbors_blocks_LOCAL = {}
        for level, N in enumerate(self.N_levels):
            # Prepare masks based on previous levels
            if level == 0:
                mask_coarse = self.masks_coarse[level]
                idxs_coarse_neighbors_blocks = self.idxs_coarse_neighbors_blocks[level].clone()
                mask_coarse_no_recompute = self.masks_coarse[level]
            else:
                # Mask using previous queries
                # binary mask
                mask_coarse = self.masks_coarse[level].clone()
                for l in range(level):
                    mask_coarse[idxs_coarse_neighbors_blocks_LOCAL[l][~close_surface_masks[l]]] = False
                # idx tensor
                if N < self.N_max:
                    idxs_coarse_neighbors_blocks = self.idxs_coarse_neighbors_blocks[level].clone()
                    idxs_coarse_neighbors_blocks = idxs_coarse_neighbors_blocks[mask_coarse[self.masks_coarse[level]]]
                else:
                    idxs_coarse_neighbors_blocks = self.idxs_coarse_neighbors_blocks[level]
                # The no_recompute version does not re-query the decoder for nodes that have already been (1/8th is saved)
                mask_coarse_no_recompute = self.masks_coarse_no_recompute[level].clone()
                for l in range(level):
                    mask_coarse_no_recompute[idxs_coarse_neighbors_blocks_LOCAL[l][~close_surface_masks[l]]] = False
            
            idxs_coarse_neighbors_blocks_LOCAL[level] = idxs_coarse_neighbors_blocks
            # Query the network
            if return_queries:
                queries_number += mask_coarse_no_recompute.sum().item() # count the total number of network queries
            xyz = self.xyz[mask_coarse_no_recompute]
            # Query + fill grid
            sdf_values[mask_coarse_no_recompute] = sdf_func(xyz).detach().to(device)

            # Prepare next levels queries
            if N < self.N_max:
                ## Which samples are close to the surface? (NOTE: where do current thresholds come from?)
                STEP_SIZE = 2. / N
                close_surface_mask = (torch.abs(sdf_values[mask_coarse.to(device)]) < 1.5 * 1.7 * STEP_SIZE)
                close_surface_masks[level] = close_surface_mask
                # For those far of the surface, we can ignore them for the future 
                # and copy the high value to their neighbors
                sdf_values[idxs_coarse_neighbors_blocks[~close_surface_mask]] = sdf_values[mask_coarse.to(device)][~close_surface_mask].unsqueeze(-1)
        
        sdf_values = sdf_values.reshape(self.N_max, self.N_max, self.N_max)
        if verbose:
            print(f"sdf-prediction took {time.time() - start_time:.3f}s.")   
        if return_queries:
            return sdf_values, queries_number
        return sdf_values
    
    
    @torch.no_grad()
    def make_sdf_func(self, model, latent, max_batch=32**3, part_idx=None, device=get_device(), **model_kwargs):
        """Make an SDF function out of a model and latent vector."""
        if latent.shape[0] != 1:  # singleton batch dimension
            latent = latent.unsqueeze(0)
        
        def sdf_func(xyz, max_batch=max_batch):
            model.eval()

            # Prepare data
            xyz_all = xyz.view(-1, 3)
            n_points = len(xyz_all)
            sdf = torch.zeros(n_points, device=device)

            # Predict SDF on a subset of points at a time
            for i in range(0, n_points, max_batch):
                xyz_subset = xyz_all[i : i + max_batch].to(device)
                if part_idx is not None:
                    sdf[i : i + max_batch] = model(latent, xyz_subset, return_parts=True, **model_kwargs)[:,part_idx,0].detach().to(device)
                else:
                    sdf[i : i + max_batch] = model(latent, xyz_subset, **model_kwargs)[:,0].detach().to(device)

            return sdf.view(xyz.shape[:-1])
        
        return sdf_func
    

    @torch.no_grad()
    def make_mesh(self, sdf_func, return_queries=False, device=get_device(), verbose=False):
        """Mesh an SDF function."""
        sdf_grid = self.fill_grid(sdf_func, return_queries=return_queries, 
                                  device=device, verbose=verbose)
        return convert_sdf_grid_to_mesh(sdf_grid, voxel_size=2. / (self.N_max - 1))