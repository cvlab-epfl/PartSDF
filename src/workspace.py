"""
Module to deal with experimental directories.
"""

import os, os.path
import shutil
import json

import numpy as np
import imageio
import torch

from .visualization import image_grid


# Experimental sub-directories
MODEL_DIR = "model"
OPTIM_DIR = "optimizer"
LATENT_DIR = "latent"
LOG_DIR = "log"
RENDER_SUBDIR = os.path.join(LOG_DIR, "rendering")
RECON_DIR = "reconstruction"
RECON_LAT_SUBDIR = "latent"
RECON_MESH_SUBDIR = "mesh"
RECON_PARTS_SUBDIR = "parts"
RECON_POSES_SUBDIR = "poses"
GEN_DIR = "generation"
GEN_MESH_SUBDIR = "mesh"
GEN_PARTS_SUBDIR = "parts"
EVAL_DIR = "evaluation"
# Files
SPECS_FILE = "specs.json"
CHECKPOINT_FILE = "checkpoint.pth"
HISTORY_FILE = os.path.join(LOG_DIR, "history.pth")


_SUBDIRS = [MODEL_DIR, OPTIM_DIR, LATENT_DIR, 
            RENDER_SUBDIR, LOG_DIR, RECON_DIR, EVAL_DIR]

def build_experiment_dir(expdir, exist_ok=True):
    """Create the experimental sub-directories."""
    for subdir in _SUBDIRS:
        os.makedirs(os.path.join(expdir, subdir), exist_ok=exist_ok)

def reset_experiment_dir(expdir, ignore_errors=True):
    """Reset the experimental directory."""
    for subdir in _SUBDIRS:
        shutil.rmtree(os.path.join(expdir, subdir), ignore_errors=ignore_errors)
    if os.path.isfile(get_checkpoint_filename(expdir)):
        os.remove(get_checkpoint_filename(expdir))
    build_experiment_dir(expdir)


def load_specs(expdir):
    """Load the experimental specifications."""
    filename = os.path.join(expdir, SPECS_FILE)

    if not os.path.isfile(filename):
        raise FileNotFoundError(f"No specification file found for experiment '{expdir}'.")
    
    with open(filename) as f:
        specs = json.load(f)
    return specs


#########################
# Files and Directories #
#########################

def get_log_dir(expdir):
    """Return the log directory for the experiment."""
    return os.path.join(expdir, LOG_DIR)


def get_recon_latent_subdir(expdir, epoch):
    """Return the path to the reconstructed latents."""
    return os.path.join(expdir, RECON_DIR, str(epoch), RECON_LAT_SUBDIR)


def get_recon_mesh_subdir(expdir, epoch):
    """Return the path to the reconstructed meshes."""
    return os.path.join(expdir, RECON_DIR, str(epoch), RECON_MESH_SUBDIR)


def get_recon_parts_subdir(expdir, epoch):
    """Return the path to the reconstructed parts."""
    return os.path.join(expdir, RECON_DIR, str(epoch), RECON_PARTS_SUBDIR)


def get_recon_poses_subdir(expdir, epoch):
    """Return the path to the reconstructed poses."""
    return os.path.join(expdir, RECON_DIR, str(epoch), RECON_POSES_SUBDIR)


def get_eval_dir(expdir, epoch):
    """Return the path to the evaluations."""
    return os.path.join(expdir, EVAL_DIR, str(epoch))


def get_gen_dir(expdir, epoch):
    """Return the path to the generated data."""
    return os.path.join(expdir, GEN_DIR, str(epoch))

def get_gen_mesh_subdir(expdir, epoch):
    """Return the path to the generated meshes."""
    return os.path.join(expdir, GEN_DIR, str(epoch), GEN_MESH_SUBDIR)

def get_gen_parts_subdir(expdir, epoch):
    """Return the path to the generated parts."""
    return os.path.join(expdir, GEN_DIR, str(epoch), GEN_PARTS_SUBDIR)


##############
# Checkpoint #
##############

def get_checkpoint_filename(expdir):
    """Get the path to the checkpoint."""
    return os.path.join(expdir, CHECKPOINT_FILE)

def save_checkpoint(expdir, checkpoint):
    """Save the training checkpoint."""
    filename = os.path.join(expdir, CHECKPOINT_FILE)
    torch.save(checkpoint, filename)

def load_checkpoint(expdir):
    """Load the training checkpoint."""
    filename = os.path.join(expdir, CHECKPOINT_FILE)
    return torch.load(filename)


def save_model(expdir, model, epoch):
    """Save the model."""
    filename = os.path.join(expdir, MODEL_DIR, f"model_{epoch}.pth")
    torch.save(model.state_dict(), filename)

def load_model(expdir, model, epoch):
    """Load the model."""
    params = torch.load(os.path.join(expdir, MODEL_DIR, f"model_{epoch}.pth"))
    model.load_state_dict(params)


def save_latents(expdir, latents, epoch):
    """Save the latent vectors."""
    filename = os.path.join(expdir, LATENT_DIR, f"latents_{epoch}.pth")
    torch.save(latents.state_dict(), filename)

def load_latents(expdir, latents, epoch):
    """Load the latent vectors."""    
    params = torch.load(os.path.join(expdir, LATENT_DIR, f"latents_{epoch}.pth"))
    latents.load_state_dict(params)


def save_poses(expdir, poses, epoch=None):
    """Save the pose vectors."""
    epoch = f"_{epoch}" if epoch is not None else ""
    filename = os.path.join(expdir, LATENT_DIR, f"poses{epoch}.pth")
    torch.save(poses.state_dict(), filename)

def load_poses(expdir, poses, epoch=None):
    """Load the poses vectors."""
    epoch = f"_{epoch}" if epoch is not None else ""
    params = torch.load(os.path.join(expdir, LATENT_DIR, f"poses{epoch}.pth"))
    poses.load_state_dict(params)


def save_optimizer(expdir, optimizer, epoch):
    """Save the optimizer."""
    filename = os.path.join(expdir, OPTIM_DIR, f"optimizer_{epoch}.pth")
    torch.save(optimizer.state_dict(), filename)

def load_optimizer(expdir, optimizer, epoch):
    """Load the optimizer."""
    params = torch.load(os.path.join(expdir, OPTIM_DIR, f"optimizer_{epoch}.pth"))
    optimizer.load_state_dict(params)


def save_scheduler(expdir, scheduler, epoch):
    """Save the learning rate scheduler."""
    filename = os.path.join(expdir, OPTIM_DIR, f"scheduler_{epoch}.pth")
    state_dict = scheduler.state_dict() if scheduler is not None else None
    torch.save(state_dict, filename)

def load_scheduler(expdir, scheduler, epoch):
    """Load the learning rate scheduler."""
    filename = os.path.join(expdir, OPTIM_DIR, f"scheduler_{epoch}.pth")
    if scheduler is not None and os.path.isfile(filename):
        params = torch.load(filename)
        if params is not None:
            scheduler.load_state_dict(params)


def save_experiment(expdir, epoch, model, latents, optimizer, scheduler):
    """Combine saving functions."""
    if model is not None:
        save_model(expdir, model, epoch)
    if latents is not None:
        save_latents(expdir, latents, epoch)
    if optimizer is not None:
        save_optimizer(expdir, optimizer, epoch)
    if scheduler is not None:
        save_scheduler(expdir, scheduler, epoch)

def load_experiment(expdir, epoch, model, latents, optimizer, scheduler):
    """Combine loading functions."""
    if model is not None:
        load_model(expdir, model, epoch)
    if latents is not None:
        load_latents(expdir, latents, epoch)
    if optimizer is not None:
        load_optimizer(expdir, optimizer, epoch)
    if scheduler is not None:
        load_scheduler(expdir, scheduler, epoch)


def save_history(expdir, history):
    """Save the history/log of the experiment."""
    filename = os.path.join(expdir, HISTORY_FILE)
    torch.save(history, filename)

def load_history(expdir, maxepoch=None):
    """Load the history/log of the experiment."""
    filename = os.path.join(expdir, HISTORY_FILE)
    history = torch.load(filename)
    if maxepoch is not None and maxepoch != 'latest':
        clip_history(history, maxepoch)
    return history

def clip_history(history, maxepoch):
    """Clip the history up to the given maxepoch."""
    epoch = history['epoch']
    if maxepoch < epoch:
        history['epoch'] = maxepoch
        for key in history.keys():
            if key != 'epoch':
                history[key] = history[key][:maxepoch]
    elif maxepoch > epoch:
        print(f'Warning: Tried to clip history at epoch {maxepoch}>{epoch}.')


def build_checkpoint(epoch, model, latents, optimizer, scheduler):
    """Build a checkpoint dictionary."""
    checkpoint = {'epoch': epoch}
    checkpoint['model_state_dict'] = model.state_dict() if model is not None else None
    checkpoint['latents_state_dict'] = latents.state_dict() if latents is not None else None
    checkpoint['optimizer_state_dict'] = optimizer.state_dict() if optimizer is not None else None
    checkpoint['scheduler_state_dict'] = scheduler.state_dict() if scheduler is not None else None
    return checkpoint

def use_checkpoint(checkpoint, model, latents, optimizer, scheduler):
    """Load the checkpoint."""
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    if latents is not None:
        latents.load_state_dict(checkpoint['latents_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


def save_renders(expdir, renders, epoch):
    """Save the renderings."""
    filename = os.path.join(expdir, RENDER_SUBDIR, f"render_{epoch}.png")
    image = (image_grid(renders) * 255.).astype(np.uint8)
    imageio.imsave(filename, image)

def save_slices(expdir, slices, epoch):
    """Save the slice plots."""
    filename = os.path.join(expdir, RENDER_SUBDIR, f"slice_{epoch}.png")
    arrays = [np.array(slice) for slice in slices]
    image = (image_grid(arrays)).astype(np.uint8)
    imageio.imsave(filename, image)