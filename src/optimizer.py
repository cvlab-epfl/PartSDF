"""
Module used for defining optimizers and schedulers.
"""

import torch.optim as optim


##############
# Optimizers #
##############

def _get_optimizer_algorithm(type="adam"):
    """Return the correct optimizer function."""
    if type is None or type.lower() in ["adam"]:
        optimizer = optim.Adam
    elif type.lower() in ["sgd"]:
        optimizer = optim.SGD
    elif type.lower() in ["rms", "rmsprop", "rms-prop"]:
        optimizer = optim.RMSprop
    elif type.lower() in ["adamw"]:
        optimizer = optim.AdamW
    else:
        raise RuntimeError(f"Unknown optimizer type \"{type}\".")
    return optimizer

def get_optimizer(models, type="adam", lrs=[0.0005, 0.001], **kwargs):
    """Get the optimizer for the models."""
    if not isinstance(models, list):
        models = [models]
    if not isinstance(lrs, list):
        lrs = [lrs] * len(models)
    return _get_optimizer_algorithm(type)([
        {
            "params": model if isinstance(model, list) else model.parameters(),
            "lr": lr,
            **kwargs
        }
        for model, lr in zip(models, lrs)
    ])


##############
# Schedulers # 
##############

def get_scheduler(optimizer, **kwargs):
    """Get a scheduler based on the given specifications."""
    if kwargs.get("Type", "Constant") == "Constant":
        scheduler = None
    elif kwargs["Type"] == "Step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            kwargs["Interval"],
            kwargs["Factor"]
        )
    elif kwargs["Type"] == "MultiStep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, 
            kwargs["Milestones"],
            kwargs["Factor"]
        )
    else:
        raise RuntimeError(f"Unknown scheduler type \"{kwargs['Type']}\".")
    return scheduler
