from utils.introspection import wrap_and_override_class_defaults

import torch.nn as nn


class IdentityLayer(nn.Module):
    """
    Return identity.
    """

    def __init__(self, *args, **kwargs):
        super(IdentityLayer, self).__init__()

    def forward(self, x):
        return x


def build_norm_layer(cfg):
    """
    Build the normalization wrapper to be pass to the models constructors.
    Parameters
    ----------
    cfg : dict
            name : str
                Exact name of the norm layer to use
            `param_name` : obj, Multiple entries
                Name(key) and default value(val) of args for the norm layer constructor.
    Returns
    -------
    norm_layer_wrapper : function
        A function that wrapps the constructor of the norm layer
        with the defaults from the config.
    """

    # Return IdentityLayer if no norm required
    if cfg is None:
        return IdentityLayer

    return wrap_and_override_class_defaults(cfg, 'torch.nn')


def print_summary(model):
    # Print model
    print(f"\nUsing model:\n{model}")

    # Print number of trainable parameters
    nb_params = 0
    nb_trainable_params = 0
    for p in model.parameters():
        param_count = p.size().numel()
        if p.requires_grad:
            nb_trainable_params += param_count
        nb_params += param_count

    print(f"# of parameters: {nb_params}")
    print(f"# of trainable parameters: {nb_trainable_params}")
