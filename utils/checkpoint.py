import os
import glob
import random

from pathlib import Path
from collections import defaultdict

import torch
import numpy as np


def load_checkpoint(exp_folder, device):
    """
    Load or create initial checkpoint.

    Parameters
    ----------
    cfg : dict
        Config dictionary.
    device : str
        Device currently used.

    Returns
    -------
    state : dict
        Dictionary containing the state of training.
    """

    state_fnames = glob.glob(os.path.join(exp_folder, 'checkpoint_state_*.pth.tar'))
    if len(state_fnames) != 0:
        # Latest state file name
        state_fname = max(state_fnames, key=os.path.getctime)
        if Path(state_fname).exists():
            print(f"Previous checkpoint found. Loading {state_fname} ...", end=' ')
            state = load_state_dict(state_fname, False, device)
            print("Done")
    else:
        state = {'random_states': None,
                 'model': None,
                 'optim': None,
                 'scheduler': None,
                 'train': None}
        print("No checkpoint available. Starting new experiment.")

    return state


def update_state(state, cfg):
    """
    Update the training state with forced config.

    Parameters
    ----------
    state : dict
        Current state of the training.
    cfg : dict
        Config dictionary.

    Returns
    -------
    state : dict
        Dictionary containing the updates state of training.
    """

    # Force upate the training configurations to the current state
    for conf_name in cfg['training'].keys():
        state['train'][conf_name] = cfg['training'][conf_name]

    # Force upate the optimization configurations to the current state
    print("Warning: --force-resume cannot update the weight decay type if it was modified.")
    print("Warning: --force-resume learning rate forced to the lr in the config even if a lr scheduler used.")
    for conf_name in cfg['optimizer'].keys():
        if conf_name != 'name':
            if conf_name == 'weight_decay':
                val = cfg['optimizer'][conf_name]['rate']
            else:
                val = cfg['optimizer'][conf_name]
            state['optim']['param_groups'][0][conf_name] = val

    return state


def load_state_dict(filename, allow_different_device, device):
    """
    Load the dictionary containing the state of training.

    Parameters
    ----------
    filename : str
        Path to the file where the state dictionary is saved.
    allow_different_device : bool
        Whether or not we allow to use state with a different
        device than the one used for saving it.
    device : str
        The device that is used. In ["cuda:0", "cpu"].

    Returns
    -------
    state : dict
        Dictionary containing the state of training.
    """
    state = torch.load(filename, map_location=lambda storage, loc: storage)
    if state['device'] != device:
        if allow_different_device:
            print(f"Model training was done with a different device. Loading with device: {device}.")
        else:
            raise Exception("Experiment started with a different device.")
    return state


def set_seed_and_random_states(seed, random_states, cuda):
    """
    Set the seed.

    Parameters
    ----------
    seed : int or dict
        The seed to be used tto start an experiment.
    random_states : dict
        Dict containing all seed state that  need to be fixed for resuming.
        None is no resuming is available.
    cuda : dict
            determininistic: bool
                To what should we put torch.backends.cudnn.deterministic.
            backends: bool
                To what should we put torch.backends.cudnn.benchmark.
    """

    torch.backends.cudnn.deterministic = cuda['deterministic']
    torch.backends.cudnn.benchmark = cuda['benchmark']

    if random_states is None:
        # Set random state from seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        # if become True add torch.cuda.empty_cache() after the very first
        # forward of the program as the benchmark mode can allocate large
        # memory blocks during the very first forward to test algorithms and
        # that won’t be good latter
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    else:

        # Resuming
        np.random.set_state(random_states['np_state'])
        random.setstate(random_states['python_state'])
        torch.set_rng_state(random_states['torch_state'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(random_states['torch_state_cuda'])


def define_train_cfg(train_cfg, patience_metrics=['quadratic_weighted_kappa'], train_state=None):
    """
    Define the training config.

    Parameters
    ----------
    train_cfg : dict
        The dictionaty containing the training config.
    patience_metrics : list
        Metrics to use for patience. Default: ['quadratic_weighted_kappa'].
    train_state : dict
        Dictionary containing the config and the state of the training loop.
        Default None.

    Returns
    -------
    train_cfg : dict
        Dictionary containing the config and the state of the training loop.
    """
    if not train_state:
        train_cfg['starting_epoch'] = 1
        train_cfg['total_time'] = 0.
        for patience_metric in patience_metrics:
            train_cfg[f"patience_{patience_metric}"] = 0
            train_cfg[f"best_valid_{patience_metric}"] = -np.inf
            train_cfg[f"loss_for_best_valid_{patience_metric}"] = np.inf
    else:
        train_cfg = train_state
        last_epoch = 0
        for patience_metric in patience_metrics:
            last_epoch = max(last_epoch, train_cfg[f"epoch_of_best_valid_{patience_metric}"])
        train_cfg['starting_epoch'] = last_epoch + 1
    return train_cfg


def save_state_dict(filename, device, model, optimizer, lr_scheduler, train_cfg):
    """
    Save the dictionary containing the state of training.

    Parameters
    ----------
    filename : str
        Path to the file where the state dictionary will be save.
    device : str
        The device that is used. In ["cuda:0", "cpu"]
    model : obj
        The model that we want to save the state.
    optimizer : obj
        The optimizer that we want to save the state.
    train_cfg: dict
        The dictionary containing training configuration and state.
    """
    state = defaultdict(dict)
    # Seed state
    state['random_states'] = {'np_state': np.random.get_state(),
                              'python_state': random.getstate(),
                              'torch_state': torch.get_rng_state()}
    if torch.cuda.is_available():
        state['random_states']['torch_state_cuda'] = torch.cuda.get_rng_state()

    # Device
    state['device'] = device

    # Model state
    if torch.cuda.device_count() > 1 and device != 'cpu':
        # Multi-gpu case
        state['model'] = model.module.state_dict()
    else:
        state['model'] = model.state_dict()

    # Optimizer state
    state['optim'] = optimizer.state_dict()

    # Scheduler state
    if lr_scheduler:
        state['scheduler'] = lr_scheduler.state_dict()

    # Train config
    state['train'] = train_cfg
    torch.save(state, filename)
