import copy

import torch
import torch.nn as nn

from model.simplenet import SimpleNet
from model.combine_features import CombineFeatures
from utils.model import IdentityLayer  # noqa: F401
from utils.model import build_norm_layer
from utils.checkpoint import load_state_dict


def get_model(model_cfg, num_class, use_both_eyes, device, model_state=None, grad_checkpointing=False):
    """
    Get the model to be used.

    Parameters
    ----------
    model_cfg : dict
        Dictionary containing the config of the model.
    num_class: int
        Numble of classes the model should be predicting.
    use_both_eyes : bool
        Tell the network if it whould use both eyes of a patient to make a prediction.
    device : str
        The device that is used. In ["cuda:0", "cpu"]
    model_state : dict
        The state of the model to be load. Default: None.

    Returns
    -------
    model : nn.Module
        The model.
    """
    pretrain_model = model_cfg['pretrain']

    if pretrain_model and model_state is None:
        print(f"Load pretrain model from: \n {pretrain_model}")
        state = load_state_dict(pretrain_model, True, device)
        model_state = state['model']

    # Define a new model
    model = build_model(model_cfg, num_class, use_both_eyes, device, model_state, grad_checkpointing=False)

    return model


def build_model(model_cfg, num_class, use_both_eyes, device, model_state=None, grad_checkpointing=False):
    """
    Design the model to be used.

    Parameters
    ----------
    model_cfg : dict
        Dictionary containing the config of the model.
    num_class: int
        Numble of classes the model should be predicting.
    use_both_eyes : bool
        Tell the network if it whould use both eyes of a patient to make a prediction.
    device : str
        The device that is used. In ["cuda:0", "cpu"]
    model_state : dict
        The state of the model to be load. Default: None.

    Returns
    -------
    model : nn.Module
        The model.
    """
    cfg_base = copy.deepcopy(model_cfg['base'])
    base_model_name = cfg_base['name']
    del cfg_base['name']

    if 'norm_layer' in cfg_base:
        cfg_base['norm_layer'] = build_norm_layer(cfg_base['norm_layer'])

    # Define base net (image embedding function)
    if base_model_name.startswith("cnn"):
        base_net = SimpleNet(model_cfg['base'])
    else:
        # Import proper model class
        exec(f"from torchvision.models import {base_model_name}")
        model_class = eval(base_model_name)

        base_net = model_class(**cfg_base)

        # DenseNet = classifier
        # ResNet, Inception, ShuffleNetV2, GoogLeNet = fc
        # AlexNet, VGG, MNASNet, MobileNetV2, SqueezeNet = classifier[]
        remove_head = True
        for base_classifier_name in ["classifier", "fc"]:
            if hasattr(base_net, base_classifier_name):
                base_classif = eval(f"base_net.{base_classifier_name}")
                # Set emb_size used in CombineFeatures
                if isinstance(base_classif, nn.Sequential):
                    if isinstance(base_classif[0], nn.Dropout):
                        if isinstance(base_classif[1], nn.Conv2d):
                            # Squeeznet
                            base_net.emb_size = base_classif[1].out_channels
                            remove_head = False
                        else:
                            # Alexnet & Mobilenet
                            base_net.emb_size = base_classif[1].in_features
                    else:
                        base_net.emb_size = base_classif[0].in_features
                else:
                    base_net.emb_size = base_classif.in_features
                # Remove the head of the network
                if remove_head:
                    exec(f"base_net.{base_classifier_name} = IdentityLayer()")
                break

    model = CombineFeatures(base_net, use_both_eyes, model_cfg['dropout_rate'], num_class, grad_checkpointing)

    # Data parallel
    if torch.cuda.device_count() > 1 and device != 'cpu':
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)
    if model_state:
        # Checkpointing: restart training from some state
        if torch.cuda.device_count() > 1 and device != 'cpu':
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)

    return model
