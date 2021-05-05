from utils.introspection import wrap_and_override_class_defaults


def get_optimizer(optim_cfg, model, optim_state=None):
    """
    Get the optimizer to be used.
    Parameters
    ----------
    optim_cfg : dict
        Dictionary containing the config of the optimizer.
    model : nn.Module
        The model to optimize.
    optim_state : dict
        The state of the optimizer to be load. Default: None.
    Returns
    -------
    optimizer : obj
        The optimizer object.
    """
    optim_class_cfg = optim_cfg.copy()
    if 'weight_decay' in optim_cfg:
        optim_class_cfg['weight_decay'] = optim_class_cfg['weight_decay']['rate']
        if optim_cfg['weight_decay']['type'] == 'Weights':
            decay = []
            no_decay = []
            for name, param in model.named_parameters():
                if param.requires_grad and not any(i in name for i in ['bias', 'bn']):
                    decay.append(param)
                else:
                    no_decay.append(param)
            params = [{'params': decay},
                      {'params': no_decay, 'weight_decay': 0.}]
        else:
            params = model.parameters()

    else:
        params = model.parameters()

    optimizer_class = wrap_and_override_class_defaults(optim_class_cfg, 'torch.optim')
    optimizer = optimizer_class(params)

    if optim_state:
        optimizer.load_state_dict(optim_state)

    return optimizer
