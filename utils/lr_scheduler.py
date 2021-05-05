from utils.introspection import wrap_and_override_class_defaults


def get_lr_scheduler(scheduler_cfg, optimizer, scheduler_state=None):
    """
    Get the lr scheduler to be used during training.

    Parameters
    ----------
    scheduler_cfg : dict
        Dictionary containing the config of the lr scheduler.
    optimizer : obj
        Optimizer class.
    scheduler_state : dict
        The state of the lr scheduler to be load. Default: None.

    Returns
    -------
    scheduler : obj
        The scheduler object.

    """
    if scheduler_cfg is None:
        return None

    if scheduler_cfg['name'] not in ["StepLR", "MultiStepLR", "ExponentialLR", "CosineAnnealingLR", "CosineAnnealingWarmRestarts"]:
        raise NotImplementedError(f"{scheduler_cfg['name']} in lr_scheduler['name'] not recognized.")

    lr_scheduler_class = wrap_and_override_class_defaults(scheduler_cfg, 'torch.optim.lr_scheduler')
    lr_scheduler = lr_scheduler_class(optimizer)

    if scheduler_state:
        lr_scheduler.load_state_dict(scheduler_state)

    return lr_scheduler
