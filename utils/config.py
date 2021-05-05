import os
import glob
import copy
import time
import pprint

import ruamel.yaml as yaml


def _validate_dataset(dataset_name):
    """
    Validate that the dataset is supported.

    Parameters
    ----------
    dataset_name : str
        Name of dataset to validate

    Raises
    -------
    ValueError
        If dataset name is not supported.
    """

    supported_datasets = ['kaggle2015', 'fake']

    if dataset_name not in supported_datasets:
        raise ValueError(f"The data_folder name '{dataset_name}' is not valid."
                         f"It should be one of the following: {supported_datasets}")


def _extract_verify_dataset_name(data_folder):
    """
    Get dataset name from folder name and verify it's a supported one.

    Parameters
    ----------
    data_folder : str
        Absolute path to the data folder. Data folder should be named "kaggle2015" or "fake".

    Returns
    -------
    dataset_name : str
        Dataset name.
    """

    dataset_name = os.path.basename(os.path.normpath(data_folder))

    _validate_dataset(dataset_name)

    return dataset_name


def _parse_orion(cfg):
    """
    Parse the config for 'not yet supported' Orion flags and inject the appropriate value.

    Note: This function should be remove when Orion add support for them.
    """

    for name in cfg:
        if type(cfg[name]) is str and cfg[name].startswith("orion#"):
            if cfg[name].endswith("exp.name"):
                cfg[name] = os.environ['ORION_EXPERIMENT_NAME']
            elif cfg[name].endswith("trial.id"):
                cfg[name] = os.environ['ORION_TRIAL_ID']
    return cfg


def _rm_cfg_key(cfg, name):
    """
    Remove key from config and return the updated config.
    """

    if name in cfg:
        del cfg[name]
    return cfg


def load_config(args, test_mode=False):
    """
    Load and validate yaml config file to dict.

    Parameters
    ----------
    args : NameSpace
        config : str
            Path to config file.
        data_folder : str
            Path to data folder.
        experiment_folder : str
            Path to checkpoint folder.
        force_resume : bool
            If true, skip checks and force resume.
        no_resume : bool
            If true, skip resume and start new exp.
    test_mode : bool
        If True, skip update/check config and setups. Default: False.

    Returns
    -------
    cfg : dict
        Config dictionary.
    """

    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
        # Hack because orion does not support templating in config file
        cfg = _parse_orion(cfg)

    if not test_mode:
        # Update data configs
        dataset_name = _extract_verify_dataset_name(args.data_folder)
        data_cfg = cfg['dataset']
        if 'name' in data_cfg and data_cfg['name'] != dataset_name:
            raise ValueError("dataset_name in config should match data_folder.")
        else:
            data_cfg['name'] = dataset_name

        # Override resuming by removing confs needed for resuming.
        if 'no_resume' in args and args.no_resume:
            cfg = _rm_cfg_key(cfg, 'mlflow_uid')
            cfg = _rm_cfg_key(cfg, 'config_uid')
            cfg = _rm_cfg_key(cfg, 'experiment_folder')

        # Set exp name if none given
        if 'experiment_name' not in cfg:
            cfg['experiment_name'] = f"retinopathy_{data_cfg['name']}_{data_cfg['target_name']}"
            print(f"Warning: experiment_name not in config file. Setting name to: {cfg['experiment_name']}")

        # Check previous config for resuming
        if 'config_uid' in cfg and 'experiment_folder' in args:
            old_conf_path = os.path.join(args.experiment_folder, cfg['experiment_name'], cfg['config_uid'], 'cfg.yml')

            if os.path.exists(old_conf_path):

                print(f"\n## Run with config_uid '{cfg['config_uid']}' already exist in experiment '{cfg['experiment_name']}'")
                if args.force_resume:
                    print(f"## As --force-resume as been used, {old_conf_path} will be ignored and the given one will be used.\n")
                    temp_path = os.path.splitext(old_conf_path)
                    old_conf_path_renamed = f"{temp_path[0]}_before_{time.strftime('%Y_%m_%d_%H%M%S')}{temp_path[1]}"
                    os.rename(old_conf_path, old_conf_path_renamed)
                else:
                    print(f"## Ignoring given config file and using: {old_conf_path}\n")
                    with open(old_conf_path, 'r') as f:
                        cfg = yaml.load(f, Loader=yaml.Loader)

    return cfg


def _update_and_verify_sub_config(cfg, allowed_cfg, dataset_name):
    """
    Update and verify local config.

    Parameters
    ----------
    cfg: dict
        Config to be checked and updated.
    allowed_cfg: dict
        Config containing list of allowed values or subdict with list of allowed
        values per dataset.
    dataset_name: str
        Name of the dataset being used.

    Returns
    -------
    cfg : dict
        Updated config.
    """

    for k, v in allowed_cfg.items():
        if isinstance(v, dict) and dataset_name in v.keys():
            v = v[dataset_name]

        if k not in cfg:
            cfg[k] = v[0]
            print(f"Warning: dataset.{k} not set. Using default value {v[0]}")
        elif cfg[k] not in v:
            raise ValueError(f"dataset.{k} must be in {v}.")
    return cfg


def _update_and_verify_data_config(data_cfg):
    """
    Update and verify data config.

    Parameters
    ----------
    data_cfg: dict
        Data config to be updated.

    Returns
    -------
    data_cfg : dict
        Updated data config.
    """

    allowed_general_cfg = {
        'resolution': [512, 1024],
        'name': ['kaggle2015', 'fake'],
        'use_both_eyes': [False, True],
        'target_name': ['screening_level', 'DR_level'],
        'feature_scaling': ['MaskedStandardization', 'GreenChannelOnly', 'GreenChannelOnlyMaskedStandardization', 'KaggleWinner', 'None']}

    allowed_train_cfg = {
        'n_views': {'kaggle2015': [1, 2, 3], 'fake': [1, 2, 3]},
        'filter_target': [False, True]}

    allowed_train_aug_cfg = {
        'rotation_type': ['RightAngle', 'Any'],
        'resized_crop': [False, True],
        'different_transform_per_view': {'kaggle2015': [False, True], 'fake': [False, True]}}

    allowed_eval_cfg = {
        'n_views': {'kaggle2015': [1, 2, 3], 'fake': [1, 2, 3]},
        'apply_train_augmentations': [False, True]}

    data_cfg = _update_and_verify_sub_config(data_cfg, allowed_general_cfg, data_cfg['name'])
    data_cfg['train'] = _update_and_verify_sub_config(data_cfg['train'], allowed_train_cfg, data_cfg['name'])
    data_cfg['train']['augmentation'] = _update_and_verify_sub_config(data_cfg['train']['augmentation'], allowed_train_aug_cfg, data_cfg['name'])
    data_cfg['eval'] = _update_and_verify_sub_config(data_cfg['eval'], allowed_eval_cfg, data_cfg['name'])

    if 'sample_size' not in data_cfg:
        data_cfg['sample_size'] = -1
        print("Warning: dataset.sample_size not set. Using default -1.")

    if 'accumulated_batch_size' in data_cfg['train'] and data_cfg['train']['accumulated_batch_size'] is not None:
        if data_cfg['train']['accumulated_batch_size'] % data_cfg['train']['batch_size'] != 0:
            raise ValueError("dataset.train.batch_size should be a multiple of the dataset.train.accumulated_batch_size.")
    else:
        data_cfg['train']['accumulated_batch_size'] = None
        print("Warning: dataset.train.accumulated_batch_size not set. Using default None.")

    if data_cfg['train']['filter_target'] and data_cfg['target_name'] == "DR_level":
        raise ValueError("dataset.train.filter_target should be False to use dataset.target_name DR_level.")

    if 'under_sampling' in data_cfg['train']:
        data_cfg['train']['under_sampling']['ratios'] = eval(data_cfg['train']['under_sampling']['ratios'])
        if 'schedule' in data_cfg['train']['under_sampling']:
            data_cfg['train']['under_sampling']['schedule'] = eval(data_cfg['train']['under_sampling']['schedule'])
        else:
            data_cfg['train']['under_sampling']['schedule'] = [1]
            print("Warning: undersampling.schedule not set. Using default: \"[1]\".")
        if len(data_cfg['train']['under_sampling']['ratios']) != len(data_cfg['train']['under_sampling']['schedule']):
            raise ValueError("undersampling.schedule and undersampling.ratios must have the same number of steps.")

    return data_cfg


def update_config_and_setup_exp_folder(args, cfg, uid):
    """
    Update the configs based on args flags and
    setup the experiment folder as needed.

    Parameters
    ----------
    args : NameSpace
        config : str
            Path to config file.
        data_folder : str
            Path to data folder.
        experiment_folder : str
            Path to checkpoint folder.
        force_resume : bool
            If true, skip checks and force resume.
        no_resume : bool
            If true, skip resume and start new exp.
    cfg : dict
        Raw config loaded directly from the config file.
    uid : str
        Unique id for the experiment run.

    Returns
    -------
    cfg : dict
        Config dictionary.
    """

    if 'mlflow_uid' not in cfg:
        cfg['mlflow_uid'] = uid

    if 'config_uid' not in cfg:
        cfg['config_uid'] = uid

    # Checkpointing
    if 'experiment_folder' not in cfg:
        # Add experiment_folder to cfg with unique identifier
        if args.no_resume:
            exp_subfolder = 'no_resume'
        else:
            exp_subfolder = os.path.join(cfg['experiment_name'], cfg['config_uid'])
        cfg['experiment_folder'] = os.path.join(args.experiment_folder, exp_subfolder)

    if os.path.exists(cfg['experiment_folder']):
        if args.no_resume:
            file_names = glob.glob(os.path.join(cfg['experiment_folder'], "checkpoint_state_*.pth.tar"))
            file_names += [os.path.join(cfg['experiment_folder'], file) for file in ["cfg.yml", "git_diff.txt"]]
            for file_name in file_names:
                if os.path.isfile(file_name):
                    os.remove(file_name)
    else:
        os.makedirs(cfg['experiment_folder'])

    # Dataset conf
    cfg['dataset'] = _update_and_verify_data_config(cfg['dataset'])

    # Scheduler conf
    if 'lr_scheduler' not in cfg:
        cfg['lr_scheduler'] = None
        print("Warning: dataset.train.lr_scheduler not set. Using default: None.")

    # Training conf
    train_cfg = cfg['training']

    if 'patience_metrics' not in train_cfg:
        train_cfg['patience_metrics'] = ['quadratic_weighted_kappa']
        print("Warning: training.patience_metrics not set. Using default: ['quadratic_weighted_kappa'].")
    elif 'j_statistic' in train_cfg['patience_metrics'] and cfg['dataset']['target_name'] != 'screening_level':
        raise ValueError("If \'j_statistic\' in training.patience_metrics, dataset.target_name should be \'screening_level\'.")

    if 'log_grad_every_n_epoch' not in train_cfg:
        print("Warning: training.log_grad_every_n_epoch. Using default 0 (off).")
        train_cfg['log_grad_every_n_epoch'] = 0
    if train_cfg['log_grad_every_n_epoch'] < 0 or type(train_cfg['log_grad_every_n_epoch']) is not int:
        raise ValueError("log_grad_every_n_epoch should be positive int. Set to 0 to turn off.")

    if 'log_image_sample_every_n_epoch' not in train_cfg:
        print("Warning: training.log_image_sample_every_n_epoch. Using default 0 (off).")
        train_cfg['log_image_sample_every_n_epoch'] = 0

    if train_cfg['log_image_sample_every_n_epoch'] < 0 or type(train_cfg['log_image_sample_every_n_epoch']) is not int:
        raise ValueError("log_image_sample_every_n_epoch should be positive int. Set to 0 to turn off.")

    if 'exact_train_stats' not in train_cfg:
        print("Warning: training.exact_train_stats not set. Using default: False.")
        train_cfg['exact_train_stats'] = False

    if 'grad_checkpointing' not in train_cfg:
        print("Warning: training.grad_checkpointing not set. Using default: False.")
        train_cfg['grad_checkpointing'] = False

    # Update loss conf
    if 'loss' not in cfg:
        cfg['loss'] = {}
        if cfg['dataset']['target_name'] == 'screening_level':
            print("Warning: No loss specified, using BCEWithLogitsLoss.")
            cfg['loss']['name'] = 'BCEWithLogitsLoss'
        else:
            print("Warning: No loss specified, using CrossEntropyLoss.")
            cfg['loss']['name'] = 'CrossEntropyLoss'

    # Make a copy to be able to delete/modify keys without impacting cfg
    save_cfg = copy.deepcopy(cfg)
    if save_cfg['dataset']['train']['accumulated_batch_size'] is None:
        del save_cfg['dataset']['train']['accumulated_batch_size']

    if 'under_sampling' in save_cfg['dataset']['train']:
        save_cfg['dataset']['train']['under_sampling']['schedule'] = str(save_cfg['dataset']['train']['under_sampling']['schedule'])
        save_cfg['dataset']['train']['under_sampling']['ratios'] = str(save_cfg['dataset']['train']['under_sampling']['ratios'])

    if save_cfg['lr_scheduler'] is None:
        del save_cfg['lr_scheduler']

    # Overwrite args.config file for checkpointing purpose
    with open(args.config, 'w') as f:
        yaml.dump(save_cfg, f, Dumper=yaml.RoundTripDumper)
    # Saving copy to experiment folder
    with open(os.path.join(cfg['experiment_folder'], 'cfg.yml'), 'w') as f:
        yaml.dump(save_cfg, f, Dumper=yaml.RoundTripDumper)

    # Print config
    print("\nUsing config:")
    pprint.pprint(cfg)

    return cfg


def load_and_update_eval_config(eval_config_path, cfg, verbose=False):
    """
    Load and update eval config.

    Parameters
    ----------
    eval_config_path : str
        Path to eval config file.
    cfg : dict
        Config dictionary.
    verbose : bool
        If True, print Warnings. Default: False.

    Returns
    -------
    cfg : dict
        Updated eval config dictionary.
    """

    # Load eval config
    with open(eval_config_path, 'r') as f:
        eval_cfg = yaml.load(f, Loader=yaml.Loader)

    # Update config using eval config
    data_cfg = cfg['dataset']
    for key, value in eval_cfg.items():
        if key in ['name', 'sample_size']:
            data_cfg[key] = eval_cfg[key]
            if verbose:
                print(f"Warning: dataset.{key} overridden by eval config to {eval_cfg[key]}.")

        elif key == 'train' and isinstance(value, dict):
            if 'n_views' in value:
                data_cfg[key]['n_views'] = eval_cfg[key]['n_views']
                if verbose:
                    print(f"Warning: dataset.train.n_views overridden by eval config to {eval_cfg[key]['n_views']}.")

            if 'augmentation' in value:
                data_cfg[key]['augmentation'].update(eval_cfg[key]['augmentation'])
                if verbose:
                    print(f"Warning: dataset.train.augmentation overridden by eval config to {eval_cfg[key]['augmentation']}.")

        elif key == 'eval' and isinstance(value, dict):
            data_cfg[key].update(eval_cfg[key])
            if verbose:
                print(f"Warning: dataset.valid overridden by eval config to {eval_cfg[key]}.")

        else:
            raise ValueError("Invalide key found in dataset test config.")

    return cfg
