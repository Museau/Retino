import os

import numpy as np

from PIL import Image


# Datset splits lenght
DATASET_SPLITS_LENGHT = {
    "Kaggle2015": {
        "train": 35116,
        "valid": 10906,
        "test": 42670}}


def check_kaggle2015_config(data_folder, cgf, split, transforms):
    """
    Check Kaggle2015 dataset config.

    Parameters
    ----------
    data_folder : str
        Data folder.
    cfg : dict
        resolution : int
            Input size in [512, 1024].
        sample_size : int
            Number of elements to use as sample size, for debugging purposes only.
            If -1, use all samples
        target_name: str
            Can be 'screening_level' or 'DR_level'. Changes the type of the targets.
        train: dict
            n_views : int
                Number of views. Valid options: 1, 2 ,3.
        eval: dict
            n_views : int
                Number of views. Valid options: 1, 2 ,3.
    split : str
        Should be in ["train",  "valid", "test"].
    transforms : callable
        Transform to be applied on a sample.
    """

    if not os.path.isdir(data_folder):
        raise ValueError(f"data_folder - {data_folder} - is not a directory")

    allowed_resolutions = [512, 1024]
    if cgf['resolution'] not in allowed_resolutions:
        raise ValueError(f"resolution must be in {allowed_resolutions}.")

    if cgf['train']['n_views'] not in range(1, 4):
        raise ValueError("train.n_views valid options are 1, 2 or 3.")

    if cgf['eval']['n_views'] not in range(1, 4):
        raise ValueError("eval.n_views valid options are 1, 2 or 3.")

    splits = ["train", "valid", "test"]
    if split not in splits:
        raise ValueError(f"split must be in {splits}.")

    dataset_split_lenght = DATASET_SPLITS_LENGHT["Kaggle2015"][split]
    if cgf['sample_size'] != -1 and \
       (cgf['sample_size'] < 1 or cgf['sample_size'] > dataset_split_lenght):
        raise ValueError(f"sample_size must be -1 to use all samples or "
                         f"in range(1, {dataset_split_lenght}).")

    allowed_target_names = ['screening_level', 'DR_level']
    if cgf['target_name'] not in allowed_target_names:
        raise ValueError(f"target_name should be one of the following value {allowed_target_names}")

    if not hasattr(transforms, "__call__"):
        raise ValueError("transform should be an object with a __call__ attribute.")

    try:
        image = Image.new("RGB", size=(cgf['resolution'], cgf['resolution']), color=(255, 0, 0))
        image = transforms(np.array(image))
    except ValueError:
        print("transform should be an image transformation that works on PIL image")
