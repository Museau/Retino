import math
import copy

import pytest
import torch

from torchvision.transforms import ToTensor

from data.data_loader import Kaggle2015, get_dataloader
from utils.checkpoint import set_seed_and_random_states


# Datset splits lenght
DATASET_SPLITS_LENGHT = {
    "train": 35116,
    "valid": 10906,
    "test": 42670}

KAGGLE_CFG = {'resolution': 512,
              'name': 'kaggle2015',
              'sample_size': -1,
              'use_both_eyes': False,
              'target_name': 'screening_level',
              'feature_scaling': 'MaskedStandardization',
              'train': {'n_views': 1,
                        'filter_target': False,
                        'augmentation': {'apply_prob': 1.0,
                                         'rotation_type': 'RightAngle',
                                         'different_transform_per_view': False},
                        'batch_size': 4,
                        'accumulated_batch_size': None,
                        'num_workers': 0,
                        },
              'eval': {'n_views': 1,
                       'apply_train_augmentations': False,
                       'batch_size': 2,
                       'num_workers': 0
                       }
              }

CUDA = {'deterministic': True, 'benchmark': False}


@pytest.mark.kaggle2015
def test_kaggle2015_data_folder(kaggle2015_folder):
    """Test kaggle2015_folder is a directory that contain the wanted files."""
    Kaggle2015(kaggle2015_folder, KAGGLE_CFG)


@pytest.mark.kaggle2015
def test_kaggle2015_invalid_data_folder():
    """Test invalid data_folder raise Value Error."""
    with pytest.raises(ValueError):
        Kaggle2015("qbfifbqofu", KAGGLE_CFG)


@pytest.mark.kaggle2015
@pytest.mark.parametrize("input_size", [512, 1024])
def test_kaggle2015_input_size(kaggle2015_folder, input_size):
    """Test input size is in [512, 1024]."""
    cfg = copy.deepcopy(KAGGLE_CFG)
    cfg['resolution'] = input_size
    Kaggle2015(kaggle2015_folder, cfg)


@pytest.mark.kaggle2015
@pytest.mark.parametrize("input_size", ["512", -134.325, -32, 124.424, 2048])
def test_kaggle2015_invalide_input_size(kaggle2015_folder, input_size):
    """Test invalid input size raise ValueError."""
    with pytest.raises(ValueError):
        cfg = copy.deepcopy(KAGGLE_CFG)
        cfg['resolution'] = input_size
        Kaggle2015(kaggle2015_folder, cfg)


@pytest.mark.kaggle2015
@pytest.mark.parametrize("n_views", [1, 2, 3])
def test_kaggle2015_n_views(kaggle2015_folder, n_views):
    """Test n_views valid options are 1, 2 or 3."""
    cfg = copy.deepcopy(KAGGLE_CFG)
    cfg['train']['n_views'] = n_views
    cfg['eval']['n_views'] = n_views
    Kaggle2015(kaggle2015_folder, cfg)


@pytest.mark.kaggle2015
@pytest.mark.parametrize("n_views", ["1", -2, 0, 1.43, 4, 100])
def test_kaggle2015_invalide_n_views(kaggle2015_folder, n_views):
    """Test invalid n_views values raised ValueError."""
    with pytest.raises(ValueError):
        cfg = copy.deepcopy(KAGGLE_CFG)
        cfg['train']['n_views'] = n_views
        cfg['eval']['n_views'] = n_views
        Kaggle2015(kaggle2015_folder, cfg)


@pytest.mark.kaggle2015
@pytest.mark.parametrize("split", ["train", "valid", "test"])
def test_kaggle2015_split(kaggle2015_folder, split):
    """Test split must be in ["train", "valid", "test"]."""
    Kaggle2015(kaggle2015_folder, KAGGLE_CFG, split=split)


@pytest.mark.kaggle2015
@pytest.mark.parametrize("split", ["qfofq", 124])
def test_kaggle2015_invalid_split(kaggle2015_folder, split):
    """Test invalid split values raise ValueError."""
    with pytest.raises(ValueError):
        Kaggle2015(kaggle2015_folder, KAGGLE_CFG, split=split)


@pytest.mark.kaggle2015
@pytest.mark.parametrize("split", ["train", "valid", "test"])
@pytest.mark.parametrize("sample_size", [-1, 50, 100, "full"])
def test_kaggle2015_sample_size(kaggle2015_folder, split, sample_size):
    """Test sample size can be -1 or in range(1, len(dataset_set)."""
    if sample_size == "full":
        sample_size = DATASET_SPLITS_LENGHT[split]
    cfg = copy.deepcopy(KAGGLE_CFG)
    cfg['sample_size'] = sample_size
    Kaggle2015(kaggle2015_folder, cfg, split=split)


@pytest.mark.kaggle2015
@pytest.mark.parametrize("split", ["train", "valid", "test"])
@pytest.mark.parametrize("sample_size", [-10, 0, 575859])
def test_kaggle2015_invalid_sample_size(kaggle2015_folder, split, sample_size):
    """Test invalid sample values raise ValueError."""
    with pytest.raises(ValueError):
        cfg = copy.deepcopy(KAGGLE_CFG)
        cfg['sample_size'] = sample_size
        Kaggle2015(kaggle2015_folder, cfg)


@pytest.mark.kaggle2015
@pytest.mark.parametrize("split", ["train", "valid", "test"])
@pytest.mark.parametrize("sample_size", [-1, 50, 100, "full"])
def test_kaggle2015_sample_size_equal_len_dataset(kaggle2015_folder, split, sample_size):
    """Test that train sample size is equal to len(dataset_set) except when it's -1.
    In that case, len(data_set) == # train samples."""
    if sample_size == "full":
        sample_size = DATASET_SPLITS_LENGHT[split]

    cfg = copy.deepcopy(KAGGLE_CFG)
    cfg['sample_size'] = sample_size
    data_set = Kaggle2015(kaggle2015_folder, cfg, split=split)
    if sample_size != -1:
        assert len(data_set) == sample_size, "len(data_set) is not equal to sample_size"
    else:
        assert len(data_set) == DATASET_SPLITS_LENGHT[split], "With sample_size = -1, len(data_set) is different than # samples"


@pytest.mark.kaggle2015
def test_kaggle2015_transform(kaggle2015_folder):
    """Test transform argument."""
    Kaggle2015(kaggle2015_folder, KAGGLE_CFG, transform=ToTensor())


@pytest.mark.kaggle2015
@pytest.mark.parametrize("transform", ["eqfq", None])
def test_kaggle2015_invalide_transform(kaggle2015_folder, transform):
    """Test invalide transform argument raised ValueError."""
    with pytest.raises(ValueError):
        Kaggle2015(kaggle2015_folder, KAGGLE_CFG, transform=transform)


@pytest.mark.kaggle2015
@pytest.mark.parametrize("input_size", [512, 1024])
@pytest.mark.parametrize("n_views", [1, 3])
@pytest.mark.parametrize("split", ["train", "valid", "test"])
def test_kaggle2015_arguments(kaggle2015_folder, input_size, n_views, split):
    """Check self corresponds to what's given as arguments."""
    cfg = copy.deepcopy(KAGGLE_CFG)
    cfg['resolution'] = input_size
    cfg['train']['n_views'] = n_views
    cfg['eval']['n_views'] = n_views
    data_set = Kaggle2015(kaggle2015_folder, cfg, split=split)
    assert data_set._n_views == n_views
    assert isinstance(data_set._labels, dict)
    assert isinstance(data_set._patient_ids, list)
    assert str(input_size) in data_set.img_dir
    if split == "train":
        assert "train" in data_set.img_dir
    else:
        assert "test" in data_set.img_dir


@pytest.mark.kaggle2015
def test_kaggle2015_outputs(kaggle2015_folder):
    """Check that sample contain the expected keys for images and masks."""
    data_set = Kaggle2015(kaggle2015_folder, KAGGLE_CFG)
    it = iter(data_set)
    sample = next(it)
    for k in ['inputs', 'masks', 'target']:
        assert k in sample, f"{k} should be a sample key"


@pytest.mark.kaggle2015
@pytest.mark.parametrize("input_size", [512, 1024])
@pytest.mark.parametrize("n_views", [1, 3])
def test_kaggle2015_image_outputs_type(kaggle2015_folder, input_size, n_views):
    """Check that type of image outputs is list."""
    cfg = copy.deepcopy(KAGGLE_CFG)
    cfg['resolution'] = input_size
    cfg['train']['n_views'] = n_views
    cfg['eval']['n_views'] = n_views
    data_set = Kaggle2015(kaggle2015_folder, cfg)
    it = iter(data_set)
    sample = next(it)
    for eye in sample['inputs']:
        assert isinstance(eye, list), "sample[inputs][0] should be a list"


@pytest.mark.kaggle2015
@pytest.mark.parametrize("input_size", [512, 1024])
@pytest.mark.parametrize("n_views", [1, 3])
def test_kaggle2015_image_outputs_len(kaggle2015_folder, input_size, n_views):
    """Check that len image outputs correspond to the number of views."""
    cfg = copy.deepcopy(KAGGLE_CFG)
    cfg['resolution'] = input_size
    cfg['train']['n_views'] = n_views
    cfg['eval']['n_views'] = n_views
    data_set = Kaggle2015(kaggle2015_folder, cfg)
    it = iter(data_set)
    sample = next(it)
    for eye in sample['inputs']:
        assert len(eye) == n_views, "len(sample[inputs][0]) should be equal to n_views"


@pytest.mark.kaggle2015
@pytest.mark.parametrize("input_size", [512, 1024])
def test_kaggle2015_image_outputs_size(kaggle2015_folder, input_size):
    """Check that image outputs size."""
    cfg = copy.deepcopy(KAGGLE_CFG)
    cfg['train']['n_views'] = 3
    cfg['resolution'] = input_size
    data_set = Kaggle2015(kaggle2015_folder, cfg)

    it = iter(data_set)
    sample = next(it)
    for eye in sample['inputs']:
        for i in range(cfg['train']['n_views']):
            assert eye[i].size() == (3, input_size, input_size)


@pytest.mark.kaggle2015
@pytest.mark.parametrize("input_size", [512, 1024])
def test_kaggle2015_image_outputs_values(kaggle2015_folder, input_size):
    """Check that image are the same when more than 1 view."""
    cfg = copy.deepcopy(KAGGLE_CFG)
    cfg['train']['n_views'] = 3
    cfg['resolution'] = input_size
    data_set = Kaggle2015(kaggle2015_folder, cfg)

    it = iter(data_set)
    sample = next(it)
    for eye in sample['inputs']:
        im_1 = eye[0]
        for i in range(cfg['train']['n_views']):
            im_2 = eye[i]
            assert torch.eq(im_1, im_2).all(), "tensor contain in sample should all be equal"


@pytest.mark.kaggle2015
@pytest.mark.parametrize("n_views", [1, 3])
def test_kaggle2015_mask_output_size(kaggle2015_folder, n_views):
    """Check mask size."""
    cfg = copy.deepcopy(KAGGLE_CFG)
    cfg['train']['n_views'] = n_views
    data_set = Kaggle2015(kaggle2015_folder, cfg)
    it = iter(data_set)
    sample = next(it)
    for mask in sample['masks']:
        assert mask.size() == (n_views, )


@pytest.mark.kaggle2015
def test_kaggle2015_data_does_not_grow(kaggle2015_folder):
    """Check that self._labels does not grow while iterate over the data set."""
    data_set = Kaggle2015(kaggle2015_folder, KAGGLE_CFG)
    data_ori = copy.deepcopy(data_set._labels)
    it = iter(data_set)
    for i in range(3):
        next(it)
    assert data_ori == data_set._labels


@pytest.mark.kaggle2015
def test_kaggle2015_train_valid_test_intersection(kaggle2015_folder):
    """Test Kaggle 2015 dataset train/valid/test intersection."""
    train_dataset = Kaggle2015(kaggle2015_folder, KAGGLE_CFG, split="train")
    train_dataset_patient_ids = set(train_dataset._labels.keys())
    valid_dataset = Kaggle2015(kaggle2015_folder, KAGGLE_CFG, split="valid")
    valid_dataset_patient_ids = set(valid_dataset._labels.keys())
    test_dataset = Kaggle2015(kaggle2015_folder, KAGGLE_CFG, split="test")
    test_dataset_patient_ids = set(test_dataset._labels.keys())
    assert len(train_dataset_patient_ids & valid_dataset_patient_ids) == 0, "No intersection should exist between train and valid"
    assert len(train_dataset_patient_ids & test_dataset_patient_ids) == 0, "No intersection should exist between train and test"
    assert len(valid_dataset_patient_ids & test_dataset_patient_ids) == 0, "No intersection should exist between valid and test"


@pytest.mark.kaggle2015
def test_get_dataloader_train_split_outputs(kaggle2015_folder):
    """Check that train data loader return right/left image and mask."""
    cfg = copy.deepcopy(KAGGLE_CFG)
    train_loader = get_dataloader(kaggle2015_folder, cfg, train=True)['train']
    it = iter(train_loader)
    batch = next(it)
    for k in ['inputs', 'masks', 'target']:
        assert k in batch, f"{k} should be a batch key"


@pytest.mark.kaggle2015
def test_get_dataloader_valid_split_outputs(kaggle2015_folder):
    """Check that valid data loader return right/left image and mask."""
    cfg = copy.deepcopy(KAGGLE_CFG)
    valid_loader = get_dataloader(kaggle2015_folder, cfg, train=True)['valid']
    it = iter(valid_loader)
    batch = next(it)
    for k in ['inputs', 'masks', 'target']:
        assert k in batch, f"{k} should be a batch key"


@pytest.mark.kaggle2015
def test_get_dataloader_test_split_outputs(kaggle2015_folder):
    """Check that test data loader return right/left image and mask."""
    cfg = copy.deepcopy(KAGGLE_CFG)
    test_loader = get_dataloader(kaggle2015_folder, cfg, train=False)['test']
    it = iter(test_loader)
    batch = next(it)
    for k in ['inputs', 'masks', 'target']:
        assert k in batch, f"{k} should be a batch key"


@pytest.mark.kaggle2015
@pytest.mark.parametrize("train_batch_size", [1, 4])
def test_get_dataloader_train_image_batch_outputs_size(kaggle2015_folder, train_batch_size):
    """Check batch train outputs size."""
    cfg = copy.deepcopy(KAGGLE_CFG)
    cfg['train']['n_views'] = 3
    cfg['train']['batch_size'] = train_batch_size
    train_loader = get_dataloader(kaggle2015_folder, cfg, train=True)['train']
    it = iter(train_loader)
    batch = next(it)
    for eye in batch['inputs']:
        for i in range(cfg['train']['n_views']):
            assert eye[i].size() == (train_batch_size, 3, cfg['resolution'], cfg['resolution'])


@pytest.mark.kaggle2015
@pytest.mark.parametrize("valid_batch_size", [1, 4])
def test_get_dataloader_valid_image_batch_outputs_size(kaggle2015_folder, valid_batch_size):
    """Check batch valid outputs size."""
    cfg = copy.deepcopy(KAGGLE_CFG)
    cfg['eval']['n_views'] = 3
    cfg['eval']['batch_size'] = valid_batch_size
    valid_loader = get_dataloader(kaggle2015_folder, cfg, train=True)['valid']
    it = iter(valid_loader)
    batch = next(it)
    for eye in batch['inputs']:
        for i in range(cfg['train']['n_views']):
            assert eye[i].size() == (valid_batch_size, 3, cfg['resolution'], cfg['resolution'])


@pytest.mark.kaggle2015
@pytest.mark.parametrize("valid_batch_size", [1, 4])
def test_get_dataloader_test_image_batch_outputs_size(kaggle2015_folder, valid_batch_size):
    """Check batch test outputs size."""
    cfg = copy.deepcopy(KAGGLE_CFG)
    cfg['eval']['n_views'] = 3
    cfg['eval']['batch_size'] = valid_batch_size
    test_loader = get_dataloader(kaggle2015_folder, cfg, train=False)['test']
    it = iter(test_loader)
    batch = next(it)
    for eye in batch['inputs']:
        for i in range(cfg['train']['n_views']):
            assert eye[i].size() == (valid_batch_size, 3, cfg['resolution'], cfg['resolution'])


@pytest.mark.kaggle2015
@pytest.mark.parametrize("apply_prob", [0., 1.])
def test_get_dataloader_train_image_equal_transforms_multiviews(kaggle2015_folder, apply_prob):
    """Check batch train transform return all elements with same transform for
    a sample."""
    cfg = copy.deepcopy(KAGGLE_CFG)
    cfg['resolution'] = 512
    cfg['train']['n_views'] = 3
    cfg['train']['augmentation']['apply_prob'] = apply_prob
    train_loader = get_dataloader(kaggle2015_folder, cfg)['train']
    it = iter(train_loader)
    batch = next(it)
    for eye in batch['inputs']:
        im_1 = eye[0]
        for i in range(cfg['train']['n_views']):
            assert torch.eq(im_1, eye[i]).all()


@pytest.mark.kaggle2015
def test_get_dataloader_train_image_no_transforms(kaggle2015_folder):
    """Check batch train with no transforms."""
    cfg = copy.deepcopy(KAGGLE_CFG)
    cfg['train']['augmentation']['apply_prob'] = 0.
    # Set seed
    set_seed_and_random_states(1234, None, CUDA)
    train_loader = get_dataloader(kaggle2015_folder, cfg)['train']
    it = iter(train_loader)
    im_1 = next(it)["inputs"][0][0]
    # Reset seed to original seed
    set_seed_and_random_states(1234, None, CUDA)
    train_loader = get_dataloader(kaggle2015_folder, cfg)['train']
    it = iter(train_loader)
    im_2 = next(it)["inputs"][0][0]
    assert torch.eq(im_1, im_2).all()


@pytest.mark.kaggle2015
def test_get_dataloader_train_image_transforms(kaggle2015_folder):
    """Check batch train transforms."""
    cfg = copy.deepcopy(KAGGLE_CFG)
    cfg['train']['augmentation']['apply_prob'] = 0.
    # Set seed
    set_seed_and_random_states(1234, None, CUDA)
    train_loader = get_dataloader(kaggle2015_folder, cfg)['train']
    it = iter(train_loader)
    im_1 = next(it)["inputs"][0][0]
    # Reset seed to original seed
    set_seed_and_random_states(1234, None, CUDA)
    cfg_transform = copy.deepcopy(cfg)
    cfg_transform['train']['augmentation']['apply_prob'] = 1.
    print(cfg_transform)
    train_loader = get_dataloader(kaggle2015_folder, cfg_transform)['train']
    it = iter(train_loader)
    im_2 = next(it)["inputs"][0][0]
    assert not torch.eq(im_1, im_2).all()


@pytest.mark.kaggle2015
@pytest.mark.parametrize("apply_prob", [0., 1.])
def test_get_dataloader_valid_image_transforms(kaggle2015_folder, apply_prob):
    """Check batch valid transforms."""
    cfg = copy.deepcopy(KAGGLE_CFG)
    cfg['train']['augmentation']['apply_prob'] = 0.
    # Set seed
    set_seed_and_random_states(1234, None, CUDA)
    valid_loader = get_dataloader(kaggle2015_folder, cfg)['valid']
    it = iter(valid_loader)
    im_1 = next(it)["inputs"][0][0]
    # Reset seed to original seed
    set_seed_and_random_states(1234, None, CUDA)
    cfg_transform = copy.deepcopy(cfg)
    cfg_transform['train']['augmentation']['apply_prob'] = apply_prob
    valid_loader = get_dataloader(kaggle2015_folder, cfg_transform)['valid']
    it = iter(valid_loader)
    im_2 = next(it)["inputs"][0][0]
    assert torch.eq(im_1, im_2).all()


@pytest.mark.kaggle2015
@pytest.mark.parametrize("apply_prob", [0., 1.])
def test_get_dataloader_test_image_transforms(kaggle2015_folder, apply_prob):
    """Check batch test transforms."""
    cfg = copy.deepcopy(KAGGLE_CFG)
    cfg['train']['augmentation']['apply_prob'] = 0.
    # Set seed
    set_seed_and_random_states(1234, None, CUDA)
    test_loader = get_dataloader(kaggle2015_folder, cfg, train=False)['test']
    it = iter(test_loader)
    im_1 = next(it)["inputs"][0][0]
    # Reset seed to original seed
    set_seed_and_random_states(1234, None, CUDA)
    cfg_transform = copy.deepcopy(cfg)
    cfg_transform['train']['augmentation']['apply_prob'] = apply_prob
    test_loader = get_dataloader(kaggle2015_folder, cfg_transform, train=False)['test']
    it = iter(test_loader)
    im_2 = next(it)["inputs"][0][0]
    assert torch.eq(im_1, im_2).all()


def test_get_dataloader_train_n_batch(kaggle2015_folder):
    """Check that train data loader return right number of batch. Should not
    return truncated batch."""
    cfg = copy.deepcopy(KAGGLE_CFG)
    cfg['sample_size'] = 357
    cfg['train']['batch_size'] = 4
    train_loader = get_dataloader(kaggle2015_folder, cfg)['train']
    assert len(train_loader) == math.floor(cfg['sample_size'] / cfg['train']['batch_size'])


def test_get_dataloader_valid_n_batch(kaggle2015_folder):
    """Check that valid data loader return right number of batch. Can contain
    truncated batch.
    """
    cfg = copy.deepcopy(KAGGLE_CFG)
    cfg['sample_size'] = 357
    cfg['eval']['batch_size'] = 4
    valid_loader = get_dataloader(kaggle2015_folder, cfg)['valid']
    assert len(valid_loader) == cfg['sample_size'] // cfg['eval']['batch_size']


def test_get_dataloader_test_n_batch(kaggle2015_folder):
    """Check that test data loader return right number of batch. Can contain
    truncated batch.
    """
    cfg = copy.deepcopy(KAGGLE_CFG)
    cfg['sample_size'] = 357
    cfg['eval']['batch_size'] = 4
    test_loader = get_dataloader(kaggle2015_folder, cfg, train=False)['test']
    assert len(test_loader) == cfg['sample_size'] // cfg['eval']['batch_size']
