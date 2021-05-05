import copy

import pytest

from train import batch_loop


CFG = {'resolution': 10,
       'name': 'fake',
       'sample_size': 10,
       'use_both_eyes': False,
       'target_name': 'screening_level',
       'feature_scaling': 'MaskedStandardization',
       'train': {'n_views': 1,
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

MODEL_CFG = {"pretrain": "",
             "base": {"name": "resnet18", "norm_layer": {"name": "BatchNorm2d"}},
             "dropout_rate": 0.0}


@pytest.fixture
def loader():
    from data.data_loader import get_dataloader

    return get_dataloader("fake", CFG, train=False)['test']


@pytest.fixture
def model():
    from torch import nn
    num_classes = 1

    net = nn.Linear(10, num_classes)
    net.num_classes = num_classes

    return net


@pytest.fixture
def opt(model):
    from torch.optim import SGD

    return SGD(model.parameters(), lr=1)


@pytest.fixture
def loss():
    from torch.nn import BCEWithLogitsLoss

    return BCEWithLogitsLoss()


@pytest.fixture
def logger(model):
    from utils.logging import GradientLogger

    return GradientLogger(model, 0)


@pytest.mark.parametrize("target_name", ["DR_level", "screening_level"])
@pytest.mark.parametrize("acc_batch_size", [1, 3, 21])  # With fixture loader batch_size=2
def test_batch_loop_invalid_acc_train_batch_size(loader, model, opt, loss, logger, target_name, acc_batch_size):
    with pytest.raises(ValueError):
        batch_loop(loader,
                   model,
                   opt,
                   loss,
                   "cpu",
                   "TRAIN",
                   grad_logger=logger,
                   acc_batch_size=acc_batch_size)


@pytest.mark.parametrize("mode", ["VALID", "TEST"])
@pytest.mark.parametrize("target_name", ["DR_level", "screening_level"])
def test_batch_loop_invalid_batch_accumulation(loader, model, opt, loss, target_name, mode):
    with pytest.raises(ValueError):
        batch_loop(loader,
                   model,
                   opt,
                   loss,
                   "cpu",
                   mode,
                   acc_batch_size=4)


@pytest.mark.parametrize("a, b", [(1, 12), (2, 12), (3, 12), (8, 8), (3, 3), (1, 3)])
def test_batch_loop_batch_accumulation_equivalence(a, b, loss, logger):
    import warnings
    warnings.filterwarnings("ignore")

    from torch.optim import SGD

    from data.data_loader import get_dataloader
    from model.factory import build_model

    # Setting up train config
    device = "cpu"
    mode = "TRAIN"

    # Setting up dataloader
    data_cfg_no_acc = copy.deepcopy(CFG)
    data_cfg_no_acc['resolution'] = 244  # Min size for resnet18
    data_cfg_no_acc['eval']['batch_size'] = max(a, b)
    data_cfg_no_acc['sample_size'] = data_cfg_no_acc['eval']['batch_size'] * 2

    data_cfg_acc = copy.deepcopy(data_cfg_no_acc)
    data_cfg_acc['eval']['batch_size'] = min(a, b)

    # Creating dataloader and models
    loader_no_acc = get_dataloader(data_cfg_no_acc['name'], data_cfg_no_acc, train=False)['test']
    loader_acc = get_dataloader(data_cfg_acc['name'], data_cfg_acc, train=False)['test']
    model_no_acc = build_model(MODEL_CFG, 1, data_cfg_no_acc['use_both_eyes'], device)
    model_acc = copy.deepcopy(model_no_acc)
    model_no_acc.train()
    model_acc.train()

    if a == b:
        assert len(loader_no_acc) == len(loader_acc)

    # Train model and compute loss for 1 epoch
    # WITHOUT batch accumulation
    _, _, loss_no_acc = batch_loop(loader_no_acc,
                                   model_no_acc,
                                   SGD(model_no_acc.parameters(), lr=1),
                                   loss,
                                   device,
                                   mode,
                                   grad_logger=logger,
                                   acc_batch_size=None)

    # Train model and compute loss for 1 epoch
    # WITH batch accumulation
    _, _, loss_acc = batch_loop(loader_acc,
                                model_acc,
                                SGD(model_acc.parameters(), lr=1),
                                loss,
                                device,
                                mode,
                                grad_logger=logger,
                                acc_batch_size=data_cfg_no_acc['eval']['batch_size'])

    assert loss_no_acc == pytest.approx(loss_acc, rel=1e-5)


@pytest.mark.parametrize("batch_size, acc_batch_size, sample_size", [(2, 6, 10), (1, 3, 10)])
def test_batch_loop_batch_accumulation_drop_last(logger, batch_size, acc_batch_size, sample_size, opt, loss):
    import warnings
    warnings.filterwarnings("ignore")

    from torch.optim import SGD

    from data.data_loader import get_dataloader
    from model.factory import build_model

    # Setting up train config
    device = "cpu"
    mode = "TRAIN"

    assert acc_batch_size % batch_size == 0
    assert sample_size % batch_size == 0
    assert sample_size % acc_batch_size != 0

    # Setting up dataloader
    data_cfg = copy.deepcopy(CFG)
    data_cfg['resolution'] = 244  # Min size for resnet18
    data_cfg['eval']['batch_size'] = batch_size
    data_cfg['train']['accumulated_batch_size'] = acc_batch_size
    data_cfg['sample_size'] = sample_size

    # Creating dataloader and models
    loader = get_dataloader(data_cfg['name'], data_cfg, train=False)['test']
    model = build_model(MODEL_CFG, 1, data_cfg['use_both_eyes'], device)

    print(model.base_net.layer1[0].bn1.num_batches_tracked)
    batch_loop(loader,
               model,
               SGD(model.parameters(), lr=1),
               loss,
               device,
               mode,
               grad_logger=logger,
               acc_batch_size=acc_batch_size)

    batch_seen = model.base_net.layer1[0].bn1.num_batches_tracked.item()

    nb_accumulated_batch = sample_size // acc_batch_size
    nb_batch_per_accumulated_batch = acc_batch_size / batch_size
    expected_batch_seen = nb_accumulated_batch * nb_batch_per_accumulated_batch

    assert batch_seen == expected_batch_seen
