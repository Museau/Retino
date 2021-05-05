"""
Note that those tests test CombineFeatures with a base net beeing a simple
linear network. Further tests should be done using more complex base net.
"""

import copy

import pytest
import torch
import torch.nn as nn

from torch.nn import BCEWithLogitsLoss

from utils.checkpoint import set_seed_and_random_states
from model.combine_features import CombineFeatures


def get_model_weight(model):
    """Get model weight in a dict."""
    param_dict = {name: param.data for name, param in model.named_parameters() if param.requires_grad}
    return param_dict


def get_model_weight_grad(model):
    """Get model weight grad in a dict."""
    param_grad_dict = {name: param.grad for name, param in model.named_parameters() if param.requires_grad}
    return param_grad_dict


def compare_dict_keys(dict_1, dict_2):
    """Check that two dict have the same keys."""
    return set(dict_1.keys()) == set(dict_2.keys())


def compare_models_weight(model_1, model_2):
    """Compare two model weights and return True if close."""
    param_dict_model_1 = get_model_weight(model_1)
    param_dict_model_2 = get_model_weight(model_2)
    assert compare_dict_keys(param_dict_model_1, param_dict_model_2)
    return all([torch.isclose(param_dict_model_1[k], param_dict_model_2[k]).all() for k in param_dict_model_1.keys()])


def compare_models_weight_grad(model_1, model_2):
    """Compare two model weight grads and return True if close."""
    param_grad_dict_model_1 = get_model_weight_grad(model_1)
    param_grad_dict_model_2 = get_model_weight_grad(model_2)
    compare_dict_keys(param_grad_dict_model_1, param_grad_dict_model_2)
    return all([torch.isclose(param_grad_dict_model_1[k], param_grad_dict_model_2[k]).all() for k in param_grad_dict_model_1.keys()])


@pytest.fixture
def set_up_models_optimizers_loss():
    """Set up 2 identical models, optimizers and a loss function. Base net,
    dropout and batch_size can be pass as parameters."""
    def _set_up(base_net=nn.Linear(32 * 32, 128), dropout=0.0, batch_size=1):
        # Setting device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Set seed
        cuda = {'deterministic': True, 'benchmark': False}
        set_seed_and_random_states(1234, None, cuda)

        # Define data
        # 1 view input
        dtype = torch.float32
        set_up = {}
        set_up['x_1view'] = [torch.randn((batch_size, 1, 32, 32), dtype=dtype, device=device)]
        # Flatten the input for now
        set_up['x_1view'] = [view.view(batch_size, -1) for view in set_up['x_1view']]
        set_up['mask_1view'] = torch.ones((batch_size, 1), dtype=dtype, device=device)
        # 3 views input
        set_up['x_3view'] = set_up['x_1view'] * 3
        # Flatten the inputs for now
        set_up['x_3view'] = [view.view(batch_size, -1) for view in set_up['x_3view']]
        set_up['mask_3view'] = torch.ones((batch_size, 3), dtype=dtype, device=device)
        # Target
        set_up['target'] = torch.ones((batch_size, 1), dtype=dtype, device=device)

        # Set up models
        base_net.emb_size = base_net.out_features
        set_up['model_1view'] = CombineFeatures(base_net, False, dropout, 1)
        set_up['model_1view'].to(device)
        set_up['model_3view'] = copy.deepcopy(set_up['model_1view'])

        # Set up optimizers
        lr = 1.
        momentum = 0.
        set_up['optimizer_1view'] = torch.optim.SGD(set_up['model_1view'].parameters(), lr=lr, momentum=momentum)
        set_up['optimizer_3view'] = torch.optim.SGD(set_up['model_3view'].parameters(), lr=lr, momentum=momentum)
        set_up['optimizer_1view'].zero_grad()
        set_up['optimizer_3view'].zero_grad()

        # Set up loss_function
        set_up['loss_function'] = BCEWithLogitsLoss()

        return set_up
    return _set_up


@pytest.mark.combine_features
def test_compare_1view_vs_3view_model(set_up_models_optimizers_loss):
    """Check that model 1 view and 3 view are the same at the beginning."""
    set_up = set_up_models_optimizers_loss()
    assert compare_models_weight(set_up['model_1view'], set_up['model_3view']), "model 1 view and 3 views weight does not match"


@pytest.mark.combine_features
def test_compare_1view_vs_3view_x(set_up_models_optimizers_loss):
    """Check that the inputs used for 1view and 3view are the same."""
    set_up = set_up_models_optimizers_loss()
    for i in range(len(set_up['x_3view'])):
        assert torch.eq(set_up['x_1view'][0], set_up['x_3view'][i]).all(), "input set_up['x_1view'] and set_up['x_3view'] differ"


@pytest.mark.combine_features
@pytest.mark.parametrize("dropout, batch_size", [(0., 1), (0., 4)])
def test_compare_1view_vs_3view_out(dropout, batch_size, set_up_models_optimizers_loss):
    """ Check that the output after one forward of model 1 view and 3 view are
    the same."""
    set_up = set_up_models_optimizers_loss(dropout=dropout, batch_size=batch_size)

    out_1view = set_up['model_1view']([set_up['x_1view']],
                                      [set_up['mask_1view']])

    out_3view = set_up['model_1view']([set_up['x_3view']],
                                      [set_up['mask_3view']])

    assert torch.isclose(out_1view, out_3view).all(), "output of model with set_up['x_1view'] and set_up['x_3view'] differ"


@pytest.mark.combine_features
@pytest.mark.parametrize("dropout, batch_size", [(0.5, 1), (0.5, 4)])
def test_compare_1view_vs_3view_out_with_dropout(dropout, batch_size, set_up_models_optimizers_loss):
    """ Check that the output after one forward of model 1 view and 3 view are
    not the same when dropout is activated."""
    set_up = set_up_models_optimizers_loss(dropout=dropout, batch_size=batch_size)

    out_1view = set_up['model_1view']([set_up['x_1view']],
                                      [set_up['mask_1view']])

    out_3view = set_up['model_1view']([set_up['x_3view']],
                                      [set_up['mask_3view']])

    assert not torch.isclose(out_1view, out_3view).all(), "output of model with set_up['x_1view'] and set_up['x_3view'] differ"


@pytest.mark.combine_features
def test_compare_1view_vs_3view_loss(set_up_models_optimizers_loss):
    """Check that the loss computed using same target and model 1 view and
    3 view output are the same."""
    set_up = set_up_models_optimizers_loss()

    out_1view = set_up['model_1view']([set_up['x_1view']],
                                      [set_up['mask_1view']])

    out_3view = set_up['model_1view']([set_up['x_3view']],
                                      [set_up['mask_3view']])

    loss_1view = set_up['loss_function'](out_1view, set_up['target'])
    loss_3view = set_up['loss_function'](out_3view, set_up['target'])

    assert torch.isclose(loss_1view, loss_3view).all(), "loss_1view and loss_3view differ"


@pytest.mark.combine_features
def test_compare_1view_vs_3view_loss_backward(set_up_models_optimizers_loss):
    """Check that the model weights and gradients are the same after
    loss.backward() for model 1 view and model 3 view."""
    set_up = set_up_models_optimizers_loss()

    out_1view = set_up['model_1view']([set_up['x_1view']],
                                      [set_up['mask_1view']])

    out_3view = set_up['model_3view']([set_up['x_3view']],
                                      [set_up['mask_3view']])

    loss_1view = set_up['loss_function'](out_1view, set_up['target'])
    loss_3view = set_up['loss_function'](out_3view, set_up['target'])

    loss_1view.backward()
    loss_3view.backward()

    assert compare_models_weight(set_up['model_1view'], set_up['model_3view']), "model 1 view and 3 views weight does not match"
    assert compare_models_weight_grad(set_up['model_1view'], set_up['model_3view']), "model 1 view and 3 views weight grad does not match"


@pytest.mark.combine_features
def test_compare_1view_vs_3view_one_optimizer_step(set_up_models_optimizers_loss):
    """Check that after one optimizer step model weights and gradients match
    for model 1 view and 3 view."""
    set_up = set_up_models_optimizers_loss()

    out_1view = set_up['model_1view']([set_up['x_1view']],
                                      [set_up['mask_1view']])

    out_3view = set_up['model_3view']([set_up['x_3view']],
                                      [set_up['mask_3view']])

    loss_1view = set_up['loss_function'](out_1view, set_up['target'])
    loss_3view = set_up['loss_function'](out_3view, set_up['target'])

    loss_1view.backward()
    loss_3view.backward()

    set_up['optimizer_1view'].step()
    set_up['optimizer_1view'].zero_grad()
    set_up['optimizer_3view'].step()
    set_up['optimizer_3view'].zero_grad()

    assert compare_models_weight(set_up['model_1view'], set_up['model_3view']), "model 1 view and 3 views weight does not match"
    assert compare_models_weight_grad(set_up['model_1view'], set_up['model_3view']), "model 1 view and 3 views weight grad does not match"


@pytest.mark.combine_features
def test_compare_1view_vs_3view_one_optimizer_step_update_weight(set_up_models_optimizers_loss):
    """Check that weight are different for model 1 view between before/after
    one optimizer.step()."""
    set_up = set_up_models_optimizers_loss()

    set_up_model = copy.deepcopy(set_up['model_1view'])

    out_1view = set_up['model_1view']([set_up['x_1view']],
                                      [set_up['mask_1view']])

    out_3view = set_up['model_3view']([set_up['x_3view']],
                                      [set_up['mask_3view']])

    loss_1view = set_up['loss_function'](out_1view, set_up['target'])
    loss_3view = set_up['loss_function'](out_3view, set_up['target'])

    loss_1view.backward()
    loss_3view.backward()

    set_up['optimizer_1view'].step()
    set_up['optimizer_1view'].zero_grad()
    set_up['optimizer_3view'].step()
    set_up['optimizer_3view'].zero_grad()

    assert not compare_models_weight(set_up_model, set_up['model_1view']), "model 1 view and 3 views weight match"


@pytest.mark.combine_features
def test_compare_1view_vs_3view_ten_optimizer_step(set_up_models_optimizers_loss):
    """Check that after 10 optimizer.step() -i.e., 10 fprop + bprop
    model 1 view and 3 view still have the same weight."""
    set_up = set_up_models_optimizers_loss()

    for i in range(10):
        out_1view = set_up['model_1view']([set_up['x_1view']],
                                          [set_up['mask_1view']])

        out_3view = set_up['model_3view']([set_up['x_3view']],
                                          [set_up['mask_3view']])

        loss_1view = set_up['loss_function'](out_1view, set_up['target'])
        loss_3view = set_up['loss_function'](out_3view, set_up['target'])

        loss_1view.backward()
        loss_3view.backward()

        set_up['optimizer_1view'].step()
        set_up['optimizer_1view'].zero_grad()
        set_up['optimizer_3view'].step()
        set_up['optimizer_3view'].zero_grad()

    assert compare_models_weight(set_up['model_1view'], set_up['model_3view']), "model 1 view and 3 views weight does not match"
    assert compare_models_weight_grad(set_up['model_1view'], set_up['model_3view']), "model 1 view and 3 views weight grad does not match"


@pytest.mark.combine_features
def test_compare_1view_vs_3view_one_optimizer_step_eval(set_up_models_optimizers_loss):
    """Check that model 1 view and 3 view give the same output in eval mode
    after one optimizer.step()."""
    set_up = set_up_models_optimizers_loss()
    out_1view = set_up['model_1view']([set_up['x_1view']],
                                      [set_up['mask_1view']])

    out_3view = set_up['model_3view']([set_up['x_3view']],
                                      [set_up['mask_3view']])

    loss_1view = set_up['loss_function'](out_1view, set_up['target'])
    loss_3view = set_up['loss_function'](out_3view, set_up['target'])

    loss_1view.backward()
    loss_3view.backward()

    set_up['optimizer_1view'].step()
    set_up['optimizer_1view'].zero_grad()
    set_up['optimizer_3view'].step()
    set_up['optimizer_3view'].zero_grad()

    set_up['model_1view'].eval()
    set_up['model_3view'].eval()

    out_1view_eval = set_up['model_1view']([set_up['x_1view']],
                                           [set_up['mask_1view']])

    out_3view_eval = set_up['model_3view']([set_up['x_3view']],
                                           [set_up['mask_3view']])

    assert torch.isclose(out_1view_eval, out_3view_eval).all(), "evaluation output of model with set_up['x_1view'] and set_up['x_3view'] differ"
