import random

import pytest
import numpy as np
import torch

from utils.checkpoint import set_seed_and_random_states

CUDA = {'deterministic': True, 'benchmark': False}


def save_random_states():
    random_states = {"np_state": np.random.get_state(),
                     "python_state": random.getstate(),
                     "torch_state": torch.get_rng_state()}
    if torch.cuda.is_available():
        random_states["torch_state_cuda"] = torch.cuda.get_rng_state()
    return random_states


@pytest.mark.parametrize("seed", [1234, 2345, 3456])
def test_set_seed_random(seed):
    set_seed_and_random_states(seed, None, CUDA)
    random_float = random.random()

    # Reset seed
    set_seed_and_random_states(seed, None, CUDA)
    assert random_float == random.random()


@pytest.mark.parametrize("seed", [1234, 2345, 3456])
def test_set_seed_numpy(seed):
    set_seed_and_random_states(seed, None, CUDA)
    numpy_random_float = np.random.random_sample()

    # Reset seed
    set_seed_and_random_states(seed, None, CUDA)
    assert numpy_random_float == np.random.random_sample()


@pytest.mark.parametrize("seed", [1234, 2345, 3456])
def test_set_seed_torch(seed):
    set_seed_and_random_states(seed, None, CUDA)
    torch_random_float = torch.rand(1)
    if torch.cuda.is_available():
        torch_cuda_random_float = torch.cuda.FloatTensor(1).normal_()

    # Reset seed
    set_seed_and_random_states(seed, None, CUDA)
    assert torch.eq(torch_random_float, torch.rand(1))
    if torch.cuda.is_available():
        assert torch.eq(torch_cuda_random_float, torch.cuda.FloatTensor(1).normal_())


@pytest.mark.parametrize("seed", [1234, 2345, 3456])
def test_set_seed_from_state_random(seed):
    set_seed_and_random_states(seed, None, CUDA)
    # Save current seed
    random_states = save_random_states()
    random_float = random.random()

    # Reset seed to saved seed
    set_seed_and_random_states(seed, random_states, CUDA)
    assert random_float == random.random()


@pytest.mark.parametrize("seed", [1234, 2345, 3456])
def test_set_seed_from_state_numpy(seed):
    set_seed_and_random_states(seed, None, CUDA)
    # Save current seed
    random_states = save_random_states()
    numpy_random_float = np.random.random_sample()

    # Reset seed to saved seed
    set_seed_and_random_states(seed, random_states, CUDA)
    assert numpy_random_float == np.random.random_sample()


@pytest.mark.parametrize("seed", [1234, 2345, 3456])
def test_set_seed_from_state_torch(seed):
    set_seed_and_random_states(seed, None, CUDA)
    # Save current seed
    random_states = save_random_states()
    torch_random_float = torch.rand(1)
    if torch.cuda.is_available():
        torch_cuda_random_float = torch.cuda.FloatTensor(1).normal_()

    # Reset seed to saved seed
    set_seed_and_random_states(seed, random_states, CUDA)
    assert torch.eq(torch_random_float, torch.rand(1))
    if torch.cuda.is_available():
        assert torch.eq(torch_cuda_random_float, torch.cuda.FloatTensor(1).normal_())
