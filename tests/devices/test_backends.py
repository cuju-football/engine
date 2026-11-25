# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks the spellings of a device and that `device='auto'` picks the GPU where it can, and that
the outputs depend on neither the backend nor the number of threads."""

import os

import pytest
import torch

import cuju
from conftest import CUDA, play_digest
from cuju._batch import Batch
from cuju._library_path import library_path


def test_threads_give_the_same_outputs():
    one = play_digest('5_vs_5_l4_r4', 'cpu', 60, environments=8, threads=1)
    several = play_digest('5_vs_5_l4_r4', 'cpu', 60, environments=8, threads=4)
    assert one == several


@pytest.mark.skipif(not CUDA, reason='no CUDA device or library')
@pytest.mark.parametrize('assets', ['5_vs_5_l4_r4', '11_vs_11_stochastic_l1_r1'])
def test_gpu_equals_cpu(assets):
    assert play_digest(assets, 'cuda', 60) == play_digest(assets, 'cpu', 60)


needs_gpu = pytest.mark.skipif(not CUDA, reason='no CUDA device or library')


def test_torch_device_of_the_cpu():
    for device in ('cpu', torch.device('cpu'), 'cpu:0'):
        batch = Batch('5_vs_5_l1_r0', environment_count=2, device=device)
        assert batch.device == 'cpu' and batch.torch_device == torch.device('cpu')
        assert batch.reset().device == torch.device('cpu')
        batch.close()


@needs_gpu
def test_an_indexed_gpu_gives_the_outputs_of_the_current_one():
    index = torch.cuda.current_device()
    for device in (f'cuda:{index}', torch.device('cuda', index)):
        batch = Batch('5_vs_5_l1_r0', environment_count=2, device=device)
        assert batch.device == 'cuda' and batch.torch_device == torch.device('cuda', index)
        assert batch.reset().device == batch.torch_device
        batch.close()
    expected = play_digest('5_vs_5_l4_r0', 'cuda', 20)
    assert play_digest('5_vs_5_l4_r0', f'cuda:{index}', 20) == expected
    assert play_digest('5_vs_5_l4_r0', torch.device('cuda', index), 20) == expected


@needs_gpu
def test_a_missing_gpu_is_refused():
    missing = torch.cuda.device_count()
    with pytest.raises(cuju.errors.DeviceError, match=f'from cuda:0 to cuda:{missing - 1}'):
        Batch('5_vs_5_l1_r0', environment_count=1, device=f'cuda:{missing}')


@pytest.mark.skipif(torch.cuda.is_available(), reason='a CUDA device is present')
def test_a_gpu_without_cuda_is_refused():
    with pytest.raises(cuju.errors.DeviceError, match="pass device='cpu'"):
        Batch('5_vs_5_l1_r0', environment_count=1, device='cuda:0')


def test_other_devices_are_refused():
    for device in ('meta', 'tpu', torch.device('meta'), None):
        with pytest.raises(cuju.errors.ChoiceError):
            Batch('5_vs_5_l1_r0', environment_count=1, device=device)
    assert library_path(torch.device('cpu')) == library_path('cpu')
    assert library_path('cuda:3') == library_path('cuda')


def expected_backend():
    gpu_library = os.path.isfile(library_path('cuda')) or os.environ.get('CUJU_CUDA_LIBRARY')
    return 'cuda' if torch.cuda.is_available() and gpu_library else 'cpu'


def test_auto_picks_a_backend():
    batch = Batch('5_vs_5_l1_r0', environment_count=2, device='auto')
    assert batch.device == expected_backend()
    assert batch.reset().device == batch.torch_device
    if batch.device == 'cuda':
        assert batch.torch_device.index == torch.cuda.current_device()
    batch.close()
    assert library_path('auto') == library_path(expected_backend())


def test_auto_without_a_gpu_takes_the_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    batch = Batch('5_vs_5_l1_r0', environment_count=2, device='auto')
    assert batch.device == 'cpu' and batch.torch_device == torch.device('cpu')
    batch.close()


def test_auto_plays_the_same_matches():
    assert play_digest('5_vs_5_l4_r0', 'auto', 20) == play_digest('5_vs_5_l4_r0', 'cpu', 20)
