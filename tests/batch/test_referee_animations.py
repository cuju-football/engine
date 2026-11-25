# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks GRF's referee animations, which play every stoppage in full, on both backends."""

import hashlib

import pytest

from conftest import CUDA, actions_at, tensor_bytes
from cuju._batch import Batch


def play(device, animations, steps=600, mask=None):
    """Returns the SHA-256 of every raw observation, reward and done flag of a batch's run."""
    batch = Batch(
        '11_vs_11_stochastic_l1_r1',
        environment_count=2,
        device=device,
        seed=3,
        representation='raw',
    )
    if animations is not None:
        batch._set_render(mask=mask, referee_animations=animations)
    digest = hashlib.sha256()
    batch.reset()
    for step in range(steps):
        actions = actions_at(step, 2, batch.agents, batch.action_count)
        observation, reward, done, _ = batch.step(actions.to(batch.torch_device))
        for name in sorted(observation):
            digest.update(tensor_bytes(observation[name]))
        digest.update(tensor_bytes(reward))
        digest.update(tensor_bytes(done))
    states = [batch.get_state(index) for index in range(2)]
    batch.close()
    return digest.hexdigest(), states


def test_the_flag_off_plays_as_before():
    assert play('cpu', False) == play('cpu', None)


def test_animations_play_the_stoppages():
    off, _ = play('cpu', False)
    on, _ = play('cpu', True)
    assert on != off


def test_the_mask_picks_environments():
    _, off = play('cpu', False)
    _, on = play('cpu', True)
    _, masked = play('cpu', True, mask=[True, False])
    assert masked[0] == on[0] and masked[1] == off[1]


@pytest.mark.skipif(not CUDA, reason='no CUDA device or library')
def test_both_backends_play_animations_alike():
    assert play('cuda', True) == play('cpu', True)
