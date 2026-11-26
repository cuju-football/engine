# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks the joints of a scene against joint math that is independent of the engine's.

The reference is the math of a reader of states: the world transform of each node from the local
ones of a state, in float64, as a recording stores them after quantizing to steps of 2 mm and of
1/32767. The scene lies in the frame that the library loads a state into, where both teams stand
on their own sides, so the reference reads that frame. The saved state keeps the right team
mirrored at its root, and the last test states the transform.
"""

import numpy as np
import pytest
import torch
from scene_support import (
    entry_nodes,
    make_batch,
    quantize,
    random_actions,
    reference,
    rotation_gap,
)

import cuju.scene as scene

SCENARIOS = [
    '5_vs_5',
    '11_vs_11_stochastic',
    'academy_3_vs_1_with_keeper',
    'academy_corner',
    'academy_empty_goal_close',
]


def sample(device, name, steps=(0, 5, 23, 60)):
    """Plays a batch, and returns the scene and the state of each environment at some steps."""
    batch = make_batch(device, name, environments=2)
    batch.reset()
    generator = torch.Generator().manual_seed(3)
    found = []
    for step in range(max(steps) + 1):
        if step in steps:
            frames = scene.of_batch(batch)
            for environment in range(2):
                found.append((frames[environment], batch.get_state(environment)))
        batch.step(random_actions(generator, batch))
    batch.close()
    return found


@pytest.mark.parametrize('name', SCENARIOS)
def test_the_joints_equal_the_reference_math_for_both_teams(device, name):
    for frames, state in sample(device, name):
        expected = reference(state, frames.team_players)
        present = ~np.isnan(expected[:, 0, 0])
        assert (frames.team[present] >= 0).all()
        assert (frames.team[~present & (frames.team != 2)] == -1).all()
        nodes = entry_nodes(frames)
        assert np.abs(nodes[present][..., :3] - expected[present][..., :3]).max() < 2e-5, (
            'positions differ by more than float32 rounding'
        )
        assert rotation_gap(nodes[present][..., 3:], expected[present][..., 3:]).max() < 1e-6
        # What a recording stores of both is within one step of quantization.
        steps = np.abs(quantize(nodes[present]) - quantize(expected[present]))
        assert steps[..., :3].max() <= 1 and steps[..., 3:].max() <= 1


@pytest.mark.parametrize('name', ['5_vs_5', '11_vs_11_stochastic'])
def test_the_saved_state_mirrors_the_right_team_and_the_scene_does_not(name):
    for frames, state in sample('cpu', name, steps=(0, 40)):
        rest = reference(state, frames.team_players, frame='rest')
        nodes = entry_nodes(frames)
        left = frames.team == 0
        right = frames.team == 1
        assert np.abs(nodes[left] - rest[left]).max() < 2e-5
        # The root of the right team, as GRF mirrors it: its position at (-x, -y, z) and its
        # rotation (x, y, z, w) at (y, x, w, -z). Its joints follow from that root.
        roots = rest[right][:, 0]
        position = roots[:, :3] * np.array([-1, -1, 1])
        rotation = np.stack([roots[:, 4], roots[:, 3], roots[:, 6], -roots[:, 5]], -1)
        assert np.abs(nodes[right][:, 0, :3] - position).max() < 2e-5
        assert np.abs(nodes[right][:, 0, 3:] - rotation).max() < 1e-6


def test_the_left_team_defends_the_left_goal_and_the_right_team_the_right_goal():
    batch = make_batch('cpu', '11_vs_11_stochastic', environments=1)
    batch.reset()
    frames = scene.of_batch(batch)
    batch.close()
    goalkeepers = frames.role == 0
    half = scene.PITCH.half_length
    assert frames.root_position[goalkeepers][:, 0] == pytest.approx([-half, half], abs=1e-3)
    assert frames.team[goalkeepers].tolist() == [0, 1]
