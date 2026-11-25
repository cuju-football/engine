# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks that a state moved to another environment continues the same match, and that a refused
state names the field in which it differs from the environment."""

import pytest

from conftest import actions_at, tensor_bytes
from cuju._batch import Batch


def test_state_round_trip(device):
    batch = Batch('5_vs_5_l4_r0', environment_count=2, device=device, rewards='scoring,checkpoints')
    batch.reset()
    for step in range(60):
        batch.step(actions_at(step, 2, 4, 19).to(device))
    batch.set_state(1, batch.get_state(0))
    ended = False
    for step in range(60, 160):
        actions = actions_at(step, 1, 4, 19).repeat(2, 1).to(device)
        observation, reward, done, information = batch.step(actions)
        if bool(done[0]):
            ended = True
            observation = information['final_observation']
        assert tensor_bytes(observation[0]) == tensor_bytes(observation[1])
        assert tensor_bytes(reward[0]) == tensor_bytes(reward[1])
        if ended:
            break
    if not ended:
        assert batch.get_state(0) == batch.get_state(1)


def test_a_state_of_the_other_parity_plays_as_in_grf(device):
    # GRF: `11_vs_11_stochastic` swaps the teams' formations with the parity of the episode.
    import hashlib

    import numpy as np
    import torch

    import cuju._state

    def batch(episode_number):
        made = Batch(
            '11_vs_11_stochastic',
            environment_count=1,
            device=device,
            agent_counts=[(0, 0)],
            episode_number=episode_number,
            game_engine_random_seed=2,
        )
        made.reset()
        return made

    actions = torch.zeros((1, 0), dtype=torch.int32, device=device)
    first = batch(1)
    for _ in range(100):
        first.step(actions)
    state = first.get_state(0)
    first.close()
    second = batch(2)
    second.set_state(0, state)
    for _ in range(115):
        second.step(actions)
    loaded = second.get_state(0)
    second.close()
    layout = cuju._state.layout()
    names = list(layout)
    digest = hashlib.sha256()
    for name, value in zip(names, cuju._state.read(loaded, names), strict=True):
        data = np.ascontiguousarray(value).tobytes()
        digest.update(data[:5] if layout[name].type == 'N7blunted6radianE' else data)
    assert digest.hexdigest() == '61019403dfa67543dcad95dc61e2ebd7f6cdcc18c50183a4e0cf96e27c2dd15a'


def state_of(assets, device):
    """Returns the state of one environment of `assets` after a few steps."""
    batch = Batch(assets, environment_count=1, device=device)
    batch.reset()
    for step in range(5):
        batch.step(actions_at(step, 1, batch.agents, batch.action_count).to(device))
    state = batch.get_state(0)
    batch.close()
    return state


def started(assets, device):
    batch = Batch(assets, environment_count=1, device=device)
    batch.reset()
    return batch


def test_another_agent_count(device):
    batch = started('5_vs_5_l4_r1', device)
    with pytest.raises(
        RuntimeError, match='right_agents is 0 in the state and 1 in the environment'
    ):
        batch.set_state(0, state_of('5_vs_5_l4_r0', device))


def test_another_scenario(device):
    batch = started('academy_empty_goal_close_l1_r0', device)
    with pytest.raises(RuntimeError, match=r'found another: ball_position is \(') as raised:
        batch.set_state(0, state_of('academy_empty_goal_l1_r0', device))
    assert 'a state loads only into an environment of the same scenario' in str(raised.value)


def test_bytes_that_stop_short_or_run_on(device):
    state = state_of('5_vs_5_l4_r0', device)
    batch = started('5_vs_5_l4_r0', device)
    with pytest.raises(RuntimeError, match='do not load: state too short at'):
        batch.set_state(0, state[:-100])
    with pytest.raises(RuntimeError, match='do not load: 8 bytes follow the end of the state'):
        batch.set_state(0, state + bytes(8))
    batch.set_state(0, state)
    assert batch.get_state(0) == state
