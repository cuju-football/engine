# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks that GRF's Python over cuju's engine plays what `cuju.env` plays, bit for bit."""

import random

import numpy as np
import pytest

import cuju.env as cuju_env

grf_env = pytest.importorskip('cuju.gfootball.env')

CASES = [
    ('academy_empty_goal_close', {'representation': 'simple115v2'}),
    ('5_vs_5', {'representation': 'raw', 'number_of_left_players_agent_controls': 2}),
    (
        '11_vs_11_easy_stochastic',
        {'representation': 'extracted', 'stacked': True, 'rewards': 'scoring,checkpoints'},
    ),
    (
        'academy_corner',
        {'representation': 'simple115', 'number_of_right_players_agent_controls': 1},
    ),
    (
        '1_vs_1_easy',
        {'representation': 'simple115v2', 'other_config_options': {'action_set': 'full'}},
    ),
]


def encode(value):
    """Returns the bytes of an observation, a reward or an information, with types and shapes."""
    if isinstance(value, dict):
        return b''.join(key.encode() + encode(value[key]) for key in sorted(value))
    if isinstance(value, (list, tuple)):
        return b'[' + b','.join(encode(item) for item in value) + b']'
    array = np.asarray(value)
    return f'{array.dtype}{array.shape}'.encode() + array.tobytes()


def play(make, name, options, steps, seed):
    """Plays random actions and returns every output and the engine part of the final state."""
    random.seed(seed)
    env = make(name, **options)
    choices = random.Random(seed + 1)
    outputs = [encode(env.reset())]
    agents = None
    for _ in range(steps):
        count = env.action_space.nvec.size if hasattr(env.action_space, 'nvec') else 0
        if agents is None:
            agents = count
        if agents:
            actions = [choices.randrange(int(env.action_space.nvec[0])) for _ in range(agents)]
        else:
            actions = choices.randrange(int(env.action_space.n))
        observation, reward, done, info = env.step(actions)
        outputs.append(encode((observation, reward, done, info)))
        if done:
            outputs.append(encode(env.reset()))
    state = env.get_state()
    size = int.from_bytes(state[:4], 'little', signed=True)
    env.close()
    return outputs, state[4 + size :]


@pytest.mark.parametrize(('name', 'options'), CASES, ids=[case[0] for case in CASES])
def test_the_port_plays_as_cuju_env(name, options):
    port, port_state = play(grf_env.create_environment, name, options, 120, seed=11)
    fast, fast_state = play(cuju_env.create_environment, name, options, 120, seed=11)
    assert len(port) == len(fast)
    for step, (first, second) in enumerate(zip(port, fast, strict=True)):
        assert first == second, f'the outputs differ at step {step}'
    assert port_state == fast_state
