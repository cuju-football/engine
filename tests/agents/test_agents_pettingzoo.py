# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks the PettingZoo adapter."""

import pytest

import cuju
from cuju import agents

pytest.importorskip('pettingzoo')
pytest.importorskip('gymnasium')


def make():
    return cuju.BatchEnvironment(
        '5_vs_5',
        environment_count=1,
        device='cpu',
        left_agents=2,
        right_agents=0,
        representation='simple115v2',
        raw=True,
        auto_reset=False,
    )


def test_the_environment_names_its_agents_and_plays_an_episode():
    batch = make()
    with agents.Match(batch, right='builtin') as match:
        env = agents.to_pettingzoo(match)
        assert env.possible_agents == ['left_0', 'left_1']
        observations, infos = env.reset(seed=1)
        assert set(observations) == set(infos) == {'left_0', 'left_1'}
        assert env.observation_space('left_0').shape == (115,)
        for _ in range(3001):
            _, _, _, truncated, _ = env.step(
                {agent: env.action_space(agent).sample() for agent in env.agents}
            )
        assert env.agents == [] and all(truncated.values())
    batch.close()


def test_the_api_test_of_pettingzoo_passes():
    from pettingzoo.test import parallel_api_test

    batch = make()
    with agents.Match(batch, right='builtin') as match:
        parallel_api_test(agents.to_pettingzoo(match), num_cycles=50)
    batch.close()
