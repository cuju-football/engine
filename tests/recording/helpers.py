# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers of the tests of cuju.recording."""

import torch

import cuju
import cuju.dumps
import cuju.recording as cr
from cuju import _kaggle
from cuju._batch import unwrap

SHORT = 'academy_empty_goal_close'
TEAMS = 'academy_3_vs_1_with_keeper'


def make_batch(device='cpu', scenario=SHORT, environments=3, **options):
    options.setdefault('raw', True)
    return cuju.BatchEnvironment(
        scenario, environment_count=environments, device=device, seed=3, **options
    )


def actions_for(generator, batch):
    return torch.randint(0, 19, (batch.environment_count, batch.agents), generator=generator).int()


def record_episodes(batch, environments, count, **options):
    """Plays `count` steps with seeded random actions, returns the recordings and the recorder."""
    found = []
    generator = torch.Generator().manual_seed(11)
    with cr.record(batch, environments, found.append, **options) as recorder:
        batch.reset()
        for _ in range(count):
            batch.step(actions_for(generator, batch))
    return found, recorder


def same_steps(first, second):
    assert first.keys() == second.keys()
    for name in first:
        assert first[name].dtype == second[name].dtype, name
        assert first[name].tobytes() == second[name].tobytes(), name


def play(tmp_path, scenario=SHORT, left=1, right=0, kaggle=False):
    """Plays one episode, and returns its recording, its dump and optionally a Kaggle episode."""
    batch = make_batch(scenario=scenario, environments=1, left_agents=left, right_agents=right)
    writer = cuju.dumps.DumpWriter(batch, str(tmp_path))
    found = []
    generator = torch.Generator().manual_seed(3)
    steps = []
    with cr.record(batch, [0], found.append):
        batch.reset()
        steps.append(_entries(batch, None))
        while not any('episode_done' in path for path in writer.paths):
            actions = actions_for(generator, batch)
            batch.step(actions)
            writer.record(actions)
            steps.append(_entries(batch, actions[0].tolist()))
    # The step that ends the episode shows the next episode, so the last entry is the final state.
    batch.close()
    path = next(path for path in writer.paths if 'episode_done' in path)
    episode = {
        'configuration': {'scenario_name': scenario},
        'info': {'TeamNames': ['red', 'blue']},
        'steps': steps[:-1],
    }
    return found[0], path, episode


def _entries(batch, actions):
    inner = unwrap(batch)
    left, right = int(inner.left_agents[0]), int(inner.right_agents[0])
    entries = []
    for agents in (range(left), range(left, left + right)):
        own = None if actions is None else [actions[agent] for agent in agents]
        entries.append(
            {
                'action': own,
                'status': 'ACTIVE',
                'observation': {
                    'players_raw': [_kaggle.players_raw(inner, 0, agent)[0] for agent in agents]
                },
            }
        )
    return entries
