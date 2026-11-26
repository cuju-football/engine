# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks how `cuju.agents.load` reads trees of controllers."""

import ntpath

import numpy as np
import pytest
import torch

from cuju import agents
from cuju.agents import standard
from cuju.errors import ConfigurationError

AGENT = 'def agent(obs):\n    return [1]\n'


def observation(environments=2, agents_per=1):
    shape = (environments, agents_per)
    return agents.Observation(
        data={'ball': torch.zeros(environments, agents_per, 3)},
        first=torch.ones(environments, dtype=torch.bool),
        active=torch.ones(shape, dtype=torch.bool),
        left_players=torch.full((environments,), 5),
        right_players=torch.full((environments,), 5),
        previous=torch.zeros(shape, dtype=torch.int32),
        step=torch.zeros(environments, dtype=torch.int64),
        side=torch.zeros(environments, dtype=torch.int8),
    )


def test_a_folder_loads_as_kaggle(tmp_path):
    (tmp_path / 'main.py').write_text(AGENT)
    assert isinstance(agents.load(tmp_path), standard.Kaggle)
    assert isinstance(agents.load(str(tmp_path / 'main.py')), standard.Kaggle)


def test_a_controller_loads_as_it_is_and_a_function_wraps():
    controller = standard.Random()
    assert agents.load(controller) is controller
    assert isinstance(agents.load(lambda observation: 0), standard.Function)


def test_entry_strings_and_names():
    assert isinstance(agents.load('random'), standard.Random)
    assert isinstance(agents.load('cuju.agents.standard:Random'), standard.Random)
    built = agents.load('builtin@0.7')
    assert isinstance(built, standard.Builtin) and built.difficulty == 0.7
    with pytest.raises(ConfigurationError, match='found'):
        agents.load('no_such_name')


def test_a_windows_path_is_not_an_entry(monkeypatch):
    monkeypatch.setattr('os.path.isabs', ntpath.isabs)
    with pytest.raises(ConfigurationError, match='expected an existing path'):
        agents.load(r'C:\no\such\agent')


def test_a_toml_file_nests_and_resolves_paths(tmp_path):
    (tmp_path / 'bot').mkdir()
    (tmp_path / 'bot' / 'main.py').write_text(AGENT)
    (tmp_path / 'pool.toml').write_text(
        'agents = 1\nentry = "pool"\n[[inner]]\nentry = "kaggle"\npath = "bot"\n'
        '[[inner]]\nentry = "random"\n'
    )
    pool = agents.load(tmp_path / 'pool.toml')
    assert isinstance(pool, standard.Pool)
    assert pool._members[0].folder == str(tmp_path / 'bot')


def test_an_unknown_key_names_its_node_path(tmp_path):
    tree = {'entry': 'pool', 'inner': [{'entry': 'random', 'step_seconds': 1}]}
    with pytest.raises(ConfigurationError, match=r'inner\[0\]\.step_seconds'):
        agents.load(tree)
    with pytest.raises(ConfigurationError, match='step_seconds'):
        agents.load({'entry': 'random', 'step_seconds': 1})


def test_a_schema_other_than_one_is_refused(tmp_path):
    (tmp_path / 'a.toml').write_text('agents = 2\nentry = "random"\n')
    with pytest.raises(ConfigurationError, match='agents = 1'):
        agents.load(tmp_path / 'a.toml')


def test_the_standard_controllers_answer():
    spec = agents.Spec(environments=2, agents=1, seed=3)
    chosen = observation()
    random = standard.Random()
    random.start(spec)
    assert random.act(chosen).actions.shape == (2, 1)
    lazy = standard.Lazy()
    assert not np.asarray(lazy.act(chosen).actions).any()
    seen = []
    each = standard.PerEnvironment(lambda one: seen.append(one.first) or [4])
    each.start(spec)
    assert np.asarray(each.act(chosen).actions).tolist() == [[4], [4]]
    assert seen == [True, True]
    function = standard.Function(
        lambda o: (torch.full(o.active.shape, 2), {'value': torch.zeros(2, 1)})
    )
    answer = function.act(chosen)
    assert answer.actions.tolist() == [[2], [2]] and 'value' in answer.extras


def test_a_pool_draws_once_and_swaps():
    pool = standard.Pool(
        [
            standard.Function(lambda o: torch.zeros(o.active.shape, dtype=torch.int64)),
            standard.Function(lambda o: torch.ones(o.active.shape, dtype=torch.int64)),
        ],
        weights=[1, 0],
    )
    pool.start(agents.Spec(environments=3, agents=1, seed=1))
    chosen = observation(3)
    assert pool.act(chosen).actions.sum() == 0
    pool.set_pool([standard.Function(lambda o: torch.full(o.active.shape, 7, dtype=torch.int64))])
    assert pool.act(chosen).actions.sum() == 21
    chosen.first[:] = False
    assert pool.act(chosen).actions.sum() == 21
