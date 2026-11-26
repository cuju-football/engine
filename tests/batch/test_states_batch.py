# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks `get_states` and `set_states` against the loop of `get_state` and `set_state`."""

import pytest

import cuju
from conftest import actions_at, tensor_bytes
from cuju.errors import ArgumentError

COUNT = 3


def make(device, **options):
    values = dict(
        environment_count=COUNT,
        device=device,
        left_agents=1,
        right_agents=1,
        representation='simple115v2',
        raw=True,
        seed=3,
    )
    values.update(options)
    return cuju.BatchEnvironment('5_vs_5', **values)


def advance(batch, steps, start=0):
    for step in range(start, start + steps):
        batch.step(actions_at(step, COUNT, 2, 19).to(batch.device))


@pytest.mark.parametrize('options', [{}, {'rewards': 'scoring,checkpoints'}, {'stacked': True}])
def test_every_entry_equals_get_state(device, options):
    with make(device, **options) as batch:
        batch.reset()
        advance(batch, 6)
        stacked = bool(options.get('stacked'))
        states = batch.get_states(stacked)
        assert len(states) == COUNT
        assert states == [batch.get_state(index, stacked) for index in range(COUNT)]


@pytest.mark.parametrize('options', [{}, {'rewards': 'scoring,checkpoints'}, {'stacked': True}])
def test_set_states_plays_on_as_a_loop_of_set_state(device, options):
    stacked = bool(options.get('stacked'))
    with (
        make(device, **options) as source,
        make(device, **options) as together,
        make(device, **options) as looped,
    ):
        for batch in (source, together, looped):
            batch.reset()
        advance(source, 7)
        states = source.get_states(stacked)
        together.set_states(states, stacked)
        for index, state in enumerate(states):
            looped.set_state(index, state, stacked)
        assert together.get_states(stacked) == looped.get_states(stacked)
        for step in range(7, 11):
            actions = actions_at(step, COUNT, 2, 19).to(together.device)
            first = together.step(actions)
            second = looped.step(actions)
            assert tensor_bytes(first[0]) == tensor_bytes(second[0])
            assert tensor_bytes(first[1]) == tensor_bytes(second[1])
            assert tensor_bytes(together.raw()['ball']) == tensor_bytes(looped.raw()['ball'])


def test_set_states_refuses_a_wrong_count_and_a_foreign_state(device):
    with make(device) as batch:
        batch.reset()
        with pytest.raises(ArgumentError, match='3 states'):
            batch.set_states(batch.get_states()[:2])
        with make(device, environment_count=1, left_agents=2) as other:
            other.reset()
            foreign = other.get_states() * COUNT
        with pytest.raises(ValueError, match='another'):
            batch.set_states(foreign)
