# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks that each literal alias of cuju.types names the values that the runtime accepts."""

import typing

import pytest

import cuju
from cuju import types
from cuju._batch import environment as batch_environment
from cuju.agents import _contract
from cuju.errors import ConfigurationError


def values(alias):
    return set(typing.get_args(alias))


def test_the_module_is_reached_through_cuju():
    assert cuju.types is types


def test_representation_matches_the_batch():
    accepted = set(batch_environment._REPRESENTATIONS) | set(
        batch_environment._PIXEL_REPRESENTATIONS
    )
    assert values(types.Representation) == accepted
    assert values(types.ControllerRepresentation) == {
        'raw',
        'simple115',
        'simple115v2',
        'extracted',
    }


def test_action_set_matches_the_batch_and_the_controller_contract():
    assert values(types.ActionSet) == set(batch_environment._ACTION_SETS)
    assert values(types.ActionSet) == set(_contract.ACTION_COUNTS)
    for name in values(types.ActionSet):
        assert _contract.normalize({'action_set': name})['action_set'] == name
    with pytest.raises(ConfigurationError):
        _contract.normalize({'action_set': 'other'})


@pytest.mark.parametrize('rewards', sorted(values(types.Rewards)))
def test_every_reward_literal_builds_a_batch(rewards):
    with cuju.BatchEnvironment('5_vs_5', environment_count=1, device='cpu', rewards=rewards):
        pass


def test_control_literals_match_the_contract():
    for name in values(types.Assignment):
        table = {'assignment': name, 'selection': [1] if name == 'fixed' else []}
        assert _contract.normalize(table)['assignment'] == name
    for name in values(types.Goalkeeper):
        table = {'goalkeeper': name, 'assignment': 'fixed', 'selection': [1]}
        assert _contract.normalize(table)['goalkeeper'] == name
    assert values(types.Side) == set(_contract.SIDES)
    assert values(types.Sides) == values(types.Side) | {'both'}


def test_device_takes_names_strings_and_torch_devices():
    arguments = typing.get_args(types.Device)
    assert typing.get_args(arguments[0]) == ('auto', 'cpu', 'cuda')
    assert arguments[1] is str


def test_reasons_are_the_ones_the_controller_error_gives():
    assert {'timeout', 'closed', 'protocol', 'shape', 'type', 'range', 'load', 'crash'} <= values(
        types.Reason
    )
    assert values(types.RecordingEnd) == {0, 1, 2}
