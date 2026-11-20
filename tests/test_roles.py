# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks the roles that a batch reports, across a goalkeeper's sending-off and a load."""

import math

import pytest
import torch

import cuju
from conftest import tensor_bytes

# Environment 28 of seed 1, whose left goalkeeper, an agent that chases the ball and slides at the
# opponent who holds it, is sent off on step 243 and hands the role to player 1.
SCENARIO = '11_vs_11_stochastic_l11_r0'
OPTIONS = {'seed': 1, 'first_index': 28, 'representation': 'raw', 'auto_reset': False}
SENT_OFF_STEP = 243
# The directions of the action set by the angle from the goalkeeper to the ball, in eighths of a
# turn from the right, where y grows downwards.
DIRECTIONS = [5, 6, 7, 8, 1, 2, 3, 4]


def chase(raw, step):
    """The actions of the eleven agents, read from the raw observation on the CPU alone."""
    keeper = raw['left_team'][0, 0, 0].cpu()
    ball = raw['ball'][0, 0, :2].cpu()
    offset = ball - keeper
    eighth = round(math.atan2(float(offset[1]), float(offset[0])) / (math.pi / 4)) % 8
    action = DIRECTIONS[eighth]
    if float(offset.norm()) < 0.03 and int(raw['ball_owned_team'][0, 0]) == 1 and step % 2 == 0:
        action = 16
    if step % 10 == 0:
        action = 13
    actions = torch.zeros((1, 11), dtype=torch.int32)
    actions[0, 0] = action
    return actions


def raw_roles(raw):
    """The roles of a left agent's raw observation, as `roles` returns them."""
    return [raw[f'{side}_team_roles'][0, 0].tolist() for side in ('left', 'right')]


def reported(batch):
    roles = batch.roles(0)
    assert all(tensor.dtype == torch.int32 for tensor in roles.values())
    return [roles['left_team_roles'].tolist(), roles['right_team_roles'].tolist()]


def test_the_roles_follow_a_goalkeeper_sent_off_and_leave_every_output_as_it_is(device):
    batch = cuju.BatchEnvironment(SCENARIO, environment_count=1, device=device, **OPTIONS)
    plain = cuju.BatchEnvironment(SCENARIO, environment_count=1, device=device, **OPTIONS)
    raw = batch.reset()
    assert tensor_bytes(plain.reset()['left_team']) == tensor_bytes(raw['left_team'])
    before = reported(batch)
    assert before == raw_roles(raw)
    assert before[0][:2] == [0, 7] and before[0].count(0) == 1
    with pytest.raises(cuju.EnvironmentIndexError):
        batch.roles(1)
    for step in range(SENT_OFF_STEP + 1):
        actions = chase(raw, step)
        raw, _, _, _ = batch.step(actions)
        unread, _, _, _ = plain.step(actions)
        # Reading the roles changes no output.
        assert all(tensor_bytes(raw[name]) == tensor_bytes(unread[name]) for name in raw)
        assert reported(batch) == raw_roles(raw)
        if not bool(raw['left_team_active'][0, 0, 0]):
            break
    assert step == SENT_OFF_STEP
    after = reported(batch)
    # The first active player of the team takes the goalkeeper's formation entry and role.
    assert after[0] == [0, 0, *before[0][2:]] and after[1] == before[1]
    state = batch.get_state(0)
    assert state == plain.get_state(0)
    # The hand-over lies outside the state, as in GRF, so the environment that loads the state
    # keeps its own: this one its hand-over, and a fresh one none.
    batch.set_state(0, state)
    # GRF: the observation right after the load already shows the hand-over, so every raw output
    # equals that of the batch that loaded nothing, and the state keeps its bytes.
    loaded, unread = batch.raw(), plain.raw()
    assert reported(batch) == after == raw_roles(loaded)
    assert all(tensor_bytes(loaded[name]) == tensor_bytes(unread[name]) for name in loaded)
    assert batch.get_state(0) == state
    raw, _, _, _ = batch.step(torch.zeros((1, 11), dtype=torch.int32))
    assert reported(batch) == after == raw_roles(raw)
    fresh = cuju.BatchEnvironment(SCENARIO, environment_count=1, device=device, **OPTIONS)
    fresh.reset()
    fresh.set_state(0, state)
    assert reported(fresh) == before == raw_roles(fresh.raw())
    for closing in (batch, plain, fresh):
        closing.close()


def test_a_multi_device_batch_reports_the_roles_of_its_parts():
    batch = cuju.MultiDeviceBatch('5_vs_5_l4_r0', ['cpu', 'cpu'], environment_count=3, seed=2)
    batch.reset()
    for index, (part, local) in enumerate([(0, 0), (0, 1), (1, 0)]):
        roles = batch.roles(index)
        expected = batch.parts[part].roles(local)
        assert roles.keys() == expected.keys() == {'left_team_roles', 'right_team_roles'}
        assert all(roles[key].tolist() == expected[key].tolist() for key in roles)
        assert len(roles['left_team_roles']) == len(roles['right_team_roles']) == 5
    batch.close()
