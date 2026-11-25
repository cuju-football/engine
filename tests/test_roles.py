# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks the roles that a batch reports, across a goalkeeper's sending-off and a load."""

import math

import pytest
import torch

import cuju
from conftest import tensor_bytes
from cuju._batch import Batch

SCENARIO = '11_vs_11_stochastic_l11_r0'
OPTIONS = {'seed': 1, 'first_index': 28, 'representation': 'raw', 'auto_reset': False}
SENT_OFF_STEP = 243
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
    batch = Batch(SCENARIO, environment_count=1, device=device, **OPTIONS)
    plain = Batch(SCENARIO, environment_count=1, device=device, **OPTIONS)
    raw = batch.reset()
    assert tensor_bytes(plain.reset()['left_team']) == tensor_bytes(raw['left_team'])
    before = reported(batch)
    assert before == raw_roles(raw)
    assert before[0][:2] == [0, 7] and before[0].count(0) == 1
    with pytest.raises(cuju.errors.ArgumentError):
        batch.roles(1)
    for step in range(SENT_OFF_STEP + 1):
        actions = chase(raw, step)
        raw, _, _, _ = batch.step(actions)
        unread, _, _, _ = plain.step(actions)
        assert all(tensor_bytes(raw[name]) == tensor_bytes(unread[name]) for name in raw)
        assert reported(batch) == raw_roles(raw)
        if not bool(raw['left_team_active'][0, 0, 0]):
            break
    assert step == SENT_OFF_STEP
    after = reported(batch)
    assert after[0] == [0, 0, *before[0][2:]] and after[1] == before[1]
    state = batch.get_state(0)
    assert state == plain.get_state(0)
    batch.set_state(0, state)
    # GRF: the observation right after the load already shows the hand-over.
    loaded, unread = batch.raw(), plain.raw()
    assert reported(batch) == after == raw_roles(loaded)
    assert all(tensor_bytes(loaded[name]) == tensor_bytes(unread[name]) for name in loaded)
    assert batch.get_state(0) == state
    raw, _, _, _ = batch.step(torch.zeros((1, 11), dtype=torch.int32))
    assert reported(batch) == after == raw_roles(raw)
    fresh = Batch(SCENARIO, environment_count=1, device=device, **OPTIONS)
    fresh.reset()
    fresh.set_state(0, state)
    assert reported(fresh) == before == raw_roles(fresh.raw())
    for closing in (batch, plain, fresh):
        closing.close()
