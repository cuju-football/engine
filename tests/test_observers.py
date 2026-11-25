# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks the step observers of a batch, its scenario name, and what a plain batch leaves alone."""

import gc
import hashlib
import logging
import os
import statistics
import subprocess
import sys
import textwrap
import time
import weakref

import pytest
import torch

import cuju
from conftest import actions_at, tensor_bytes
from cuju._batch import Batch

ASSETS = '5_vs_5_l4_r0'


def observed_digest(device, observers, steps=60, environments=4):
    """`play_digest` with `observers` added, and a masked reset and a split step on the way."""
    batch = Batch(ASSETS, environment_count=environments, device=device, seed=7)
    for observer in observers:
        batch.add_observer(observer)
    digest = hashlib.sha256()
    digest.update(tensor_bytes(batch.reset()))
    for step in range(steps):
        actions = actions_at(step, environments, batch.agents, batch.action_count)
        if step % 3 == 0:
            batch.step_async(actions.to(batch.torch_device))
            observation, reward, done, _ = batch.step_wait()
        else:
            observation, reward, done, _ = batch.step(actions.to(batch.torch_device))
        for tensor in (observation, reward, done):
            digest.update(tensor_bytes(tensor))
        if step == steps // 2:
            mask = torch.tensor([step % 2 == index % 2 for index in range(environments)])
            digest.update(tensor_bytes(batch.reset(mask=mask)))
    for index in range(environments):
        digest.update(batch.get_state(index))
    batch.close()
    return digest.hexdigest()


def test_observers_leave_every_output_as_it_is(device):
    seen = []

    def reader(batch, done, information):
        # An observer that reads every state and the scoring signal, as a renderer does.
        if done is not None:
            seen.append(int(information['score_reward'].sum()))
            information['reward'].sum().item()
            information['actions'].sum().item()
            for index in range(batch.environment_count):
                batch.get_state(index)

    plain = observed_digest(device, [])
    assert observed_digest(device, [reader, lambda batch, done, information: None]) == plain
    assert len(seen) == 60


def test_observers_run_in_order_with_what_each_call_returns():
    batch = Batch(ASSETS, environment_count=3, device='cpu', seed=1)
    calls = []

    def first(batch, done, information):
        calls.append(('first', done, information))

    def second(batch, done, information):
        calls.append(('second', done, information))

    assert batch.add_observer(first) is first
    batch.add_observer(second)
    assert batch.observers == (first, second)
    batch.reset()
    _, reward, done, information = batch.step(torch.zeros((3, 4), dtype=torch.int32))
    batch.reset(mask=[True, False, True])
    batch.step_async(torch.zeros((3, 4), dtype=torch.int32))
    _, waited_reward, waited, waited_information = batch.step_wait()
    assert [name for name, _, _ in calls] == ['first', 'second'] * 4
    assert calls[0][1] is None and calls[0][2] == {'reset': None}
    # A step hands its observers its information and its rewards, the step's own tensors, in a
    # dictionary of their own, and returns its information as it was.
    for index, done_flags, returned, rewards in (
        (2, done, information, reward),
        (6, waited, waited_information, waited_reward),
    ):
        assert calls[index][1] is done_flags
        assert calls[index][2] is not returned and 'reward' not in returned
        assert calls[index][2]['reward'] is rewards
        assert all(calls[index][2][key] is value for key, value in returned.items())
        assert calls[index][2].keys() == {*returned, 'reward', 'actions'}
        assert calls[index][2]['actions'].tolist() == [[0] * 4] * 3
    assert calls[4][1] is None and calls[4][2]['reset'].tolist() == [True, False, True]
    batch.remove_observer(first)
    batch.step(torch.zeros((3, 4), dtype=torch.int32))
    assert [name for name, _, _ in calls[8:]] == ['second']
    batch.close()
    assert calls[-1][1] is None and calls[-1][2] == {'close': True}
    assert batch.observers == ()


def actions_of(observed):
    """An observer that keeps a copy of the actions of each step and the key of each reset."""

    def observer(batch, done, information):
        if done is None:
            observed.append(('actions' in information, None))
            return
        actions = information['actions']
        assert actions.dtype == torch.int32 and actions.device == batch.torch_device
        assert tuple(actions.shape) == (batch.environment_count, batch.agents)
        observed.append((done.bool().any().item(), actions.clone()))

    return observer


def test_observers_receive_the_actions_of_every_step(device):
    batch = Batch(ASSETS, environment_count=3, device=device, seed=3)
    observed = []
    batch.add_observer(actions_of(observed))
    batch.reset()
    on_device = actions_at(1, 3, 4, batch.action_count).to(batch.torch_device)
    _, _, _, _ = batch.step(on_device)
    # Actions already in the batch's form reach the observers without a copy.
    seen = []
    batch.add_observer(lambda batch, done, information: seen.append(information.get('actions')))
    batch.step(on_device)
    assert seen[-1].data_ptr() == on_device.data_ptr()
    played = [on_device.cpu(), on_device.cpu()]
    forms = [
        actions_at(2, 3, 4, batch.action_count).tolist(),
        actions_at(3, 3, 4, batch.action_count).to(torch.int64),
        actions_at(4, 3, 4, batch.action_count),
        # A negative index counts from the end, and the observers see it as given.
        -1 - actions_at(5, 3, 4, batch.action_count),
    ]
    if device == 'cuda':
        forms.append(actions_at(6, 3, 4, batch.action_count).pin_memory())
    for actions in forms:
        batch.step(actions)
        played.append(torch.as_tensor(actions, dtype=torch.int32))
    batch.reset(mask=[True, False, True])
    # After `step_async` the observers see the actions that the step started with, even when the
    # caller writes its tensor before `step_wait`.
    for host in (False, True):
        actions = actions_at(7 + host, 3, 4, batch.action_count)
        actions = actions if host else actions.to(batch.torch_device)
        batch.step_async(actions)
        played.append(actions.cpu().clone())
        actions.fill_(0)
        batch.step_wait()
    batch.close()
    steps = [entry for entry in observed if entry[1] is not None]
    # The two resets and the close carry no actions.
    assert [entry for entry in observed if entry[1] is None] == [(False, None)] * 3
    assert len(steps) == len(played)
    for (_, seen_actions), actions in zip(steps, played, strict=True):
        assert seen_actions.cpu().tolist() == actions.tolist()


def test_observers_receive_the_actions_of_a_step_that_resets(device):
    batch = Batch('academy_empty_goal_close_l1_r0', environment_count=2, device=device, seed=5)
    observed = []
    batch.add_observer(actions_of(observed))
    batch.reset()
    resets = 0
    for step in range(400):
        # The first environment shoots, the second moves, so only one ends on most steps.
        actions = torch.tensor([[12 if step % 2 else 5], [step % 9]], dtype=torch.int32)
        _, _, done, _ = batch.step(actions.squeeze(1) if step % 3 else actions)
        ended, seen = observed[-1]
        assert ended == bool(done.bool().any())
        assert seen.cpu().tolist() == actions.tolist()
        resets += int(ended)
        if resets == 3:
            break
    assert resets == 3
    batch.close()


def test_reset_done_hands_the_ended_environments():
    batch = Batch(
        'academy_empty_goal_close_l1_r0', environment_count=2, device='cpu', auto_reset=False
    )
    resets = []
    batch.add_observer(lambda batch, done, information: resets.append(information.get('reset')))
    batch.reset()
    shoot = torch.tensor([[12], [0]], dtype=torch.int32)
    done = torch.zeros(2)
    for _ in range(400):
        _, _, done, _ = batch.step(shoot)
        if bool(done.any()):
            break
    ended = done.bool().tolist()
    assert any(ended)
    batch.reset_done()
    assert resets[-1].tolist() == ended
    batch.close()


def test_a_raising_observer_is_removed_with_one_warning(caplog):
    batch = Batch(ASSETS, environment_count=2, device='cpu')
    later = []

    def failing(batch, done, information):
        raise OSError('disk full')

    batch.add_observer(failing)
    batch.add_observer(lambda batch, done, information: later.append(done is not None))
    with caplog.at_level(logging.WARNING, logger='cuju'):
        batch.reset()
        for _ in range(3):
            batch.step(torch.zeros((2, 4), dtype=torch.int32))
    warnings = [record for record in caplog.records if record.name == 'cuju']
    assert len(warnings) == 1
    assert 'OSError: disk full' in warnings[0].getMessage()
    assert len(batch.observers) == 1 and later == [False, True, True, True]
    batch.close()


def test_an_observer_may_remove_itself_and_others():
    batch = Batch(ASSETS, environment_count=1, device='cpu')
    calls = []

    def once(batch, done, information):
        calls.append('once')
        batch.remove_observer(once)
        batch.remove_observer(other)

    def other(batch, done, information):
        calls.append('other')

    batch.add_observer(once)
    batch.add_observer(other)
    batch.reset()
    batch.step(torch.zeros((1, 4), dtype=torch.int32))
    assert calls == ['once'] and batch.observers == ()
    batch.close()


def test_an_observer_may_remove_itself_and_close_again_during_the_close_call(caplog):
    batch = Batch(ASSETS, environment_count=1, device='cpu')
    calls = []

    def leaving(batch, done, information):
        calls.append(('leaving', information))
        if information.get('close'):
            batch.remove_observer(leaving)
            batch.close()
            batch.remove_observer(skipped)

    def skipped(batch, done, information):
        calls.append(('skipped', information))

    def last(batch, done, information):
        calls.append(('last', information))
        # The batch frees its buffers only once every observer has had its close call.
        batch.get_state(0)

    for observer in (leaving, skipped, last):
        batch.add_observer(observer)
    batch.reset()
    with caplog.at_level(logging.WARNING, logger='cuju'):
        batch.close()
    assert [record for record in caplog.records if record.name == 'cuju'] == []
    assert calls[3:] == [('leaving', {'close': True}), ('last', {'close': True})]
    assert batch.observers == ()
    # Removing an observer of a closed batch does nothing, and so does a second close.
    batch.remove_observer(last)
    batch.close()
    with pytest.raises(cuju.errors.ClosedError):
        batch.reset()


@pytest.mark.parametrize('call', ['step', 'step_wait', 'reset', 'reset_done'])
def test_an_observer_that_closes_the_batch_makes_its_call_raise(call):
    batch = Batch(ASSETS, environment_count=2, device='cpu')
    batch.reset()
    actions = torch.zeros((2, 4), dtype=torch.int32)
    calls = []

    def closing(batch, done, information):
        calls.append(('closing', information.get('close', False)))
        if not information.get('close'):
            batch.close()

    def later(batch, done, information):
        calls.append(('later', information.get('close', False)))

    batch.add_observer(closing)
    batch.add_observer(later)
    with pytest.raises(cuju.errors.ClosedError, match='an observer closed the batch'):
        if call == 'step':
            batch.step(actions)
        elif call == 'step_wait':
            batch.step_async(actions)
            batch.step_wait()
        elif call == 'reset':
            batch.reset()
        else:
            batch.reset_done()
    # Every observer has its close call, and the one after the closing observer misses the step.
    assert calls == [('closing', False), ('closing', True), ('later', True)]
    with pytest.raises(cuju.errors.ClosedError):
        batch.step(actions)


def test_a_dropped_batch_makes_no_close_call_and_a_weak_observer_lets_it_go():
    calls = []

    class Watcher:
        def __init__(self, batch, weak):
            self.batch = weakref.ref(batch) if weak else batch
            batch.add_observer(self)

        def __call__(self, batch, done, information):
            calls.append(information.get('close', False))

    for weak in (True, False):
        batch = Batch(ASSETS, environment_count=1, device='cpu')
        Watcher(batch, weak)
        batch.reset()
        gone = weakref.ref(batch)
        del batch
        # An observer that keeps the batch forms a cycle, which only the collector breaks.
        assert (gone() is None) is weak
        gc.collect()
        assert gone() is None
        assert calls == [False]
        calls.clear()


def test_observer_refusals():
    batch = Batch(ASSETS, environment_count=1, device='cpu')

    def observer(batch, done, information):
        pass

    with pytest.raises(cuju.errors.ArgumentError, match='callable'):
        batch.add_observer('not callable')
    batch.add_observer(observer)
    with pytest.raises(cuju.errors.ArgumentError, match='not added yet'):
        batch.add_observer(observer)
    batch.remove_observer(observer)
    with pytest.raises(cuju.errors.ArgumentError, match='observer of the batch'):
        batch.remove_observer(observer)
    batch.close()
    with pytest.raises(cuju.errors.ClosedError):
        batch.add_observer(observer)


def test_scenario_name(tmp_path):
    batch = Batch(
        [ASSETS, '5_vs_5_l1_r0', ASSETS], environment_count=1, device='cpu', assignment=[0]
    )
    assert batch.scenario_name == '5_vs_5_l4_r0, 5_vs_5_l1_r0'
    batch.close()
    scenario = tmp_path / 'my_scenario.py'
    scenario.write_text(
        textwrap.dedent(
            """
            from . import *


            def build_scenario(builder):
                builder.config().game_duration = 10
                builder.SetBallPosition(0.6, 0.0)
                builder.SetTeam(Team.e_Left)
                builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
                builder.AddPlayer(0.6, 0.0, e_PlayerRole_CF)
                builder.SetTeam(Team.e_Right)
                builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
            """
        )
    )
    batch = Batch(str(scenario), environment_count=1, device='cpu')
    assert batch.scenario_name == 'my_scenario'
    batch.close()


def run(code, **variables):
    """Runs `code` in a new interpreter that sees the same `cuju`, and returns what it prints."""
    environment = {**os.environ, **variables}
    environment.pop('CUJU_BATCH_HOOKS', None)
    environment.update(variables)
    result = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
        env=environment,
    )
    return result.stdout.split()


def test_a_step_without_observers_costs_no_more_than_one_with_a_no_op_observer():
    """A generous guard on the one check per step that a batch without observers pays.

    The steps of one small environment on the CPU leave Python its largest share. Rounds of steps
    without observers and with an observer that does nothing alternate, so drift touches both.
    """
    batch = Batch('5_vs_5_l1_r0', environment_count=1, device='cpu', threads=1)
    actions = torch.zeros((1, 1), dtype=torch.int32)
    batch.reset()
    for _ in range(100):
        batch.step(actions)

    def no_op(batch, done, information):
        pass

    plain, observed = [], []
    for _ in range(9):
        for samples, observing in ((plain, False), (observed, True)):
            if observing:
                batch.add_observer(no_op)
            start = time.perf_counter()
            for _ in range(200):
                batch.step(actions)
            samples.append(time.perf_counter() - start)
            if observing:
                batch.remove_observer(no_op)
    batch.close()
    assert statistics.median(plain) <= statistics.median(observed) * 1.5
