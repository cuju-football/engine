# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks the private names that cuju-arena and cuju-render import."""

import inspect

from cuju import _action_log, _batch, _kaggle, _raw, _state


def parameters(function):
    return list(inspect.signature(function).parameters)


def test_unwrap_returns_the_batch_of_a_wrapper():
    class Wrapper:
        _batch = object()

    assert parameters(_batch.unwrap) == ['batch']
    assert _batch.unwrap(Wrapper()) is Wrapper._batch
    inner = object()
    assert _batch.unwrap(inner) is inner


def test_batch_keeps_the_constructor_options():
    options = parameters(_batch.Batch.__init__)
    assert options[:2] == ['self', 'assets']
    for option in (
        'environment_count',
        'representation',
        'rewards',
        'action_set',
        'seed',
        'auto_reset',
        'with_raw',
        'difficulty',
        'device',
        'threads',
        'channel_dimensions',
        'stacked',
        'agent_counts',
        'render_frames',
        'render_resolution_x',
        'render_resolution_y',
        'physics_steps_per_frame',
        'game_engine_random_seed',
        'episode_number',
    ):
        assert option in options


def test_batch_keeps_the_methods_of_the_family():
    for name in (
        'step',
        'reset',
        'close',
        'raw',
        'final_raw',
        'get_state',
        'set_state',
        'roles',
        'add_observer',
        'remove_observer',
    ):
        assert callable(getattr(_batch.Batch, name))
    assert isinstance(_batch.Batch.observers, property)
    assert parameters(_batch.Batch.add_observer) == ['self', 'observer']
    assert parameters(_batch.Batch.remove_observer) == ['self', 'observer']
    assert parameters(_batch.Batch.get_state)[:2] == ['self', 'index']
    assert parameters(_batch.Batch.set_state)[:3] == ['self', 'index', 'state']
    assert parameters(_batch.Batch.roles) == ['self', 'index']
    assert 'reverse' in parameters(_batch.Batch.reset)


def test_observers_are_called_with_the_batch_the_done_flags_and_the_information():
    batch = _batch.Batch('academy_empty_goal_close', environment_count=1, device='cpu')
    calls = []

    def observer(*arguments):
        calls.append(arguments)

    try:
        assert batch.add_observer(observer) is observer
        assert batch.observers == (observer,)
        batch.reset()
        batch.step(_idle(batch))
        assert calls
        for name in ('left_players', 'right_players', 'difficulty'):
            assert hasattr(batch, name)
        assert all(len(call) == 3 and call[0] is batch for call in calls)
        assert isinstance(calls[-1][2], dict)
        batch.remove_observer(observer)
        assert batch.observers == ()
    finally:
        batch.close()


def _idle(batch):
    import torch

    return torch.zeros(batch.environment_count, batch.agents, dtype=torch.int32)


def test_state_keeps_the_layout_and_the_readers():
    assert _state.Field._fields == ('path', 'offset', 'size', 'type')
    assert parameters(_state.layout) == []
    assert parameters(_state.gather) == ['state', 'fields', 'frame']
    assert parameters(_state.read) == ['state', 'fields', 'frame']
    for name in ('layout', 'Field', 'gather', 'read'):
        assert name in _state.__all__


def test_raw_keeps_grf_observations():
    assert parameters(_raw.grf_observations) == ['batch', 'environment', 'agents', 'final']


def test_kaggle_keeps_the_runners_and_the_helpers():
    assert parameters(_kaggle.AgentRunner.__init__) == ['self', 'function', 'configuration']
    assert parameters(_kaggle.AgentRunner.act) == ['self', 'obs', 'count']
    assert parameters(_kaggle.KaggleAgent.__init__) == [
        'self',
        'function',
        'batch',
        'agent',
        'configuration',
        'agents',
    ]
    assert parameters(_kaggle.action_index) == ['value', 'count']
    assert parameters(_kaggle.action_indices) == ['value', 'count', 'players']
    assert callable(_kaggle.accepts_configuration)
    assert callable(_kaggle.observation)


def test_action_log_keeps_the_digest_and_the_replay():
    assert parameters(_action_log.Digest.update) == ['self', 'value']
    assert parameters(_action_log.Digest.hexdigest) == ['self']
    assert parameters(_action_log.replay)[:2] == ['path', 'device']
