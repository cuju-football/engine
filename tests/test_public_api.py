# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks the public `BatchEnvironment`, which maps onto the private `Batch`."""

import os

import pytest
import torch

import cuju
from conftest import CUDA, actions_at, tensor_bytes
from cuju import errors
from cuju._batch import Batch

NAMES = [
    'BatchEnvironment',
    'Error',
    '__version__',
    'engine_status',
    'errors',
    'scenario_names',
]


def public_names():
    return sorted(name for name in dir(cuju) if not name.startswith('_') or name == '__version__')


def test_the_namespace_holds_exactly_the_listed_names():
    assert sorted(cuju.__all__) == NAMES
    assert public_names() == sorted(NAMES)
    assert dir(cuju) == sorted(NAMES)


@pytest.mark.parametrize(
    'name',
    [
        'ActionLog',
        'ActionRangeError',
        'Digest',
        'EnvironmentIndexError',
        'MultiDeviceBatch',
        'SignatureError',
        'TrackerError',
        'ConfigurationError',
        'bundled_assets',
        'bundled_scenarios',
        'digest',
        'library_path',
        'memory_estimate',
        'replay',
    ],
)
def test_the_names_of_0_3_are_gone(name):
    assert not hasattr(cuju, name)


def test_the_scenarios_have_a_name_function():
    names = cuju.scenario_names()
    assert '5_vs_5' in names and '11_vs_11_kaggle' in names
    assert names == sorted(names)


def test_the_classes_of_the_design_stand_where_it_puts_them():
    from cuju import gfootball_engine

    for name in ('SignatureError', 'TrackerError'):
        assert getattr(gfootball_engine, name).__module__.startswith('cuju.gfootball_engine')
        assert not hasattr(errors, name)
    for name in ('ActionRangeError', 'EnvironmentIndexError'):
        assert not hasattr(errors, name)
    expected = {
        'Error': (Exception,),
        'ConfigurationError': (ValueError,),
        'ChoiceError': (errors.ConfigurationError, KeyError),
        'ArgumentError': (ValueError, IndexError),
        'StateError': (ValueError, RuntimeError),
        'ClosedError': (RuntimeError,),
        'DeviceError': (RuntimeError,),
        'UndefinedPathError': (RuntimeError,),
        'LibraryError': (RuntimeError,),
        'MissingFileError': (FileNotFoundError,),
    }
    assert sorted(errors.__all__) == sorted(expected)
    for name, bases in expected.items():
        for base in bases:
            assert issubclass(getattr(errors, name), base), name
        assert issubclass(getattr(errors, name), cuju.Error)


def same_outputs(device, left, right, **options):
    """Plays a public and a private batch of one scenario side by side, and compares them."""
    names = {'raw': 'with_raw', 'frames': 'render_frames'}
    private_options = {
        names.get(key, key): [value] if key == 'difficulty' else value
        for key, value in options.items()
        if key != 'frame_size'
    }
    with (
        cuju.BatchEnvironment(
            '5_vs_5',
            environment_count=3,
            device=device,
            seed=5,
            left_agents=left,
            right_agents=right,
            **options,
        ) as public,
        Batch(
            '5_vs_5',
            environment_count=3,
            device=device,
            seed=5,
            agent_counts=[(left, right)],
            signed_actions=False,
            **private_options,
        ) as private,
    ):
        assert public.agents == private.agents == left + right
        first = public.reset(), private.reset()
        assert tensor_bytes(first[0]) == tensor_bytes(first[1])
        for step in range(60):
            actions = actions_at(step, 3, public.agents, public.action_count).to(device)
            a = public.step(actions)
            b = private.step(actions)
            assert tensor_bytes(a[0]) == tensor_bytes(b[0])
            assert tensor_bytes(a[1]) == tensor_bytes(b[1])
            assert a[2].dtype == torch.bool
            assert a[2].tolist() == b[2].bool().tolist()
            assert tensor_bytes(a[3]['final_observation']) == tensor_bytes(
                b[3]['final_observation']
            )
            assert tensor_bytes(a[3]['score_reward']) == tensor_bytes(b[3]['score_reward'])
        stacked = options.get('stacked', False)
        assert public.get_state(1, stacked) == private.get_state(1, stacked)


@pytest.mark.parametrize('agents', [(1, 0), (2, 1), (0, 0)])
def test_the_public_class_gives_the_outputs_of_the_private_one(device, agents):
    same_outputs(device, *agents)


def test_the_options_that_change_results_reach_the_private_class(device):
    same_outputs(device, 2, 2, representation='extracted', rewards='scoring,checkpoints')
    same_outputs(device, 1, 1, stacked=True, action_set='v2', difficulty=(0.3, None))


def test_raw_adds_the_raw_observation_and_its_final_form(device):
    with cuju.BatchEnvironment('5_vs_5', environment_count=2, device=device, raw=True) as batch:
        batch.reset()
        _, _, _, information = batch.step(torch.zeros((2, 1), dtype=torch.int32))
        assert set(information) == {'final_observation', 'score_reward', 'final_raw'}
        assert 'ball' in batch.raw() and 'ball' in information['final_raw']
    with cuju.BatchEnvironment('5_vs_5', environment_count=2, device=device) as plain:
        plain.reset()
        assert set(plain.step(torch.zeros((2, 1), dtype=torch.int32))[3]) == {
            'final_observation',
            'score_reward',
        }
        with pytest.raises(errors.LibraryError, match='raw'):
            plain.raw()
        with pytest.raises(errors.LibraryError, match='frame'):
            plain.frame()
    with cuju.BatchEnvironment(
        '5_vs_5', environment_count=2, device=device, representation='raw'
    ) as raw:
        observation = raw.reset()
        assert isinstance(observation, dict) and raw.raw().keys() == observation.keys()
        assert 'final_raw' in raw.step(torch.zeros((2, 1), dtype=torch.int32))[3]


def test_the_attributes_describe_the_batch(device):
    with cuju.BatchEnvironment(
        '5_vs_5', environment_count=4, device=device, left_agents=3, right_agents=2, stacked=True
    ) as batch:
        assert (batch.environment_count, batch.agents) == (4, 5)
        assert (batch.left_agents, batch.right_agents) == (3, 2)
        assert (batch.left_players, batch.right_players) == (5, 5)
        assert batch.action_count == 19 and batch.stacked and batch.scenario_name == '5_vs_5'
        assert batch.device == torch.device('cpu' if device == 'cpu' else batch.device)
        assert isinstance(batch.device, torch.device)
        assert batch.device.type == device
        assert tuple(batch.difficulty.shape) == (4, 2)
        assert batch.memory_bytes > 0
        assert set(batch.run_config) >= {'threads', 'debug', 'allocator', 'log'}
        assert batch.run_config['threads'] == {'value': 0, 'source': 'default'}


def test_the_spaces_follow_the_agents():
    pytest.importorskip('gymnasium')
    with cuju.BatchEnvironment('5_vs_5', environment_count=3, device='cpu', left_agents=2) as batch:
        assert batch.single_observation_space.shape == (2, 115)
        assert batch.observation_space.shape == (3, 2, 115)
        assert batch.single_action_space.nvec.tolist() == [19, 19]
        assert batch.action_space.shape == (3, 2)


def test_a_cpu_batch_refuses_actions_outside_the_set():
    with cuju.BatchEnvironment('5_vs_5', environment_count=2, device='cpu') as batch:
        batch.reset()
        for value in (-1, 19):
            with pytest.raises(errors.ArgumentError, match='from 0 to 18') as raised:
                batch.step(torch.full((2, 1), value, dtype=torch.int32))
            assert isinstance(raised.value, IndexError) and isinstance(raised.value, ValueError)
        with pytest.raises(errors.ArgumentError):
            batch.step_async(torch.full((2, 1), -3, dtype=torch.int32))
        with pytest.raises(errors.ArgumentError, match='shape'):
            batch.step(torch.zeros((2, 2), dtype=torch.int32))
        with pytest.raises(TypeError):
            batch.step(torch.zeros((2, 1)))


@pytest.mark.skipif(not CUDA, reason='no CUDA device or library')
def test_a_gpu_batch_plays_an_action_outside_the_set_as_idle_unless_debug_is_on():
    def play(value, **config):
        with cuju.BatchEnvironment(
            '5_vs_5', environment_count=2, device='cuda', config=config
        ) as batch:
            batch.reset()
            observation, *_ = batch.step(torch.full((2, 1), value, dtype=torch.int32).cuda())
            return tensor_bytes(observation)

    idle = play(0)
    assert play(-1) == idle and play(99) == idle
    for value in (-1, 99):
        with pytest.raises(errors.ArgumentError, match='from 0 to 18'):
            play(value, debug=True)


def test_closing_ends_every_call(device):
    batch = cuju.BatchEnvironment('5_vs_5', environment_count=1, device=device)
    batch.reset()
    batch.close()
    batch.close()
    for call in (
        batch.reset,
        batch.reset_done,
        lambda: batch.step(torch.zeros((1, 1), dtype=torch.int32)),
    ):
        with pytest.raises(errors.ClosedError):
            call()
    with pytest.raises(TypeError):
        import pickle

        pickle.dumps(batch)


def test_errors_of_the_design():
    with pytest.raises(errors.LibraryError, match='first step'):
        with cuju.BatchEnvironment('5_vs_5', environment_count=1, device='cpu') as batch:
            batch.step(torch.zeros((1, 1), dtype=torch.int32))
    with pytest.raises(errors.ConfigurationError, match='players') as raised:
        cuju.BatchEnvironment('5_vs_5', environment_count=1, device='cpu', left_agents=6)
    assert isinstance(raised.value, ValueError)
    with pytest.raises(errors.ConfigurationError, match='agent counts'):
        cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=1, device='cpu')
    with pytest.raises(errors.ConfigurationError, match='left_agents'):
        cuju.BatchEnvironment('5_vs_5', environment_count=1, device='cpu', left_agents=-1)
    with pytest.raises(errors.ChoiceError):
        cuju.BatchEnvironment('5_vs_5', environment_count=1, device='cpu', representation='x')
    with pytest.raises(errors.ChoiceError):
        cuju.BatchEnvironment('5_vs_5', environment_count=1, device='tpu')
    with pytest.raises(errors.ConfigurationError, match='difficulty'):
        cuju.BatchEnvironment('5_vs_5', environment_count=1, device='cpu', difficulty=0.5)
    with pytest.raises(errors.MissingFileError):
        cuju.BatchEnvironment('no_such_scenario.py', environment_count=1, device='cpu')
    with cuju.BatchEnvironment('5_vs_5', environment_count=2, device='cpu') as batch:
        with pytest.raises(errors.ArgumentError, match='from 0 to 1') as raised:
            batch.get_state(2)
        assert isinstance(raised.value, IndexError)
        with pytest.raises(errors.StateError):
            batch.set_state(0, b'\x00' * 8)


def test_reset_takes_seeds_a_mask_and_reverse_flags(device):
    with cuju.BatchEnvironment('5_vs_5', environment_count=3, device=device) as batch:
        first = tensor_bytes(batch.reset(seeds=[1, 2, 3]))
        batch.reset(seeds=[1, 2, 3])
        assert tensor_bytes(batch.reset(seeds=[1, 2, 3])) == first
        batch.step(torch.zeros((3, 1), dtype=torch.int32).to(device))
        observation = batch.reset(mask=[True, False, False], seeds=[1, 2, 3])
        assert observation.shape[0] == 3
        batch.reset(seeds=0, reverse=[0, 1, 0])


def test_the_asynchronous_step_equals_the_step():
    with (
        cuju.BatchEnvironment('5_vs_5', environment_count=3, device='cpu') as a,
        cuju.BatchEnvironment('5_vs_5', environment_count=3, device='cpu') as b,
    ):
        a.reset(), b.reset()
        actions = actions_at(1, 3, 1, 19)
        a.step_async(actions)
        waited = a.step_wait()
        direct = b.step(actions)
        assert waited[2].dtype == torch.bool
        for x, y in zip(waited[:3], direct[:3], strict=True):
            assert tensor_bytes(x) == tensor_bytes(y)


def test_reset_done_starts_the_ended_environments():
    with cuju.BatchEnvironment(
        'academy_empty_goal_close', environment_count=2, device='cpu', auto_reset=False
    ) as batch:
        batch.reset()
        for _ in range(400):
            _, _, done, _ = batch.step(torch.zeros((2, 1), dtype=torch.int32))
            if bool(done.any()):
                break
        assert bool(done.any())
        assert batch.reset_done() is None
        assert not bool(batch._batch._outputs()[2].any())


def test_states_round_trip_in_grf_format(device):
    with cuju.BatchEnvironment('5_vs_5', environment_count=2, device=device) as batch:
        batch.reset()
        for step in range(5):
            batch.step(actions_at(step, 2, 1, 19).to(device))
        state = batch.get_state(0)
        batch.set_state(1, state)
        assert batch.get_state(1) == state


def test_difficulty_changes_at_the_next_reset(device):
    with cuju.BatchEnvironment(
        '5_vs_5', environment_count=3, device=device, difficulty=(0.2, 0.4)
    ) as batch:
        assert batch.difficulty[0].tolist() == pytest.approx([0.2, 0.4])
        batch.set_difficulty((0.9, None), mask=[False, True, False])
        batch.reset()
        rows = batch.difficulty.tolist()
        assert rows[0] == pytest.approx([0.2, 0.4])
        assert rows[1][0] == pytest.approx(0.9) and rows[1][1] == pytest.approx(0.05)
        batch.set_difficulty(torch.tensor([[0.1, 0.1], [0.2, 0.2], [float('nan'), 0.3]]))
        assert batch.difficulty[2].tolist() == pytest.approx([0.05, 0.3])
        with pytest.raises(errors.ConfigurationError):
            batch.set_difficulty((1.5, 0.0))
        with pytest.raises(errors.ArgumentError):
            batch.set_difficulty(torch.zeros((2, 2)))


def test_frames_need_the_renderer_and_the_media():
    from cuju import _media, _pixels

    if not os.path.isfile(_pixels.library_path()):
        pytest.skip(f'needs the renderer library {_pixels.library_path()}, built with the renderer')
    if not os.environ.get('CUJU_GRF_MEDIA'):
        try:
            _media.root()
        except errors.MissingFileError:
            pytest.skip("needs GRF's media, see tests/test_render_frames.py")
    with cuju.BatchEnvironment(
        '5_vs_5', environment_count=1, device='cpu', frames=True, frame_size=(320, 180)
    ) as batch:
        batch.reset()
        assert tuple(batch.frame().shape) == (1, 180, 320, 3)
