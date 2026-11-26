# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks that the episodes of a batch become GRF dumps that play the same match again."""

import os
import pickle

import numpy as np
import pytest
import torch

import cuju
import cuju.dumps
from conftest import CUDA, actions_at, tensor_bytes
from cuju._batch import Batch

DIRECTIONS = [
    'left',
    'top_left',
    'top',
    'top_right',
    'right',
    'bottom_right',
    'bottom',
    'bottom_left',
]
FLIPPED = {name: DIRECTIONS[(index + 4) % 8] for index, name in enumerate(DIRECTIONS)}
DEFAULT_ACTIONS = [
    'idle',
    *DIRECTIONS,
    'long_pass',
    'high_pass',
    'short_pass',
    'shot',
    'sprint',
    'release_direction',
    'release_sprint',
    'sliding',
    'dribble',
    'release_dribble',
]


def play(batch, writer, episodes, environment, device):
    """Steps the batch with the actions of `actions_at` until `environment` ends `episodes`."""
    ended = step = 0
    raws = []
    while ended < episodes:
        actions = actions_at(step, batch.environment_count, batch.agents, batch.action_count)
        _, _, done, _ = batch.step(actions.to(device))
        if writer is not None:
            writer.record(actions)
        if batch._raw_fields is not None:
            source = batch.final_raw() if bool(done[environment]) else batch.raw()
            raws.append({name: tensor[environment].clone() for name, tensor in source.items()})
        ended += int(done[environment])
        step += 1
    return raws


def same(first, second, where='dump'):
    """Asserts that two loaded dumps or steps are equal, apart from the wall-clock times."""
    if isinstance(first, torch.Tensor):
        assert isinstance(second, torch.Tensor) and first.dtype == second.dtype, where
        assert first.shape == second.shape and tensor_bytes(first) == tensor_bytes(second), where
    elif isinstance(first, dict):
        assert isinstance(second, dict) and list(first) == list(second), where
        for key in first:
            if key != 'time':
                same(first[key], second[key], f'{where}.{key}')
    elif isinstance(first, list):
        assert isinstance(second, list) and len(first) == len(second), where
        for index, (a, b) in enumerate(zip(first, second, strict=True)):
            same(a, b, f'{where}[{index}]')
    else:
        assert type(first) is type(second) and first == second, (where, first, second)


def replay_dump(path, device, directory, with_raw=True):
    """Replays a dump in a new batch from its configuration and actions, and returns its dump."""
    steps = cuju.dumps.read_dump(path)
    config = steps[0]['debug']['config']
    left = right = 0
    if config['players']:
        fields = dict(part.split('=') for part in config['players'][0].split(':')[1].split(','))
        left, right = int(fields['left_players']), int(fields['right_players'])
    batch = Batch(
        f'{config["level"]}_l{left}_r{right}',
        environment_count=1,
        device=device,
        with_raw=with_raw,
        game_engine_random_seed=config['game_engine_random_seed'],
        episode_number=config['episode_number'] - 2,
        physics_steps_per_frame=config['physics_steps_per_frame'],
    )
    writer = cuju.dumps.DumpWriter(batch, directory)
    batch.reset(reverse=int(config['reverse_team_processing']))
    for step in steps:
        row = [
            DEFAULT_ACTIONS.index(FLIPPED.get(name, name) if agent >= left else name)
            for agent, name in enumerate(step['debug']['action'])
        ]
        actions = torch.tensor([row + [0] * (batch.agents - len(row))], dtype=torch.int32)
        _, _, done, _ = batch.step(actions.to(device))
        writer.record(actions)
    assert bool(done[0])
    batch.close()
    return cuju.dumps.read_dump(writer.paths[0])


def test_full_episode_dumps_hold_grf_steps(tmp_path):
    batch = Batch(
        'academy_3_vs_1_with_keeper_l1_r1', environment_count=3, device='cpu', with_raw=True
    )
    writer = cuju.dumps.DumpWriter(batch, tmp_path, environments=[0, 2])
    batch.reset()
    raws = play(batch, writer, 2, 2, 'cpu')
    paths = [path for path in writer.paths if 'environment_2_' in path]
    assert [os.path.basename(path) for path in paths][:2] == [
        'episode_done_environment_2_episode_1.dump',
        'episode_done_environment_2_episode_2.dump',
    ]
    steps = cuju.dumps.read_dump(paths[0])
    config = steps[0]['debug']['config']
    assert config['level'] == 'academy_3_vs_1_with_keeper'
    assert config['players'] == ['agent:left_players=1,right_players=1']
    assert config['episode_number'] == 2 and isinstance(config['reverse_team_processing'], bool)
    assert 0 <= config['game_engine_random_seed'] <= 2000000000
    assert all('config' not in step['debug'] for step in steps[1:])
    assert [step['debug']['frame_cnt'] for step in steps] == list(range(len(steps)))
    assert steps[-1]['cumulative_reward'] == sum(step['reward'] for step in steps)
    for index, step in enumerate(steps):
        actions = actions_at(index, 3, 2, 19)[2].tolist()
        right = DEFAULT_ACTIONS[actions[1]]
        assert step['debug']['action'] == [DEFAULT_ACTIONS[actions[0]], FLIPPED.get(right, right)]
        observation, raw = step['observation'], raws[index]
        assert torch.equal(observation['left_team'], raw['left_team'][0, :4].double())
        assert torch.equal(observation['right_team'], raw['right_team'][0, :2].double())
        assert observation['right_team_designated_player'] == int(raw['designated'][1])
        assert observation['right_agent_controlled_player'] == [int(raw['active'][1])]
        assert observation['steps_left'] == int(raw['steps_left'][0])


def test_a_team_without_agents_has_its_designated_player(tmp_path):
    batch = Batch(
        'academy_counterattack_easy_l2_r0', environment_count=1, device='cpu', with_raw=True
    )
    writer = cuju.dumps.DumpWriter(batch, tmp_path)
    batch.reset()
    raws = play(batch, writer, 1, 0, 'cpu')
    steps = cuju.dumps.read_dump(writer.paths[0])
    assert len(steps) == len(raws)
    for step, raw in zip(steps, raws, strict=True):
        observation = step['observation']
        assert observation['left_team_designated_player'] == int(raw['designated'][0])
        # GRF: the right team has a designated player too, although no agent sees it.
        assert -1 <= observation['right_team_designated_player'] < len(observation['right_team'])
        assert observation['right_agent_sticky_actions'] == []


def test_dump_plays_again_from_its_actions(tmp_path):
    batch = Batch(
        'academy_pass_and_shoot_with_keeper_l1_r1',
        environment_count=2,
        device='cpu',
        seed=3,
        with_raw=True,
    )
    writer = cuju.dumps.DumpWriter(batch, tmp_path / 'first', environments=[1])
    batch.reset()
    play(batch, writer, 1, 1, 'cpu')
    original = cuju.dumps.read_dump(writer.paths[0])
    replayed = replay_dump(writer.paths[0], 'cpu', tmp_path / 'again')
    same(
        original[0]['debug']['config']['episode_number'],
        replayed[0]['debug']['config']['episode_number'],
    )
    for steps in (original, replayed):
        del steps[0]['debug']['config']
    same(original, replayed)


def test_goal_dumps_end_at_the_goal(tmp_path):
    batch = Batch('academy_empty_goal_close', environment_count=1, device='cpu')
    writer = cuju.dumps.DumpWriter(
        batch, tmp_path, write_full_episode_dumps=False, write_goal_dumps=True
    )
    batch.reset()
    actions = torch.full((1, 1), DEFAULT_ACTIONS.index('right'), dtype=torch.int32)
    for _ in range(400):
        _, _, done, _ = batch.step(actions)
        writer.record(actions)
        if bool(done[0]):
            break
    assert len(writer.paths) == 1
    assert os.path.basename(writer.paths[0]).startswith('score_environment_0_episode_1_step_')
    steps = cuju.dumps.read_dump(writer.paths[0])
    # GRF: the 100 steps before the goal at most, then the step that scored.
    assert len(steps) <= 101 and steps[-1]['reward'] == 1
    assert 'config' in steps[0]['debug'] and 'observation' not in steps[0]


def test_dump_frequency_selects_every_other_episode(tmp_path):
    batch = Batch('academy_empty_goal_close', environment_count=1, device='cpu')
    writer = cuju.dumps.DumpWriter(batch, tmp_path, dump_frequency=2)
    batch.reset()
    actions = torch.full((1, 1), DEFAULT_ACTIONS.index('right'), dtype=torch.int32)
    episodes = 0
    while episodes < 3:
        _, _, done, _ = batch.step(actions)
        writer.record(actions)
        episodes += int(done[0])
    assert [os.path.basename(path) for path in writer.paths] == [
        'episode_done_environment_0_episode_1.dump',
        'episode_done_environment_0_episode_3.dump',
    ]


def test_a_writer_changes_no_output(tmp_path):
    def outputs(with_writer):
        batch = Batch('5_vs_5_l2_r1', environment_count=2, device='cpu')
        writer = cuju.dumps.DumpWriter(batch, tmp_path) if with_writer else None
        result = [tensor_bytes(batch.reset())]
        for step in range(40):
            actions = actions_at(step, 2, 3, 19)
            observation, reward, done, _ = batch.step(actions)
            if writer is not None:
                writer.record(actions)
            result += [tensor_bytes(observation), tensor_bytes(reward), tensor_bytes(done)]
        return [*result, batch.get_state(0), batch.get_state(1)]

    assert outputs(True) == outputs(False)


def test_a_writer_met_mid_episode_starts_with_the_next(tmp_path):
    batch = Batch('academy_empty_goal_close', environment_count=1, device='cpu')
    batch.reset()
    actions = torch.full((1, 1), DEFAULT_ACTIONS.index('right'), dtype=torch.int32)
    batch.step(actions)
    writer = cuju.dumps.DumpWriter(batch, tmp_path)
    episodes = 0
    while episodes < 2:
        _, _, done, _ = batch.step(actions)
        writer.record(actions)
        episodes += int(done[0])
    assert [os.path.basename(path) for path in writer.paths] == [
        'episode_done_environment_0_episode_2.dump'
    ]


def test_difficulties_name_a_scenario_file(tmp_path):
    batch = Batch(
        'academy_empty_goal_close', environment_count=1, device='cpu', difficulty=[(0.5, None)]
    )
    writer = cuju.dumps.DumpWriter(batch, tmp_path)
    batch.reset()
    play(batch, writer, 1, 0, 'cpu')
    level = cuju.dumps.read_dump(writer.paths[0])[0]['debug']['config']['level']
    assert level == 'academy_empty_goal_close_difficulty_0p5_none'
    with open(tmp_path / (level + '.py'), encoding='utf-8') as handle:
        source = handle.read()
    assert "import_module('gfootball.scenarios.academy_empty_goal_close')" in source
    assert 'builder.config().left_team_difficulty = 0.5\n' in source
    assert 'right_team_difficulty' not in source


def test_read_dump_refuses_other_objects(tmp_path):
    path = tmp_path / 'other.dump'
    path.write_bytes(pickle.dumps(os.system))
    with pytest.raises(pickle.UnpicklingError):
        cuju.dumps.read_dump(path)


def test_read_dump_refuses_values_nested_too_deep(tmp_path):
    value = {}
    for _ in range(40):
        value = {'next': value}
    path = tmp_path / 'deep.dump'
    path.write_bytes(pickle.dumps(value))
    with pytest.raises(pickle.UnpicklingError, match='nested'):
        cuju.dumps.read_dump(path)


def test_read_dump_refuses_a_value_bomb(tmp_path):
    value = [0] * 100
    for _ in range(5):
        value = [value] * 100
    path = tmp_path / 'bomb.dump'
    path.write_bytes(pickle.dumps({'bomb': value}))
    assert path.stat().st_size < 4096
    with pytest.raises(pickle.UnpicklingError, match='more values'):
        cuju.dumps.read_dump(path)


def test_iterate_dump_equals_read_dump(tmp_path):
    steps = [{'step': index, 'values': [index, 'a', (1.5, None)]} for index in range(4)]
    path = tmp_path / 'steps.dump'
    path.write_bytes(b''.join(pickle.dumps(step) for step in steps))
    iterator = cuju.dumps.iterate_dump(path)
    assert next(iterator) == steps[0]
    assert list(iterator) == steps[1:]
    assert list(cuju.dumps.iterate_dump(path)) == cuju.dumps.read_dump(path) == steps


class _Runs:
    def __reduce__(self):
        return (eval, ('1 + 1',))


@pytest.mark.parametrize('protocol', [0, 1, 2, 4])
def test_read_dump_reads_what_numpy_pickles(tmp_path, protocol):
    step = {
        'a': np.arange(6, dtype=np.int64).reshape(2, 3),
        'b': np.float32(1.5),
        'c': np.int32(-7),
        'd': np.bool_(True),
        'e': [np.array([0.5], dtype=np.float64)],
    }
    path = tmp_path / 'numpy.dump'
    path.write_bytes(pickle.dumps(step, protocol=protocol) * 2)
    for steps in (cuju.dumps.read_dump(path), list(cuju.dumps.iterate_dump(path))):
        assert len(steps) == 2
        assert steps[0]['a'].tolist() == [[0, 1, 2], [3, 4, 5]]
        assert steps[0]['a'].dtype == torch.int64
        assert (steps[0]['b'], steps[0]['c'], steps[0]['d']) == (1.5, -7, True)
        assert steps[0]['e'][0].tolist() == [0.5]


READERS = [cuju.dumps.read_dump, lambda path: list(cuju.dumps.iterate_dump(path))]


@pytest.mark.parametrize('reader', READERS)
def test_dump_readers_refuse_code(tmp_path, reader):
    for value in (_Runs(), {'step': [_Runs()]}, os.system):
        path = tmp_path / 'code.dump'
        path.write_bytes(pickle.dumps(value))
        with pytest.raises(pickle.UnpicklingError):
            reader(path)


@pytest.mark.parametrize('reader', READERS)
def test_dump_readers_refuse_an_array_that_claims_more_than_its_size(tmp_path, reader):
    data = pickle.dumps({'a': np.zeros(4, dtype=np.float32)}, protocol=2)
    claim = data.replace(b'K\x04\x85', b'J\xff\xff\xff\x7f\x85', 1)
    assert claim != data
    path = tmp_path / 'claim.dump'
    path.write_bytes(claim)
    with pytest.raises(pickle.UnpicklingError, match='more values'):
        reader(path)


def test_dumps_of_a_match_read_by_both_readers(tmp_path):
    batch = Batch('academy_empty_goal_close', environment_count=1, device='cpu')
    writer = cuju.dumps.DumpWriter(batch, tmp_path)
    batch.reset()
    play(batch, writer, 1, 0, 'cpu')
    (path,) = writer.paths
    steps = cuju.dumps.read_dump(path)
    streamed = list(cuju.dumps.iterate_dump(path))
    assert len(steps) == len(streamed) > 1
    for step, other in zip(steps, streamed, strict=True):
        assert pickle.dumps(_listed(step)) == pickle.dumps(_listed(other))


def _listed(value):
    if isinstance(value, torch.Tensor):
        return (str(value.dtype), tuple(value.shape), value.flatten().tolist())
    if isinstance(value, dict):
        return {key: _listed(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_listed(item) for item in value]
    return value


@pytest.mark.skipif(not CUDA, reason='no CUDA device or library')
def test_gpu_dumps_equal_cpu_dumps(tmp_path):
    loaded = []
    for device in ('cpu', 'cuda'):
        batch = Batch(
            'academy_3_vs_1_with_keeper_l1_r1', environment_count=2, device=device, with_raw=True
        )
        writer = cuju.dumps.DumpWriter(batch, tmp_path / device, write_goal_dumps=True)
        batch.reset()
        play(batch, writer, 2, 1, device)
        loaded.append([cuju.dumps.read_dump(path) for path in writer.paths])
        for step in (step for steps in loaded[-1] for step in steps if 'config' in step['debug']):
            step['debug']['config']['tracesdir'] = None
    same(loaded[0], loaded[1])


def test_a_writer_raises_cuju_errors(tmp_path):
    batch = Batch('academy_empty_goal_close', environment_count=1, device='cpu')
    with pytest.raises(cuju.errors.ArgumentError, match=r'found \[1\]'):
        cuju.dumps.DumpWriter(batch, tmp_path, environments=[1])
    writer = cuju.dumps.DumpWriter(batch, tmp_path)
    batch.reset()
    actions = torch.zeros((1, 1), dtype=torch.int32)
    batch.step(actions)
    with pytest.raises(cuju.errors.ArgumentError, match='actions of the step'):
        writer.record(torch.zeros((1, 2), dtype=torch.int32))
    with pytest.raises(cuju.errors.ArgumentError, match='from -19 to 18, found 19'):
        writer.record(torch.full((1, 1), 19, dtype=torch.int32))
    batch.close()
