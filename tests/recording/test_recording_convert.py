# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks the converters of GRF dumps and Kaggle episodes, and the reader of dumps."""

import json
import os
import pickle

import numpy as np
import pytest
import torch
from helpers import SHORT, TEAMS, play, same_steps

import cuju
import cuju.dumps
import cuju.recording as cr


def same_recording(first, second):
    for name in ('actions', 'rewards', 'done'):
        assert getattr(first, name).dtype == getattr(second, name).dtype
        assert getattr(first, name).tobytes() == getattr(second, name).tobytes(), name
    assert first.end == second.end
    same_steps(first.step, second.step)


@pytest.mark.parametrize(
    'scenario, left, right', [(SHORT, 1, 0), (TEAMS, 1, 1), (TEAMS, 0, 1), (TEAMS, 2, 1)]
)
def test_a_dump_converts_to_the_recording_that_the_batch_made(tmp_path, scenario, left, right):
    made, path, _ = play(tmp_path, scenario, left, right)
    converted = cr.from_grf_dump(path)
    assert converted.end == made.end == 1 and converted.state == made.state
    assert (converted.left_agents, converted.right_agents) == (left, right)
    assert converted.provenance == {'source': 'grf-dump', 'file': os.path.basename(path)}
    assert converted.difficulty == made.difficulty and converted.players == made.players
    assert converted.scenario == made.scenario and converted.engine == cuju.__version__
    assert converted.steps == len(cuju.dumps.read_dump(path))
    same_recording(converted, made)
    # The state and actions play the match again in the public batch, so the file is exact.
    converted.replay()
    assert cr.read(converted.write()).replay().shape[0] == converted.steps + 1


def test_a_dump_converts_from_a_file_and_from_bytes(tmp_path):
    made, path, _ = play(tmp_path)
    with open(path, 'rb') as handle:
        first = cr.from_grf_dump(handle)
    second = cr.from_grf_dump(open(path, 'rb').read())
    assert 'file' not in first.provenance
    same_recording(first, made)
    assert first.write() == second.write()


def test_a_dump_on_the_gpu_converts_to_the_same_bits(tmp_path, device):
    made, path, _ = play(tmp_path)
    same_recording(cr.from_grf_dump(path, device=device), made)


def _numpy(value):
    # The reader gives arrays as tensors, which a pickle of numpy arrays writes back.
    if isinstance(value, torch.Tensor):
        return value.numpy()
    if isinstance(value, dict):
        return {key: _numpy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_numpy(item) for item in value]
    return value


def _rewritten(path, tmp_path, change):
    entries = [_numpy(entry) for entry in cuju.dumps.read_dump(path)]
    change(entries)
    target = str(tmp_path / 'changed.dump')
    with open(target, 'wb') as handle:
        for entry in entries:
            handle.write(pickle.dumps(entry, protocol=2))
    return target


def test_a_dump_that_differs_from_its_replay_is_refused(tmp_path):
    _, path, _ = play(tmp_path)

    def shot(entries):
        entries[5]['debug']['action'][0] = 'sliding'

    target = _rewritten(path, tmp_path, shot)
    with pytest.raises(cr.RecordingError, match=r'differs from its replay at step \d+ in '):
        cr.from_grf_dump(target)

    def short(entries):
        del entries[-1]

    cut = _rewritten(path, tmp_path, short)
    assert cr.from_grf_dump(cut).end == 0

    def long(entries):
        entries.append(entries[-1])

    with pytest.raises(cr.RecordingError, match='ends the episode at step'):
        cr.from_grf_dump(_rewritten(path, tmp_path, long))


def test_a_dump_of_a_part_of_an_episode_or_of_other_players_is_refused(tmp_path):
    _, path, _ = play(tmp_path)

    def late(entries):
        entries[0]['debug']['frame_cnt'] = 4

    with pytest.raises(cr.RecordingError, match='kick off'):
        cr.from_grf_dump(_rewritten(path, tmp_path, late))

    def bot(entries):
        entries[0]['debug']['config']['players'] = ['agent:left_players=1', 'bot:right_players=1']

    with pytest.raises(cr.RecordingError, match='only agents'):
        cr.from_grf_dump(_rewritten(path, tmp_path, bot))

    def frames(entries):
        entries[0]['debug']['config']['physics_steps_per_frame'] = 5

    with pytest.raises(cr.RecordingError, match='10 physics steps'):
        cr.from_grf_dump(_rewritten(path, tmp_path, frames))

    def action(entries):
        entries[2]['debug']['action'] = ['no_such_action']

    with pytest.raises(cr.RecordingError, match="found 'no_such_action'"):
        cr.from_grf_dump(_rewritten(path, tmp_path, action))


def test_a_truncated_or_empty_dump_is_refused(tmp_path):
    _, path, _ = play(tmp_path)
    data = open(path, 'rb').read()
    for bad in (data[: len(data) // 2], b'', b'text'):
        with pytest.raises(cr.RecordingError, match='expected a GRF dump'):
            cr.from_grf_dump(bad)


def pickled(value, protocol=2):
    return pickle.dumps(value, protocol=protocol)


class Hostile:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        code = f'import pathlib; pathlib.Path({str(self.marker)!r}).write_text("ran")'
        return (exec, (code,))


def test_a_dump_that_names_code_runs_none_of_it(tmp_path):
    marker = tmp_path / 'ran.txt'
    for protocol in (2, 4):
        with pytest.raises(cr.RecordingError, match=r'expected a GRF dump, found .*exec'):
            cr.from_grf_dump(pickled({'observation': Hostile(marker)}, protocol))
    assert not marker.exists()


def test_a_dump_that_is_not_a_run_of_steps_is_refused():
    for data in (pickled([1, 2]), pickled({'a': np.array([1, 2], dtype=object)}), b'\x80\x04'):
        with pytest.raises(cr.RecordingError, match='expected a GRF dump'):
            cr.from_grf_dump(data)


def agent_counts(episode):
    sizes = [len(entry['observation']['players_raw']) for entry in episode['steps'][0]]
    return sizes[0], sizes[1]


@pytest.mark.parametrize('left, right', [(1, 1), (0, 1), (2, 1)])
def test_a_kaggle_episode_converts_to_the_recording_that_the_batch_made(tmp_path, left, right):
    made, _, episode = play(tmp_path, TEAMS, left, right)
    path = tmp_path / 'episode.json'
    path.write_text(json.dumps(episode))
    converted = cr.from_kaggle(path)
    assert agent_counts(episode) == (left, right)
    assert (converted.left_agents, converted.right_agents) == (left, right)
    assert converted.scenario == TEAMS and converted.teams == ['red', 'blue']
    assert converted.state is None and converted.provenance == {'source': 'kaggle'}
    assert converted.steps == made.steps - 1 and converted.end == 0
    assert converted.players == made.players
    assert converted.actions.tobytes() == made.actions[:-1].tobytes()
    for name, array in made.step.items():
        assert converted.step[name].dtype == array.dtype, name
        assert converted.step[name].tobytes() == array[:-1].tobytes(), name
    assert not converted.rewards.any()
    assert cr.read(converted.write()).steps == converted.steps


def test_a_kaggle_episode_that_ended_keeps_its_end(tmp_path):
    _, _, episode = play(tmp_path, TEAMS, 1, 1)
    episode['steps'] = [
        [{**entry, 'status': 'DONE'} for entry in step]
        if index == len(episode['steps']) - 1
        else step
        for index, step in enumerate(episode['steps'])
    ]
    converted = cr.from_kaggle(json.dumps(episode).encode())
    assert converted.end == 1 and converted.done[-1] and not converted.done[:-1].any()


def test_a_kaggle_episode_that_does_not_read_is_refused(tmp_path):
    _, _, episode = play(tmp_path, TEAMS, 1, 1)
    for bad in (b'', b'[1]', b'{"steps": []}', b'{"steps": [[{}], [{}]]}', b'{"steps": 3}'):
        with pytest.raises(cr.RecordingError, match='Kaggle'):
            cr.from_kaggle(bad)
    broken = json.loads(json.dumps(episode))
    del broken['steps'][3][0]['observation']['players_raw'][0]['ball']
    with pytest.raises(cr.RecordingError, match='the same fields at every step'):
        cr.from_kaggle(json.dumps(broken).encode())
    wide = json.loads(json.dumps(episode))
    wide['steps'][2][0]['action'] = [40]
    with pytest.raises(cr.RecordingError, match='sets of GRF'):
        cr.from_kaggle(json.dumps(wide).encode())
