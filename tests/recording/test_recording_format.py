# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks the file, its layout and the Recording that builds it."""

import io
import json
import re
import subprocess
import sys
import zipfile

import numpy as np
import pytest
from helpers import same_steps

import cuju.recording as cr
from cuju.recording import _files


def built(**changes):
    steps = 4
    values = {
        'scenario': '5_vs_5',
        'left_agents': 2,
        'right_agents': 1,
        'actions': np.arange(steps * 3).reshape(steps, 3) % 19,
        'rewards': np.linspace(0, 1, steps * 3, dtype=np.float32).reshape(steps, 3),
        'end': 1,
        'state': b'a state',
        'step': {
            'ball': np.arange((steps + 1) * 3, dtype=np.float32).reshape(steps + 1, 3),
            'score': np.zeros((steps + 1, 2), np.int32),
        },
        'array': {'value': np.arange((steps + 1) * 3, dtype=np.float64).reshape(steps + 1, 3)},
        'provenance': {'episode': np.int64(3), 'name': 'built'},
        'teams': ['red', 'blue'],
        'difficulty': (1.0, 0.6),
        'players': (5, 5),
        'engine': '0.4.0',
    }
    values.update(changes)
    return cr.Recording(**values)


def assert_same(first, second):
    assert first.header == second.header
    for name in ('actions', 'rewards', 'done'):
        assert getattr(first, name).dtype == getattr(second, name).dtype
        assert getattr(first, name).tobytes() == getattr(second, name).tobytes()
    assert first.end == second.end and first.state == second.state
    same_steps(first.step, second.step)
    same_steps(first.array, second.array)


def test_a_recording_round_trips_through_a_path_a_file_and_bytes(tmp_path):
    match = built()
    path = tmp_path / 'below' / 'match.cjr'
    assert match.write(path) is None
    assert_same(cr.read(path), match)
    assert_same(cr.read(str(path)), match)
    with open(path, 'rb') as handle:
        assert_same(cr.read(handle), match)
    buffer = io.BytesIO()
    assert match.write(buffer) is None
    buffer.seek(0)
    assert_same(cr.read(buffer), match)
    data = match.write()
    assert isinstance(data, bytes) and data == path.read_bytes()
    assert_same(cr.read(data), match)
    assert_same(cr.read(bytearray(data)), match)
    assert cr.read(data).provenance == {'episode': 3, 'name': 'built'}


def test_the_same_recording_writes_the_same_bytes():
    assert built().write() == built().write()


def test_the_members_follow_the_layout(tmp_path):
    names = zipfile.ZipFile(io.BytesIO(built().write())).namelist()
    assert names == [
        'header.json',
        'state.npy',
        'actions.npy',
        'rewards.npy',
        'done.npy',
        'end.npy',
        'step/ball.npy',
        'step/score.npy',
        'array/value.npy',
    ]
    archive = zipfile.ZipFile(io.BytesIO(built(state=None).write()))
    assert 'state.npy' not in archive.namelist()
    assert all(info.compress_type == zipfile.ZIP_DEFLATED for info in archive.infolist())
    header = json.loads(archive.read('header.json'))
    assert header == {
        'format': 'cuju-recording',
        'version': 1,
        'scenario': '5_vs_5',
        'left_agents': 2,
        'right_agents': 1,
        'action_set': 'default',
        'reward_terms': 'scoring',
        'difficulty': [1.0, 0.6],
        'steps': 4,
        'players': {'left': 5, 'right': 5},
        'engine': '0.4.0',
        'provenance': {'episode': 3, 'name': 'built'},
        'teams': ['red', 'blue'],
    }


def test_numpy_reads_a_file_without_cuju(tmp_path):
    path = tmp_path / 'match.cjr'
    built().write(path)
    code = (
        'import sys\n'
        "sys.modules['cuju'] = None\n"
        'import json, numpy as np\n'
        'data = np.load(sys.argv[1])\n'
        "assert json.loads(data['header.json'])['scenario'] == '5_vs_5'\n"
        "assert data['actions'].shape == (4, 3) and data['actions'].dtype == np.int16\n"
        "assert data['rewards'].dtype == np.float32 and data['done'].dtype == np.bool_\n"
        "assert data['end'].shape == () and data['end'].dtype == np.uint8 and data['end'] == 1\n"
        "assert data['state'].dtype == np.uint8 and data['state'].tobytes() == b'a state'\n"
        "assert data['step/ball'].shape == (5, 3) and data['array/value'].shape == (5, 3)\n"
    )
    done = subprocess.run([sys.executable, '-c', code, str(path)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_defaults_fill_what_a_builder_leaves_out():
    match = cr.Recording('5_vs_5', 1, np.zeros((3, 1), int))
    assert match.steps == 3 and match.end == 0 and not match.done.any()
    assert match.rewards.shape == (3, 1) and match.state is None and match.step == {}
    ended = cr.Recording('5_vs_5', 1, np.zeros((3, 1), int), end=2)
    assert ended.done.tolist() == [False, False, True]
    flagged = cr.Recording('5_vs_5', 1, np.zeros((2, 1), int), done=[False, True])
    assert flagged.end == 1
    assert cr.read(ended.write()).end == 2


@pytest.mark.parametrize(
    'changes, message',
    [
        ({'scenario': ''}, 'a scenario name'),
        ({'left_agents': -1}, 'left_agents'),
        ({'actions': np.zeros((4, 2), int)}, 'the shape'),
        ({'actions': np.zeros((4, 3), float)}, 'whole number'),
        ({'actions': np.full((4, 3), -2)}, 'from -1'),
        ({'rewards': np.zeros((3, 3))}, 'rewards'),
        ({'action_set': 'big'}, 'action set'),
        ({'end': 3}, 'end'),
        ({'end': 0, 'done': [False, False, False, True]}, 'match the end'),
        ({'done': [True, False, False, True]}, 'last step alone'),
        ({'state': b''}, 'a state with bytes'),
        ({'step': {'ball': np.zeros((4, 3))}}, '5 rows'),
        ({'step': {'a/b': np.zeros((5, 3))}}, 'slash'),
        ({'array': {'text': np.array(['a'] * 5)}}, 'numbers'),
        ({'provenance': {'bad': object()}}, 'provenance'),
        ({'teams': ['one']}, 'two team names'),
    ],
)
def test_a_recording_that_does_not_fit_is_refused(changes, message):
    with pytest.raises(cr.RecordingError, match=message):
        built(**changes)


def test_a_recording_error_is_a_value_error():
    assert issubclass(cr.RecordingError, ValueError)


def test_a_file_that_is_not_a_recording_is_refused(tmp_path):
    for data in (b'', b'not a zip', b'PK\x03\x04broken'):
        with pytest.raises(cr.RecordingError, match='expected a recording'):
            cr.read(data)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('other.txt', 'text')
    with pytest.raises(cr.RecordingError, match=re.escape('header.json')):
        cr.read(buffer.getvalue())
    with pytest.raises(FileNotFoundError):
        cr.read(tmp_path / 'missing.cjr')


def test_a_newer_major_version_and_other_formats_are_refused():
    match = built()
    header = match.header
    members = {'actions': match.actions, 'rewards': match.rewards, 'done': match.done}
    with pytest.raises(cr.RecordingError, match='version 1 or older, found version 2'):
        cr.read(_files.write_archive({**header, 'version': 2}, members, None))
    with pytest.raises(cr.RecordingError, match='the format'):
        cr.read(_files.write_archive({**header, 'format': 'arena'}, members, None))
    with pytest.raises(cr.RecordingError, match='a whole number'):
        cr.read(_files.write_archive({**header, 'version': 'one'}, members, None))


def test_a_file_that_misses_a_member_or_disagrees_is_refused():
    match = built()
    header = match.header
    members = {'actions': match.actions, 'rewards': match.rewards, 'done': match.done}
    with pytest.raises(cr.RecordingError, match=re.escape('end.npy')):
        cr.read(_files.write_archive(header, members, None))
    members['end'] = np.asarray(1, np.uint8)
    cr.read(_files.write_archive(header, members, None))
    with pytest.raises(cr.RecordingError, match='3 steps, found 4'):
        cr.read(_files.write_archive({**header, 'steps': 3}, members, None))
    with pytest.raises(cr.RecordingError, match='complete header'):
        cr.read(_files.write_archive({'format': 'cuju-recording', 'version': 1}, members, None))
