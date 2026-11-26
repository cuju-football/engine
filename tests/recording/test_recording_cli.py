# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks the commands of cuju recording."""

import io
import json
import os
import zipfile

import numpy as np
import pytest
from helpers import SHORT, TEAMS, play

import cuju.recording as cr
from cuju import __main__ as cli


def run(capsys, *words):
    status = cli.main(['recording', *words])
    out, err = capsys.readouterr()
    return status, out, err


@pytest.fixture(scope='module')
def files(tmp_path_factory):
    folder = tmp_path_factory.mktemp('cli')
    made, dump, _ = play(folder, SHORT)
    path = folder / 'match.cjr'
    made.write(path)
    _, _, episode = play(folder, TEAMS, 1, 1)
    episode_path = folder / 'episode.json'
    episode_path.write_text(json.dumps(episode))
    return {
        'folder': folder,
        'recording': path,
        'match': made,
        'dump': dump,
        'kaggle': episode_path,
    }


def test_show_prints_what_a_recording_holds(files, capsys):
    status, out, err = run(capsys, 'show', str(files['recording']))
    assert status == 0 and err == ''
    match = files['match']
    assert SHORT in out and f'steps      {match.steps}, terminated' in out
    assert '1 left, 0 right' in out and 'state      kept' in out and 'left_team' in out
    status, out, _ = run(capsys, 'show', str(files['recording']), '--json')
    answer = json.loads(out)
    assert status == 0 and next(iter(answer)) == 'schema' and answer['schema'] == 'recording.show/1'
    assert out.startswith('{\n  "schema"')
    assert answer['header']['scenario'] == SHORT and answer['header']['steps'] == match.steps
    assert answer['end'] == 'terminated' and answer['score'] == match.step['score'][-1].tolist()
    assert answer['state'] is True and answer['step']['ball'] == {
        'dtype': 'float32',
        'shape': [match.steps + 1, 3],
    }


def test_check_validates_and_replays(files, capsys):
    path = str(files['recording'])
    status, out, _ = run(capsys, 'check', path)
    assert status == 0 and 'valid steps' in out
    status, out, _ = run(capsys, 'check', path, '--replay')
    assert status == 0 and "equals the recording's" in out
    status, out, _ = run(capsys, 'check', path, '--replay', '--device', 'cpu', '--json')
    answer = json.loads(out)
    assert status == 0 and answer['schema'] == 'recording.check/1'
    assert answer['replayed'] is True and answer['difference'] is None


def test_check_exits_1_for_a_recording_that_does_not_reproduce(files, capsys):
    tampered = cr.read(files['recording'])
    tampered.actions[2:] = 5
    path = files['folder'] / 'tampered.cjr'
    tampered.write(path)
    status, out, _ = run(capsys, 'check', str(path))
    assert status == 0
    status, out, _ = run(capsys, 'check', str(path), '--replay')
    assert status == 1 and 'differs from the recording at step' in out
    status, out, _ = run(capsys, 'check', str(path), '--replay', '--json')
    answer = json.loads(out)
    assert status == 1 and 'differs from the recording' in answer['difference']


def test_a_file_that_is_no_recording_exits_1_with_one_error_line(files, capsys, tmp_path):
    bad = tmp_path / 'bad.cjr'
    bad.write_bytes(b'not a recording')
    for command in ('show', 'check'):
        status, out, err = run(capsys, command, str(bad))
        assert status == 1 and out == ''
        lines = [line for line in err.splitlines() if ': error: ' in line]
        assert len(lines) == 1 and lines[0].startswith(f'cuju recording {command}: error: ')
        status, out, _ = run(capsys, command, str(bad), '--json')
        answer = json.loads(out)
        assert status == 1 and answer['schema'] == 'error/1' and answer['exit'] == 1
        assert answer['error_id'] == 'recording.unreadable'
    status, _, err = run(capsys, 'show', str(tmp_path / 'missing.cjr'))
    assert status == 1 and 'no such file' in err


def test_check_refuses_members_that_the_format_does_not_name(files, capsys, tmp_path):
    path = tmp_path / 'extra.cjr'
    path.write_bytes(files['recording'].read_bytes())
    member = io.BytesIO()
    np.lib.format.write_array(member, np.zeros(2))
    with zipfile.ZipFile(path, 'a') as archive:
        archive.writestr('extra.npy', member.getvalue())
    assert cr.read(path).steps == files['match'].steps
    status, _, err = run(capsys, 'check', str(path))
    assert status == 1 and 'found extra' in err


def test_convert_writes_recordings_for_dumps_and_episodes(files, capsys, tmp_path):
    out_folder = tmp_path / 'recordings'
    status, out, err = run(
        capsys, 'convert', files['dump'], str(files['kaggle']), '-o', str(out_folder)
    )
    assert status == 0 and err == ''
    names = sorted(path.name for path in out_folder.iterdir())
    assert names == ['episode.cjr', os.path.basename(files['dump']).replace('.dump', '.cjr')]
    dump_file = next(path for path in out_folder.iterdir() if path.name != 'episode.cjr')
    converted = cr.read(dump_file)
    assert converted.end == 1 and converted.state is not None
    converted.replay()
    assert cr.read(out_folder / 'episode.cjr').state is None
    status, out, err = run(capsys, 'convert', files['dump'], '-o', str(out_folder))
    assert status == 1 and 'pass --overwrite' in err
    status, out, _ = run(
        capsys, 'convert', files['dump'], '-o', str(out_folder), '--overwrite', '--json'
    )
    answer = json.loads(out)
    assert status == 0 and answer['schema'] == 'recording.convert/1' and answer['failed'] == []
    assert answer['converted'][0]['kind'] == 'grf-dump'
    assert answer['converted'][0]['steps'] == converted.steps


def test_convert_goes_on_after_a_file_that_fails(files, capsys, tmp_path):
    bad = tmp_path / 'bad.dump'
    bad.write_bytes(b'not a dump')
    missing = tmp_path / 'missing.dump'
    status, out, _ = run(
        capsys,
        'convert',
        str(bad),
        str(missing),
        str(files['kaggle']),
        '-o',
        str(tmp_path / 'out'),
        '--json',
    )
    answer = json.loads(out)
    assert status == 1
    assert [item['source'] for item in answer['failed']] == [str(bad), str(missing)]
    assert [item['kind'] for item in answer['converted']] == ['kaggle']
    assert (tmp_path / 'out' / 'episode.cjr').exists()


def test_a_bad_device_and_bad_arguments_are_usage_errors(files, capsys):
    status, out, err = run(capsys, 'check', str(files['recording']), '--replay', '--device', 'x')
    assert status == 2 and 'cuju recording check: error: ' in err
    for words in (
        [],
        ['no-such-command'],
        ['show'],
        ['show', 'a', 'b'],
        ['show', 'a', '--no-such-option'],
        ['convert', 'a'],
        ['check', 'a', '--device'],
    ):
        status, out, err = run(capsys, *words)
        assert status == 2 and out == '' and 'Traceback' not in err, words
        assert sum(': error: ' in line for line in err.splitlines()) == 1, words
    status, out, err = run(capsys, 'show', '--json', '--no-such-option')
    assert status == 2 and json.loads(out)['schema'] == 'error/1' and ': error: ' in err
