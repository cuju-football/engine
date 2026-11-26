# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks `agents.check` and the commands of cuju agents."""

import json
import sys

from cuju import __main__ as cli
from cuju import agents
from cuju.agents._process import source

GOOD = 'def agent(obs):\n    return [0]\n'
SLOW = 'import time\n\ndef agent(obs):\n    time.sleep(0.6)\n    return [0]\n'
BROKEN = 'def agent(obs):\n    return [99]\n'
RANDOM = 'import os\n\ndef agent(obs):\n    return [os.urandom(1)[0] % 19]\n'


def folder(tmp_path, text, name='agent'):
    path = tmp_path / name
    path.mkdir()
    (path / 'main.py').write_text(text)
    return path


def run(capsys, *words):
    status = cli.main(['agents', *words])
    out, err = capsys.readouterr()
    return status, out, err


def test_check_passes_a_good_controller_and_reports_timing(tmp_path):
    report = agents.check(folder(tmp_path, GOOD), '5_vs_5', steps=5)
    assert report.ok and set(report.actions) == {'left', 'right'}
    assert {row['side'] for row in report.timing} == {'left', 'right'}
    assert all(row['calls'] == 5 for row in report.timing)


def test_check_reports_a_breach_with_its_reason(tmp_path):
    report = agents.check(folder(tmp_path, BROKEN), '5_vs_5', steps=3)
    assert not report.ok
    assert {failure['reason'] for failure in report.failures} == {'IndexError'}


def test_check_finds_nondeterminism(tmp_path):
    report = agents.check(folder(tmp_path, RANDOM, 'a'), '5_vs_5', steps=20, twice=True)
    assert not report.ok and report.failures[0]['reason'] == 'nondeterministic'
    report = agents.check(folder(tmp_path, GOOD, 'b'), '5_vs_5', steps=20, twice=True)
    assert report.ok


def test_check_in_a_process_and_through_a_command(tmp_path):
    assert agents.check(folder(tmp_path, GOOD), '5_vs_5', steps=3, process=True).ok
    command = [sys.executable, '-X', 'utf8', '-c', '{stub}']
    assert agents.check(None, '5_vs_5', steps=3, command=command).ok
    assert not agents.check(None, '5_vs_5', steps=3, command=[sys.executable, '-c', 'pass']).ok


def test_the_commands(tmp_path, capsys):
    path = str(folder(tmp_path, GOOD))
    status, out, _ = run(capsys, 'check', path, '--scenario', '5_vs_5', '--steps', '3', '--json')
    found = json.loads(out)
    assert status == 0 and found['schema'] == 'agents.check/1' and found['ok']
    status, out, _ = run(
        capsys,
        'check',
        str(folder(tmp_path, BROKEN, 'bad')),
        '--scenario',
        '5_vs_5',
        '--steps',
        '3',
    )
    assert status == 1 and 'does not play' in out
    status, out, _ = run(
        capsys, 'play', path, 'builtin', '--scenario', '5_vs_5', '--episodes', '1', '--json'
    )
    assert status == 0 and json.loads(out)['schema'] == 'agents.play/1'
    status, out, _ = run(capsys, 'parts', '--json')
    assert status == 0 and {row['name'] for row in json.loads(out)['parts']} >= {
        'kaggle',
        'process',
    }
    target = tmp_path / 'client.py'
    assert run(capsys, 'client', '-o', str(target))[0] == 0
    assert target.read_bytes() == source()
    status, _, err = run(capsys, 'check', str(tmp_path / 'missing'), '--scenario', '5_vs_5')
    assert status == 2 and 'error' in err
