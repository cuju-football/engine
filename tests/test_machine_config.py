# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks the machine config, the file that holds the settings that change only speed."""

import json
import logging
import os

import pytest
import torch

import cuju
from cuju import _config, errors
from cuju._batch import Batch


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """Returns a function that writes the machine config and points `CUJU_CONFIG` at it."""
    file = tmp_path / 'engine.toml'
    monkeypatch.setenv('CUJU_CONFIG', str(file))

    def write(text):
        file.write_text(text, encoding='utf-8')
        return file

    return write


def test_a_missing_file_leaves_every_default(machine):
    assert _config.read() == {}
    assert _config.values() == _config.DEFAULTS
    assert _config.differing() == {}


def test_the_default_path_follows_the_platform(monkeypatch, tmp_path):
    monkeypatch.delenv('CUJU_CONFIG')
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    monkeypatch.setenv('APPDATA', str(tmp_path))
    path = _config.path().replace('\\', '/')
    assert path.endswith('/cuju/engine.toml')
    assert path.startswith(str(tmp_path).replace('\\', '/'))


def test_the_file_sets_the_keys_of_run(machine):
    machine(
        '# the machine\n[run]\nthreads = 3\ndebug = true\nallocator = "torch"\n'
        'pin_threads = true\nnuma = false\npin_memory = false\nrender_threads = 2\n'
        'render_workers = 4\nlog = true\n'
    )
    assert _config.values() == {
        'threads': 3,
        'debug': True,
        'allocator': 'torch',
        'pin_threads': True,
        'numa': False,
        'pin_memory': False,
        'render_threads': 2,
        'render_workers': 4,
        'log': True,
    }
    assert set(_config.differing()) == {
        'threads',
        'debug',
        'allocator',
        'pin_threads',
        'pin_memory',
        'render_threads',
        'render_workers',
        'log',
    }


@pytest.mark.parametrize(
    ('text', 'message'),
    [
        ('[other]\nthreads = 1\n', 'only the table'),
        ('threads = 1\n', 'only the table'),
        ('[run]\nspeed = 1\n', 'speed'),
        ('[run]\nthreads = -1\n', 'threads'),
        ('[run]\nthreads = "many"\n', 'threads'),
        ('[run]\ndebug = 1\n', 'debug'),
        ('[run]\nallocator = "malloc"\n', 'allocator'),
        ('[run]\npin_memory = "yes"\n', 'pin_memory'),
        ('[run]\nnuma = "a"\n', 'numa'),
        ('[run]\n[other]\n', 'only the table'),
        ('[run\n', 'TOML'),
    ],
)
def test_the_file_refuses_anything_but_run_and_its_keys(machine, text, message):
    machine(text)
    with pytest.raises(errors.ConfigurationError, match=message) as raised:
        _config.read()
    assert isinstance(raised.value, ValueError)
    with pytest.raises(errors.ConfigurationError):
        cuju.BatchEnvironment('5_vs_5', environment_count=1, device='cpu')


def test_a_call_overrides_the_file_and_says_so(machine):
    machine('[run]\nthreads = 2\ndebug = true\n')
    resolved = _config.resolve({'threads': 5, 'log': True})
    assert resolved['threads'] == {'value': 5, 'source': 'call'}
    assert resolved['log']['source'] == 'call'
    assert resolved['debug']['value'] is True and resolved['debug']['source'].startswith('file ')
    assert resolved['allocator'] == {'value': 'cuda', 'source': 'default'}
    with pytest.raises(errors.ConfigurationError, match='speed'):
        _config.resolve({'speed': 1})
    with pytest.raises(errors.ConfigurationError):
        _config.resolve({'threads': 'x'})


def test_a_batch_reports_the_settings_it_ran_with(machine):
    machine('[run]\nthreads = 2\n')
    with cuju.BatchEnvironment(
        '5_vs_5', environment_count=2, device='cpu', config={'pin_memory': False}
    ) as batch:
        assert batch.run_config['threads']['value'] == 2
        assert batch.run_config['threads']['source'].startswith('file ')
        assert batch.run_config['pin_memory'] == {'value': False, 'source': 'call'}
        assert batch._batch.threads == 2
    with pytest.raises(errors.ConfigurationError):
        cuju.BatchEnvironment('5_vs_5', environment_count=1, device='cpu', config={'x': 1})


def test_the_machine_options_resolve_pin_memory(machine):
    assert _config.machine_options({'pin_memory': True})['pin_memory'] is True
    assert _config.machine_options({'pin_memory': False})['pin_memory'] is False
    assert _config.machine_options()['pin_memory'] == torch.cuda.is_available()


def test_a_node_map_stands_in_for_the_numa_nodes(machine, monkeypatch):
    machine('[run]\nnuma = "0,0,1,1"\n')
    assert _config.values()['numa'] == '0,0,1,1'
    monkeypatch.delenv('CUJU_NUMA_NODES', raising=False)
    for threads in (1, 4):
        with Batch('5_vs_5', environment_count=4, device='cpu', threads=threads, numa='0,0,1,1'):
            pass
    assert 'CUJU_NUMA_NODES' not in os.environ


def test_the_log_key_logs_each_batch(machine, caplog):
    machine('[run]\nlog = true\n')
    with caplog.at_level(logging.INFO, logger='cuju'):
        with cuju.BatchEnvironment('5_vs_5', environment_count=1, device='cpu'):
            pass
    assert any('created 1 environments' in record.getMessage() for record in caplog.records)
    caplog.clear()
    machine('[run]\nlog = false\n')
    with caplog.at_level(logging.INFO, logger='cuju'):
        with cuju.BatchEnvironment('5_vs_5', environment_count=1, device='cpu'):
            pass
        with cuju.BatchEnvironment(
            '5_vs_5', environment_count=1, device='cpu', config={'log': True}
        ):
            pass
    assert sum('created 1 environments' in record.getMessage() for record in caplog.records) == 1


def test_the_render_keys_reach_the_renderers(machine):
    from cuju._batch import pixels

    assert pixels.split(8, cpus=12, workers=2, threads=3) == (2, 3)
    assert pixels.split(8, cpus=12, workers=0, threads=3) == (4, 3)
    assert pixels.split(8, cpus=12) == (8, 1)
    assert pixels.split(2, cpus=12) == (2, 6)


def test_the_check_prints_the_config_and_its_differences(machine, capsys):
    from cuju import check

    machine('[run]\nthreads = 3\ndebug = true\npin_memory = false\n')
    lines = check._config_lines()
    assert lines[0].startswith('machine config: ') and lines[0].endswith(', read')
    assert 'threads = 3' in lines[1] and 'debug = true' in lines[1]
    assert any('debug is on' in line for line in lines)
    machine('[run]\nthreads = 0\n')
    assert check._config_lines()[1] == '  every value equals the default'
    machine('[run]\nspeed = 1\n')
    assert 'refused' in check._config_lines()[0]
    machine('[run]\n')
    assert json.dumps(check._config_lines())
