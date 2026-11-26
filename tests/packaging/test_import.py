# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks what `import cuju` imports, in a fresh interpreter."""

import importlib.util
import json
import os
import subprocess
import sys

import pytest


def run(code):
    """Returns what `code` prints in a new interpreter that sees the same `cuju`."""
    result = subprocess.run(
        [sys.executable, '-c', code], capture_output=True, text=True, check=True, timeout=120
    )
    return result.stdout.split()


def test_import_leaves_gym_and_gymnasium_out():
    found = run(
        "import sys, cuju; print('gym' in sys.modules, 'gymnasium' in sys.modules, "
        "'cuju.env' in sys.modules)"
    )
    assert found == ['False', 'False', 'False']


def test_import_leaves_torch_out_until_a_batch_is_used():
    found = run(
        "import sys, cuju, cuju._state; print('torch' in sys.modules); "
        "print('BatchEnvironment' in dir(cuju), sorted(cuju.__all__) == sorted(set(cuju.__all__)))"
        '; from cuju import BatchEnvironment; '
        "print('torch' in sys.modules, BatchEnvironment.__module__, "
        'cuju.scenario_names.__module__)'
    )
    assert found == ['False', 'True', 'True', 'True', 'cuju', 'cuju']


def test_scenarios_and_the_library_path_leave_torch_out():
    found = run(
        'import sys, cuju.scenarios, cuju._library_path as path; '
        "path.library_path('cpu'); print('torch' in sys.modules)"
    )
    assert found == ['False']


def test_every_public_name_of_cuju_resolves():
    found = run(
        'import cuju; missing = [name for name in cuju.__all__ if not hasattr(cuju, name)]; '
        "print(missing or 'none'); from cuju._action_log import ActionLog, Digest, digest, replay; "
        "print(ActionLog.__name__, 'ActionLog' in dir(cuju))"
    )
    assert found == ['none', 'ActionLog', 'False']


def test_action_log_imports_first():
    found = run("import cuju._action_log; print('ok')")
    assert found == ['ok']


def test_import_of_gfootball_works_without_gym():
    found = run(
        "import sys; sys.modules['gym'] = sys.modules['gymnasium'] = None; "
        'import cuju.gfootball; '
        "print('cuju.gfootball.env' in sys.modules, cuju.gfootball._gym.VENDORED)"
    )
    assert found == ['True', 'True']


def test_import_of_gfootball_binds_env():
    # GRF: `import gfootball` imports `gfootball.env`, so the attribute serves at once.
    found = run(
        'import sys, cuju.gfootball; '
        "print('cuju.gfootball.env' in sys.modules, "
        'callable(cuju.gfootball.env.create_environment))'
    )
    assert found == ['True', 'True']


def test_import_does_nothing_beyond_defining_names():
    found = run(
        'import sys, time; before = list(sys.meta_path); start = time.perf_counter(); import cuju; '
        'spent = time.perf_counter() - start; '
        "print(sys.meta_path == before, [name for name in ('torch', 'numpy', 'gymnasium', 'gym') "
        'if name in sys.modules], spent < 0.25)'
    )
    assert found == ['True', '[]', 'True']


@pytest.mark.skipif(importlib.util.find_spec('gymnasium') is None, reason='no gymnasium')
@pytest.mark.parametrize('order', ['import gymnasium, cuju.env', 'import cuju.env, gymnasium'])
def test_gymnasium_gets_the_ids_in_either_import_order(order):
    found = run(
        f'{order}; print(gymnasium.make("GFootball-academy_empty_goal_close-simple115v2-v0")'
        '.spec.id)'
    )
    assert found == ['GFootball-academy_empty_goal_close-simple115v2-v0']


@pytest.mark.skipif(importlib.util.find_spec('gymnasium') is None, reason='no gymnasium')
def test_gymnasium_makes_an_id_by_the_module_that_registers_it():
    found = run(
        'import gymnasium; '
        'print(gymnasium.make("cuju.env:GFootball-5_vs_5-simple115v2-v0").spec.id)'
    )
    assert found == ['GFootball-5_vs_5-simple115v2-v0']


def test_the_ids_are_declared_as_a_gymnasium_entry_point():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(root, 'pyproject.toml')
    if not os.path.isfile(path):
        pytest.skip('no source tree')
    text = open(path, encoding='utf-8').read()
    assert '[project.entry-points."gymnasium.envs"]' in text
    assert '__root__ = "cuju.env:register_environments"' in text


def test_a_portion_of_the_package_on_the_path_joins_it(tmp_path):
    portion = tmp_path / 'portion' / 'cuju'
    (portion / 'fakepkg').mkdir(parents=True)
    (portion / 'fakepkg' / '__init__.py').write_text('PORTION = True\n')
    home = os.path.dirname(os.path.dirname(importlib.util.find_spec('cuju').origin))
    code = (
        'import json, os, cuju\n'
        'import cuju.fakepkg\n'
        'print(json.dumps([list(cuju.__path__), os.path.dirname(cuju.__file__), '
        'cuju.fakepkg.PORTION]))\n'
    )
    environment = dict(os.environ)
    environment.pop('CUJU_BATCH_HOOKS', None)
    environment['PYTHONPATH'] = os.pathsep.join([home, str(tmp_path / 'portion')])
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    path, own, found = json.loads(result.stdout)
    normal = [os.path.normcase(folder) for folder in path]
    assert normal[0] == os.path.normcase(own)
    assert os.path.normcase(str(portion)) in normal
    assert found
