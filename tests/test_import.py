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
        "import sys, cuju, cuju.state; print('torch' in sys.modules); "
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
        "print(missing or 'none'); from cuju.action_log import ActionLog, Digest, digest, replay; "
        "print(ActionLog.__name__, 'ActionLog' in dir(cuju))"
    )
    assert found == ['none', 'ActionLog', 'False']


def test_action_log_imports_first():
    found = run("import cuju.action_log; print('ok')")
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


@pytest.mark.skipif(importlib.util.find_spec('gymnasium') is None, reason='no gymnasium')
def test_gymnasium_imported_later_gets_the_ids():
    found = run(
        'import cuju, gymnasium; '
        "print('GFootball-11_vs_11_kaggle-simple115v2-v0' in gymnasium.envs.registry)"
    )
    assert found == ['True']


def test_only_a_portion_of_a_package_joins_the_package(tmp_path):
    stray = tmp_path / 'stray' / 'cuju'
    (stray / 'fakepkg').mkdir(parents=True)
    (stray / '__init__.py').write_text("raise SystemExit('the stray cuju was imported')\n")
    (stray / 'fakepkg' / '__init__.py').write_text('STRAY = True\n')
    portion = tmp_path / 'portion' / 'cuju'
    (portion / 'fakepkg').mkdir(parents=True)
    (portion / 'fakepkg' / '__init__.py').write_text('PORTION = True\n')
    home = os.path.dirname(os.path.dirname(importlib.util.find_spec('cuju').origin))
    code = (
        'import json, os, cuju\n'
        'path = list(cuju.__path__)\n'
        'import cuju.fakepkg\n'
        'stray = hasattr(cuju.fakepkg, "STRAY")\n'
        'print(json.dumps([path, os.path.dirname(cuju.__file__), stray]))\n'
    )
    environment = dict(os.environ)
    environment.pop('CUJU_BATCH_HOOKS', None)
    environment['PYTHONPATH'] = os.pathsep.join([home, str(tmp_path / 'stray')])
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path / 'portion',
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    path, own, stray_loaded = json.loads(result.stdout)
    normal = [os.path.normcase(folder) for folder in path]
    assert normal[0] == os.path.normcase(own)
    assert os.path.normcase(str(portion)) in normal
    assert os.path.normcase(str(stray)) not in normal
    assert all(os.path.isabs(folder) for folder in path)
    assert not stray_loaded
