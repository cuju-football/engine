# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks that `cuju.gfootball` is GRF's Python package with its import lines changed, and its gym.

`ci/port_gfootball.py` writes the tree, `GRF_SOURCES` and `GYM_SOURCES`. These tests remove the
lines it adds, undo its rewrites with its own tables and compare the digests, so the claim that the
port is GRF's code, and that its vendored gym is gym's, stays checked without either source.
"""

import ast
import hashlib
import importlib.util
import os
import subprocess
import sys

import pytest

import cuju

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE = os.path.join(os.path.dirname(cuju.__file__), 'gfootball')
SHIPPED_SCENARIOS = os.path.join(os.path.dirname(cuju.__file__), 'data', 'scenarios')
# The folder that holds `cuju`, for the interpreters that the tests start.
HOME = os.path.dirname(os.path.dirname(os.path.abspath(cuju.__file__)))


def _load_generator():
    path = os.path.join(ROOT, 'ci', 'port_gfootball.py')
    spec = importlib.util.spec_from_file_location('port_gfootball', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


port = _load_generator()
GRF_DIGESTS = port.read_sources(os.path.join(PACKAGE, port.GRF_SOURCES))
GYM_DIGESTS = port.read_sources(os.path.join(PACKAGE, *port.GYM_SOURCES.split('/')))


def _read(*parts):
    with open(os.path.join(*parts), 'rb') as handle:
        return handle.read().decode('utf-8')


def _run(code, path=()):
    """Returns what `code` prints in a new interpreter, with the folders `path` ahead of `cuju`."""
    environment = dict(os.environ)
    inherited = [entry for entry in [os.environ.get('PYTHONPATH')] if entry]
    environment['PYTHONPATH'] = os.pathsep.join([*map(str, path), HOME, *inherited])
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
        timeout=300,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.split()


@pytest.mark.parametrize('path', sorted(GRF_DIGESTS))
def test_each_ported_file_is_grf_s_file_with_its_imports_changed(path):
    ported = _read(PACKAGE, *path.split('/'))
    original = port.unport(ported, path, port.GRF_REWRITES, 'grf')
    assert hashlib.sha256(original.encode('utf-8')).hexdigest() == GRF_DIGESTS[path]
    # The port is the image of GRF's file, so it holds no change that the inverse would undo.
    assert port.port(original, path, port.GRF_REWRITES, 'grf') == ported


@pytest.mark.parametrize('path', sorted(GYM_DIGESTS))
def test_each_vendored_file_is_gym_s_file_with_its_imports_changed(path):
    ported = _read(PACKAGE, *port.VENDORED.split('/'), *path.split('/'))
    original = port.unport(ported, path, port.GYM_REWRITES, 'gym')
    assert hashlib.sha256(original.encode('utf-8')).hexdigest() == GYM_DIGESTS[path]
    assert port.port(original, path, port.GYM_REWRITES, 'gym') == ported


def test_the_tree_holds_the_listed_files_alone():
    found = set()
    for folder, folders, names in os.walk(PACKAGE):
        folders[:] = [name for name in folders if name != '__pycache__']
        for name in names:
            relative = os.path.relpath(os.path.join(folder, name), PACKAGE)
            found.add(relative.replace(os.sep, '/'))
    expected = {
        *GRF_DIGESTS,
        *(f'{port.VENDORED}/{path}' for path in GYM_DIGESTS),
        *port.OWN_FILES,
        port.GRF_SOURCES,
        port.GYM_SOURCES,
    }
    assert found == expected
    assert sorted(GYM_DIGESTS) == sorted(port.GYM_FILES)


def _code(path):
    """Returns the statements of the module at `path`, without its docstring and its imports."""
    tree = ast.parse(_read(path))
    body = tree.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return [ast.dump(node) for node in body if not isinstance(node, (ast.Import, ast.ImportFrom))]


def test_the_shipped_scenarios_are_grf_s_scenarios():
    # The shipped copies carry cuju's header and format, so their code is compared, constant by
    # constant and statement by statement, with GRF's files in the port.
    grf = sorted(
        path.removeprefix('scenarios/').removesuffix('.py')
        for path in GRF_DIGESTS
        if path.startswith('scenarios/')
        and path.count('/') == 1
        and path != 'scenarios/__init__.py'
    )
    shipped = sorted(
        name.removesuffix('.py') for name in os.listdir(SHIPPED_SCENARIOS) if name.endswith('.py')
    )
    assert len(grf) == 19
    assert shipped == grf
    for name in grf:
        ported = _code(os.path.join(PACKAGE, 'scenarios', f'{name}.py'))
        assert ported == _code(os.path.join(SHIPPED_SCENARIOS, f'{name}.py')), name


def test_import_of_cuju_leaves_gym_and_absl_out():
    found = _run("import sys, cuju; print('gym' in sys.modules, 'absl' in sys.modules)")
    assert found == ['False', 'False']


def test_without_gym_the_vendored_core_serves():
    # A `None` in `sys.modules` makes the import fail, as an installation without gym does.
    found = _run(
        "import sys; sys.modules['gym'] = None\n"
        'from cuju.gfootball import _gym as gym\n'
        'print(gym.VENDORED, gym.__version__, gym.Env.__module__)\n'
        "spec = gym.spec('GFootball-academy_corner-SMM-v0')\n"
        "print(spec.entry_point, spec._kwargs['representation'])\n"
    )
    assert found == [
        'True',
        '0.21.0',
        'cuju.gfootball._gym._vendored.core',
        'cuju.gfootball.env:create_environment',
        'extracted',
    ]


def _fake_gym(root, version, source):
    """Writes a package `gym` of `version` under `root`, with `source` as its `__init__.py`."""
    package = root / 'gym'
    (package / 'envs').mkdir(parents=True)
    (package / '__init__.py').write_text(source, encoding='utf-8')
    (package / 'envs' / '__init__.py').write_text('', encoding='utf-8')
    (package / 'envs' / 'registration.py').write_text(
        'registry = {}\n\n\ndef register(id, **kwargs):\n    registry[id] = kwargs\n',
        encoding='utf-8',
    )
    information = root / f'gym-{version}.dist-info'
    information.mkdir()
    (information / 'METADATA').write_text(
        f'Metadata-Version: 2.1\nName: gym\nVersion: {version}\n', encoding='utf-8'
    )


_GYM_021 = """
import types
from gym import envs
from gym.envs.registration import register

__version__ = '0.21.0'


class Env:
    pass


Wrapper = ObservationWrapper = ActionWrapper = RewardWrapper = Space = Env
spaces = wrappers = logger = error = types.SimpleNamespace()
make = spec = register
"""


def test_gym_0_21_serves_as_itself(tmp_path):
    _fake_gym(tmp_path, '0.21.0', _GYM_021)
    found = _run(
        'import gym\n'
        'from cuju.gfootball import _gym\n'
        'print(_gym.VENDORED, _gym.Env is gym.Env, _gym.register is gym.register)\n'
        "print(gym.envs.registration.registry['GFootball-academy_corner-SMM-v0']['entry_point'])\n",
        path=[tmp_path],
    )
    assert found == ['False', 'True', 'True', 'cuju.gfootball.env:create_environment']


def _fake_numpy_version(root, version):
    """Writes the metadata of a numpy of `version` under `root`, which lookups find first."""
    information = root / f'numpy-{version}.dist-info'
    information.mkdir()
    (information / 'METADATA').write_text(
        f'Metadata-Version: 2.1\nName: numpy\nVersion: {version}\n', encoding='utf-8'
    )


@pytest.mark.parametrize('version', ['0.22.0', '0.23.1', '0.24.1'])
def test_a_gym_that_grf_runs_on_serves_as_itself(tmp_path, version):
    # GRF: `import gfootball` registers its ids with every gym up to 0.24, which run it.
    _fake_gym(tmp_path, version, _GYM_021)
    found = _run(
        'import gym\nfrom cuju.gfootball import _gym\nprint(_gym.VENDORED, _gym.Env is gym.Env)\n',
        path=[tmp_path],
    )
    assert found == ['False', 'True']


@pytest.mark.parametrize(('numpy', 'vendored'), [('1.26.4', 'False'), ('2.5.4', 'True')])
def test_gym_0_25_serves_as_itself_with_numpy_1(tmp_path, numpy, vendored):
    # GRF: gym 0.25's step checker reads `np.bool8`, which numpy 2 removed.
    _fake_gym(tmp_path, '0.25.2', _GYM_021)
    _fake_numpy_version(tmp_path, numpy)
    found = _run('from cuju.gfootball import _gym; print(_gym.VENDORED)', path=[tmp_path])
    assert found == [vendored]


def test_gym_0_26_leaves_the_vendored_core(tmp_path):
    # GRF: gym 0.26 takes gymnasium's interface, and GRF's `reset` fails there.
    _fake_gym(tmp_path, '0.26.2', "raise SystemExit('a later gym was imported')\n")
    found = _run(
        "import sys\nfrom cuju.gfootball import _gym\nprint(_gym.VENDORED, 'gym' in sys.modules)\n",
        path=[tmp_path],
    )
    assert found == ['True', 'False']


def test_a_gym_0_21_that_fails_to_import_leaves_the_vendored_core(tmp_path):
    _fake_gym(tmp_path, '0.21.0', "raise ImportError('a broken gym')\n")
    found = _run('from cuju.gfootball import _gym; print(_gym.VENDORED)', path=[tmp_path])
    assert found == ['True']


@pytest.mark.skipif(importlib.util.find_spec('gym') is None, reason='no gym')
def test_the_installed_gym_serves_by_its_version():
    found = _run(
        'import importlib.metadata, re\n'
        'def version(name):\n'
        "    found = re.match(r'(\\d+)\\.(\\d+)', importlib.metadata.version(name))\n"
        '    return tuple(map(int, found.groups()))\n'
        "gym, numpy = version('gym'), version('numpy')\n"
        'runs = gym <= (0, 24) or (gym == (0, 25) and numpy < (2, 0))\n'
        'from cuju.gfootball import _gym\n'
        'print(_gym.VENDORED != runs)\n'
    )
    assert found == ['True']


def test_the_vendored_gym_runs_spaces_wrappers_and_make():
    found = _run(
        "import sys; sys.modules['gym'] = None\n"
        'import numpy as np\n'
        'from cuju.gfootball import _gym as gym\n'
        'box = gym.spaces.Box(low=0, high=255, shape=(72, 96, 4), dtype=np.uint8)\n'
        'box.seed(0)\n'
        'print(box.contains(box.sample()), gym.spaces.Discrete(19).n)\n'
        'print(gym.spaces.MultiDiscrete([19] * 3).sample().shape == (3,))\n'
        'class Constant(gym.Env):\n'
        '    observation_space = action_space = gym.spaces.Discrete(2)\n'
        '    def reset(self):\n'
        '        return 0\n'
        '    def step(self, action):\n'
        '        return 1, 1.0, True, {}\n'
        "gym.register(id='Constant-v0', entry_point=Constant, max_episode_steps=5)\n"
        "env = gym.make('Constant-v0')\n"
        'print(type(env).__name__, env.reset(), env.step(0)[1])\n'
        'print(type(gym.Wrapper(Constant()).unwrapped).__name__)\n'
    )
    assert found == ['True', '19', 'True', 'TimeLimit', '0', '1.0', 'Constant']


def test_env_imports_without_opencv_and_raises_at_its_first_use():
    found = _run(
        "import sys; sys.modules['cv2'] = None\n"
        'import cuju.gfootball.env\n'
        'from cuju.gfootball import _cv2\n'
        "print(hasattr(_cv2, '__path__'))\n"
        'try:\n'
        '    _cv2.resize\n'
        'except ModuleNotFoundError as error:\n'
        '    print(error.name)\n'
    )
    assert found == ['False', 'cv2']


def test_the_levels_of_scenarios_and_their_tests_build():
    # GRF: a level names a module of `gfootball.scenarios`, such as `tests.symmetric`.
    found = _run(
        'from cuju.gfootball.env import config\n'
        "for level in ('tests.symmetric', '11_vs_11_stochastic', 'academy_corner'):\n"
        '    scenario = config.Config({"level": level}).ScenarioConfig()\n'
        '    print(len(scenario.left_team), len(scenario.right_team))\n'
    )
    assert found == ['11', '11', '11', '11', '11', '11']
