# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks cuju-gfootball, which imports cuju's port under GRF's names, each in a new interpreter."""

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import textwrap
import venv
import zipfile

import pytest

import cuju

pytest.importorskip('absl')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT = os.path.join(ROOT, 'packages', 'cuju-gfootball')
SOURCE = os.path.join(PROJECT, 'src')
PACKAGE = os.path.join(os.path.dirname(cuju.__file__), 'gfootball')
HOME = os.path.dirname(os.path.dirname(os.path.abspath(cuju.__file__)))
SCENARIO = 'academy_empty_goal_close'

if not os.path.isdir(SOURCE):
    pytest.skip(
        'needs the source tree, which holds packages/cuju-gfootball', allow_module_level=True
    )


def _run(code, *, alias=True, path=(), check=True, timeout=600, arguments=(), temporary=None):
    prelude = f'import site\nsite.addsitedir({SOURCE!r})\n' if alias else ''
    environment = dict(os.environ)
    inherited = [entry for entry in [os.environ.get('PYTHONPATH')] if entry]
    environment['PYTHONPATH'] = os.pathsep.join([*map(str, path), HOME, *inherited])
    environment['UNITTEST_IN_DOCKER'] = '1'
    if temporary is not None:
        # GRF's tests keep their dumps in the temporary folder, which stays apart from other runs.
        environment['TMPDIR'] = str(temporary)
    result = subprocess.run(
        [sys.executable, *arguments, '-c', prelude + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=environment,
    )
    if check:
        assert result.returncode == 0, result.stderr
    return result


def _json(code, **options):
    return json.loads(_run(code, **options).stdout.strip().splitlines()[-1])


def _installed():
    return _json(
        'import importlib.util, json; print(json.dumps(bool('
        "importlib.util.find_spec('cuju.gfootball_alias'))))",
        alias=False,
    )


def test_names_of_grf_are_the_modules_of_cuju():
    found = _json("""
        import importlib.util, json
        import cuju.gfootball_alias
        import gfootball, gfootball.env.football_env
        import gfootball_engine, _gameplayfootball
        import cuju.gfootball, cuju.gfootball.env.football_env, cuju.gfootball_engine
        from gfootball.env import config
        from gfootball.scenarios import *
        print(json.dumps([
            cuju.gfootball_alias.active(),
            gfootball is cuju.gfootball,
            gfootball.env.football_env is cuju.gfootball.env.football_env,
            config is cuju.gfootball.env.config,
            gfootball_engine is cuju.gfootball_engine,
            _gameplayfootball is cuju.gfootball_engine._gameplayfootball,
            importlib.util.find_spec('gfootball') is not None,
            importlib.util.find_spec('gfootball.env.wrappers') is not None,
            gfootball.__spec__.name,
            'Team' in dir() and 'e_PlayerRole_GK' in dir(),
        ]))
    """)
    assert found == [True, True, True, True, True, True, True, True, 'cuju.gfootball', True]


def test_a_name_that_neither_holds_stays_missing():
    result = _run('import gfootball.env.no_such_module', check=False)
    assert "No module named 'gfootball.env.no_such_module'" in result.stderr


def test_classes_carry_the_module_names_of_grf():
    found = _json("""
        import json
        from gfootball.env import football_action_set, football_env, football_env_core
        import _gameplayfootball
        print(json.dumps([
            football_action_set.CoreAction.__module__,
            football_env_core.EnvState.__module__,
            football_env.FootballEnv.__module__,
            _gameplayfootball.e_BackendAction.__module__,
            _gameplayfootball.GameEnv.__module__,
            repr(_gameplayfootball.e_BackendAction.left),
        ]))
    """)
    assert found == [
        'gfootball.env.football_action_set',
        'gfootball.env.football_env_core',
        'gfootball.env.football_env',
        '_gameplayfootball',
        '_gameplayfootball',
        '_gameplayfootball.e_BackendAction.left',
    ]


def test_cuju_keeps_its_own_names_without_the_alias():
    if _installed():
        pytest.skip('cuju-gfootball is installed, so every interpreter here has the alias')
    found = _json(
        """
        import importlib.util, json
        from cuju.gfootball.env import football_action_set
        print(json.dumps([football_action_set.CoreAction.__module__,
                          importlib.util.find_spec('gfootball') is None]))
    """,
        alias=False,
    )
    assert found == ['cuju.gfootball.env.football_action_set', True]


def test_a_port_module_loaded_first_is_renamed_on_activation():
    found = _json(
        """
        import json
        from cuju.gfootball.env import football_action_set
        import cuju.gfootball_alias
        import gfootball.env.football_action_set as grf
        print(json.dumps([football_action_set.CoreAction.__module__,
                          grf is football_action_set]))
    """,
        alias=False,
        arguments=('-S',),
        path=(SOURCE, *_site_packages()),
    )
    assert found == ['gfootball.env.football_action_set', True]


def _site_packages():
    return [entry for entry in sys.path if entry and os.path.isdir(entry)]


_NAMES = """
    import json, pickle

    def names(path):
        found = set()

        class Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                found.add(f'{module}.{name}')
                return super().find_class(module, name)

        with open(path, 'rb') as handle:
            while handle.peek(1):
                Unpickler(handle).load()
        return sorted(name for name in found if not name.startswith(('numpy', 'builtins')))
"""


def _needs_torch():
    found = _json(
        'import importlib.util; print(int(importlib.util.find_spec("torch") is not None))'
    )
    if not found:
        pytest.skip('needs PyTorch, which cuju.env and cuju.dumps run on')


def _write_dump(tmp_path, module):
    code = f"""
        import glob, json
        import {module} as football
        env = football.create_environment({SCENARIO!r}, logdir={str(tmp_path)!r})
        env.reset()
        for step in range(30):
            env.step(step % 19)
        env.write_dump('alias')
        env.close()
        print(json.dumps(glob.glob({str(tmp_path / 'alias_*.dump')!r})))
    """
    found = _json(code, alias=module.startswith('gfootball'))
    assert len(found) == 1
    return found[0]


def test_dumps_through_the_alias_name_grf_and_open_in_every_reader(tmp_path):
    _needs_torch()
    path = _write_dump(tmp_path, 'gfootball.env')
    found = _json(
        textwrap.dedent(_NAMES)
        + textwrap.dedent(f"""
        import cuju.dumps
        from gfootball.env import script_helpers
        steps = script_helpers.ScriptHelpers().load_dump({path!r})
        print(json.dumps([names({path!r}), len(steps), len(cuju.dumps.read_dump({path!r}))]))
    """)
    )
    assert found[0] == [
        '_gameplayfootball.e_BackendAction',
        'gfootball.env.football_action_set.CoreAction',
    ]
    assert found[1] == found[2] > 0


def test_dumps_that_name_grf_open_in_the_port_through_the_alias(tmp_path):
    _needs_torch()
    path = _write_dump(tmp_path, 'cuju.env')
    code = textwrap.dedent(_NAMES) + textwrap.dedent(f"""
        from cuju.gfootball.env import script_helpers
        steps = script_helpers.ScriptHelpers().load_dump({path!r})
        script_helpers.ScriptHelpers().dump_to_txt({path!r}, {str(tmp_path / 'dump.txt')!r}, False)
        print(json.dumps([names({path!r}), len(steps)]))
    """)
    found = _json(code)
    assert found[0] == [
        '_gameplayfootball.e_BackendAction',
        'gfootball.env.football_action_set.CoreAction',
    ]
    assert found[1] > 0
    assert (tmp_path / 'dump.txt').stat().st_size > 0
    if not _installed():
        result = _run(code, alias=False, check=False)
        assert "No module named 'gfootball'" in result.stderr


@pytest.mark.parametrize('video', [False, True], ids=['dump', 'video'])
def test_kaggles_calls_into_the_trace_write_dumps_and_videos(tmp_path, video):
    if video and not _json(
        'import importlib.util; print(int(bool(importlib.util.find_spec("cv2"))))'
    ):
        pytest.skip('a video needs OpenCV, from the extra gfootball-video')
    found = _json(f"""
        import json, os
        import gfootball.env as football
        env = football.create_environment(
            {SCENARIO!r}, stacked=False, representation='raw', logdir={str(tmp_path)!r},
            write_goal_dumps=False, write_full_episode_dumps=True, write_video={video!r},
            render=False, number_of_left_players_agent_controls=1,
            number_of_right_players_agent_controls=0,
            other_config_options={{'video_format': 'webm'}})
        env.reset()
        for _ in range(10):
            env.step([0])
        internal = env
        while not hasattr(internal, '_env'):
            internal = internal.env
        trace = internal._env._trace
        trace._dump_config['episode_done']._min_frequency = 0
        dumps = trace.process_pending_dumps(True)
        print(json.dumps([internal._env._step, [
            [entry['name'], sorted(key for key in entry if key != 'name'),
             all(os.path.getsize(entry[key]) > 0 for key in entry if key != 'name')]
            for entry in dumps]]))
    """)
    keys = ['dump', 'video'] if video else ['dump']
    assert found == [9, [['episode_done', keys, True]]]


# GRF's own tests, unchanged.


def _load_script(name):
    folder = 'make' if name == 'port_gfootball' else 'checks'
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(ROOT, 'ci', folder, f'{name}.py')
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


import test_grf_suite  # noqa: E402


@pytest.fixture(scope='module')
def grf_suite(tmp_path_factory):
    """Runs GRF's test modules, as GRF ships them, in one interpreter, and returns each result."""
    port = _load_script('port_gfootball')
    digests = port.read_sources(os.path.join(PACKAGE, port.GRF_SOURCES))
    originals = tmp_path_factory.mktemp('grf_tests')
    for name in test_grf_suite.MODULES:
        path = f'env/{name}.py'
        with open(os.path.join(PACKAGE, 'env', f'{name}.py'), encoding='utf-8') as handle:
            text = port.unport(handle.read(), path, port.GRF_REWRITES, 'grf')
        assert hashlib.sha256(text.encode('utf-8')).hexdigest() == digests[path]
        with open(originals / f'{name}.py', 'w', encoding='utf-8', newline='') as handle:
            handle.write(text)
    output = originals / 'results.json'
    code = f"""
        import importlib, importlib.util, json, os, sys, unittest
        from absl import flags
        from cuju.gfootball import _gym
        from test_grf_suite import _run_case
        sys.path.insert(0, {str(originals)!r})
        flags.FLAGS.mark_as_parsed()
        results = {{}}
        unsupported = _gym.VENDORED and importlib.util.find_spec('gym') is not None
        for name in {list(test_grf_suite.MODULES)!r}:
            if unsupported:
                import gym
                reason = f'gym {{gym.__version__}} is outside the gyms that GRF runs on'
                results[name] = ['skip', reason]
                continue
            try:
                module = importlib.import_module(name)
            except ImportError as error:
                results[name] = ['skip', str(error)]
                continue
            if hasattr(module, 'fast_run'):
                module.fast_run = {not test_grf_suite.FULL!r}
            stack = [unittest.defaultTestLoader.loadTestsFromModule(module)]
            while stack:
                item = stack.pop()
                if isinstance(item, unittest.TestSuite):
                    stack.extend(reversed(list(item)))
                    continue
                test = f'{{type(item).__name__}}.{{item._testMethodName}}'
                result = _run_case(item, test)
                failures = result.errors + result.failures
                if result.skipped:
                    results[test] = ['skip', result.skipped[0][1]]
                else:
                    results[test] = ['fail' if failures else 'pass',
                                     '\\n'.join(text for _, text in failures)]
        with open({str(output)!r}, 'w') as handle:
            json.dump(results, handle)
    """
    temporary = originals / 'tmp'
    temporary.mkdir()
    _run(
        code,
        path=(os.path.dirname(os.path.abspath(__file__)),),
        timeout=7200 if test_grf_suite.FULL else 3600,
        temporary=temporary,
    )
    with open(output) as handle:
        return json.load(handle)


@pytest.mark.parametrize(('module_name', 'test'), test_grf_suite._cases())
def test_grf_suite_through_gfootball(grf_suite, module_name, test):
    outcome = grf_suite.get(module_name) or grf_suite[test]
    if outcome[0] == 'skip':
        pytest.skip(outcome[1])
    if test in _failures_of_the_served_gym():
        assert outcome[0] == 'fail', f'{test} passes, where GRF fails it under this gym'
        pytest.xfail(f'GRF fails {test} under this gym as well')
    assert outcome[0] == 'pass', outcome[1]


def _failures_of_the_served_gym():
    return tuple(
        _json(
            'import json, test_grf_suite; print(json.dumps(list(test_grf_suite._gym_failures())))',
            alias=False,
            path=(os.path.dirname(os.path.abspath(__file__)),),
        )
    )


def test_gym_makes_the_ids_of_grf():
    found = _json(f"""
        import json
        from cuju.gfootball import _gym
        if _gym.VENDORED:
            gym = _gym
        else:
            import gym
        made = []
        import gfootball
        for name in ('gfootball:GFootball-{SCENARIO}-SMM-v0',
                     'GFootball-{SCENARIO}-simple115v2-v0'):
            try:
                env = gym.make(name)
            except Exception as error:
                made.append('Error' if isinstance(error, gym.error.Error) else repr(error))
                continue
            env.reset()
            env.step(env.action_space.sample())
            env.close()
            made.append(list(env.observation_space.shape))
        print(json.dumps([_gym.VENDORED, gym.__version__, made]))
    """)
    vendored, version, made = found
    assert made[1] == [115]
    found_version = tuple(int(part) for part in re.findall(r'\d+', version)[:2])
    if not vendored and found_version in ((0, 22), (0, 23)):
        assert made[0] == 'Error'
    else:
        assert made[0] == [72, 96, 4]


def _grf_package(folder):
    package = folder / 'gfootball'
    package.mkdir()
    (package / '__init__.py').write_text('GRF = True\n')
    return folder


_REFUSED = """
    import json, warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        import gfootball
    import cuju.gfootball_alias
    from cuju.gfootball.env import football_action_set
    try:
        import gfootball_engine
        engine = 'imported'
    except ImportError:
        engine = 'missing'
    print(json.dumps([cuju.gfootball_alias.active(), cuju.gfootball_alias.status(),
                      getattr(gfootball, 'GRF', False),
                      [str(item.message) for item in caught if item.category is RuntimeWarning],
                      football_action_set.CoreAction.__module__, engine]))
"""


def test_the_alias_stays_out_beside_a_package_of_grf(tmp_path):
    folder = _grf_package(tmp_path)
    active, status, grf, warned, module, engine = _json(_REFUSED, path=(folder,))
    assert not active
    assert str(folder / 'gfootball') in status
    assert grf is True
    assert len(warned) == 1
    assert 'cuju-gfootball is inactive' in warned[0]
    assert str(folder / 'gfootball') in warned[0]
    assert module == 'cuju.gfootball.env.football_action_set'
    assert engine == 'missing'


def test_the_alias_stays_out_beside_the_distribution_of_grf(tmp_path):
    info = tmp_path / 'gfootball-2.10.3.dist-info'
    info.mkdir()
    (info / 'METADATA').write_text('Metadata-Version: 2.1\nName: gfootball\nVersion: 2.10.3\n')
    found = _json(
        """
        import json, warnings
        warnings.simplefilter('ignore')
        import cuju.gfootball_alias
        print(json.dumps([cuju.gfootball_alias.active(), cuju.gfootball_alias.status()]))
    """,
        path=(tmp_path,),
    )
    assert found == [
        False,
        "cuju-gfootball is inactive: GRF's distribution `gfootball` 2.10.3 is installed, which "
        'cuju-gfootball leaves in place.',
    ]


def test_the_alias_stays_out_beside_an_editable_grf(tmp_path):
    folder = _grf_package(tmp_path)
    found = _json(f"""
        import importlib.machinery, json, sys, warnings
        warnings.simplefilter('ignore')

        class Editable:
            def find_spec(self, name, path=None, target=None):
                if name == 'gfootball':
                    return importlib.machinery.PathFinder.find_spec(name, [{str(folder)!r}])

        sys.meta_path.append(Editable())
        import gfootball, cuju.gfootball_alias
        print(json.dumps([cuju.gfootball_alias.active(), getattr(gfootball, 'GRF', False)]))
    """)
    assert found == [False, True]


def test_the_command_says_why(tmp_path):
    active = _run(
        'import runpy; runpy.run_module("cuju.gfootball_alias", run_name="__main__")', check=False
    )
    assert active.returncode == 0
    assert 'cuju-gfootball is active' in active.stdout
    folder = _grf_package(tmp_path)
    inactive = _run(
        'import runpy; runpy.run_module("cuju.gfootball_alias", run_name="__main__")',
        check=False,
        path=(folder,),
    )
    assert inactive.returncode == 1
    assert 'cuju-gfootball is inactive' in inactive.stdout


@pytest.fixture(scope='module')
def wheel(tmp_path_factory):
    if importlib.util.find_spec('hatchling') is None:
        pytest.skip('building the wheel of cuju-gfootball needs hatchling')
    folder = tmp_path_factory.mktemp('wheel')
    subprocess.run(
        [
            sys.executable,
            '-m',
            'pip',
            'wheel',
            '--no-deps',
            '--no-build-isolation',
            '-q',
            '-w',
            str(folder),
            PROJECT,
        ],
        check=True,
        timeout=600,
    )
    (path,) = folder.glob('cuju_gfootball-*.whl')
    return path


def _version():
    with open(
        os.path.join(SOURCE, 'cuju', 'gfootball_alias', '_version.py'), encoding='utf-8'
    ) as handle:
        return re.search(r"__version__ = '([^']+)'", handle.read())[1]


def test_wheel_holds_the_alias_and_its_license(wheel):
    version = _version()
    info = f'cuju_gfootball-{version}.dist-info'
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata = archive.read(f'{info}/METADATA').decode()
    assert _load_script('check_wheel').alias_problems(wheel) == []
    assert names == {
        'cuju_gfootball.pth',
        'cuju/gfootball_alias/__init__.py',
        'cuju/gfootball_alias/__main__.py',
        'cuju/gfootball_alias/_startup.py',
        'cuju/gfootball_alias/_version.py',
        f'{info}/METADATA',
        f'{info}/WHEEL',
        f'{info}/RECORD',
        f'{info}/licenses/LICENSE',
    }
    assert re.search(r'^Requires-Dist: cuju-engine<0\.5,>=0\.4\.0$', metadata, re.M)
    assert re.search(r'^License-Expression: Apache-2\.0$', metadata, re.M)
    assert wheel.name == f'cuju_gfootball-{version}-py3-none-any.whl'


def test_wheel_check_finds_a_wheel_that_is_not_the_alias(tmp_path):
    path = tmp_path / 'cuju_gfootball-0.1.0-py3-none-any.whl'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('cuju_gfootball.pth', 'import os\n')
        archive.writestr('gfootball/__init__.py', '')
    assert _load_script('check_wheel').alias_problems(path) == [
        'missing cuju/gfootball_alias/__init__.py',
        'missing cuju/gfootball_alias/__main__.py',
        'missing cuju/gfootball_alias/_startup.py',
        'missing cuju/gfootball_alias/_version.py',
        "expected cuju_gfootball.pth to hold 'import cuju.gfootball_alias._startup\\n', "
        "found 'import os\\n'",
        'missing LICENSE',
        'expected the alias alone, found gfootball/__init__.py and 0 more',
        'missing the dependency on a range of cuju',
    ]


def test_install_activates_and_uninstall_leaves_nothing(wheel, tmp_path):
    home = tmp_path / 'venv'
    venv.EnvBuilder(with_pip=True).create(home)
    python = home / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    site_packages = subprocess.run(
        [str(python), '-c', 'import sysconfig; print(sysconfig.get_paths()["purelib"])'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    with open(os.path.join(site_packages, '_outer.pth'), 'w') as handle:
        handle.write('\n'.join([HOME, *_site_packages()]) + '\n')

    def files():
        return sorted(
            os.path.relpath(os.path.join(folder, name), site_packages)
            for folder, _, names in os.walk(site_packages)
            for name in names
        )

    before = files()
    pip = [str(python), '-m', 'pip', '-q', '--disable-pip-version-check']
    subprocess.run([*pip, 'install', '--no-deps', '--no-index', str(wheel)], check=True)
    check = subprocess.run(
        [
            str(python),
            '-c',
            'import gfootball, cuju.gfootball, cuju.gfootball_alias; '
            'print(gfootball is cuju.gfootball, cuju.gfootball_alias.__version__)',
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
    )
    assert check.stdout.split() == ['True', _version()]
    subprocess.run([*pip, 'uninstall', '-y', 'cuju-gfootball'], check=True)
    assert files() == before
    gone = subprocess.run(
        [str(python), '-c', 'import gfootball'], capture_output=True, text=True, cwd=tmp_path
    )
    assert "No module named 'gfootball'" in gone.stderr


def test_release_checks_the_wheel_and_publishes_it_after_the_engine():
    path = os.path.join(ROOT, '.github', 'workflows', 'release.yml')
    if not os.path.isfile(path):
        pytest.skip('needs the source tree, which lacks the release workflow')
    with open(path, encoding='utf-8') as handle:
        workflow = handle.read()
    job = workflow[workflow.index('\n    gfootball:') : workflow.index('\n    wheel:')]
    assert job.count('python -m build packages/cuju-gfootball') == 2
    assert 'python ci/checks/compare_builds.py' in job
    assert 'python ci/checks/check_wheel.py --alias dist/*.whl' in job
    engine = workflow[workflow.index('\n    publish:') : workflow.index('\n    publish-gfootball:')]
    assert 'gfootball' in engine[: engine.index('if:')]
    assert "pattern: '{sdist,wheel-*}'" in engine
    assert 'cuju_gfootball' not in engine
    publish = workflow[workflow.index('\n    publish-gfootball:') :]
    publish = publish[: publish.index('\n    publish-grf-media:')]
    assert 'needs: [gfootball, publish]' in publish
    assert 'environment: pypi-gfootball' in publish and 'id-token: write' in publish
    assert 'name: cuju-gfootball' in publish
    assert 'skip-existing' not in workflow
    check = publish.index('https://pypi.org/pypi/cuju-gfootball/$version/json')
    assert publish.index('packages/cuju-gfootball/src/cuju/gfootball_alias/_version.py') < check
    step = publish[check : publish.index('gh-action-pypi-publish')]
    assert '200) ' in step and 'upload=false' in step
    assert '404) ' in step and 'upload=true' in step
    assert '*) ' in step and 'exit 1' in step
    assert "if: steps.pypi.outputs.upload == 'true'" in publish[check:]
