# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks how cuju meets the packages that install beside it, with a fake package `cuju.fakepkg`.

Every check runs in a new interpreter whose path holds a folder with a portion of the package
`cuju` and a fake distribution that declares entry points of the group `cuju.batch_hooks`.
"""

import importlib.util
import os
import subprocess
import sys
import textwrap

import pytest

HOOK = """
import os


def run(batch):
    with open(os.environ['FAKE_LOG'], 'a') as log:
        log.write('hook ' + str(batch.environment_count) + '\\n')


def fail(batch):
    raise RuntimeError('the hook failed')


def command(arguments):
    print('fakecmd ran with', *arguments)
    return 3
"""

BATCH = """
    import cuju

    batch = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=2, device='cpu')
    batch.reset()
    print(batch.environment_count)
    batch.close()
"""


@pytest.fixture
def site(tmp_path):
    """Returns a folder with the portion `cuju/fakepkg`, `fakehook.py` and a fake distribution."""
    folder = tmp_path / 'site'
    (folder / 'cuju' / 'fakepkg').mkdir(parents=True)
    (folder / 'cuju' / 'fakepkg' / '__init__.py').write_text('FAKE = True\n')
    (folder / 'fakehook.py').write_text(HOOK)
    info = folder / 'cuju_fakehook-1.0.dist-info'
    info.mkdir()
    (info / 'METADATA').write_text(
        'Metadata-Version: 2.1\nName: cuju-fakehook\nVersion: 1.0\nSummary: a fake package\n'
    )
    (info / 'entry_points.txt').write_text(
        '[cuju.batch_hooks]\nfake = fakehook:run\nbroken = fakehook:fail\n'
        '[cuju.commands]\nfakecmd = fakehook:command\ncheck = fakehook:command\n'
    )
    return folder


def run(code, *folders, **variables):
    """Runs `code` in a new interpreter that sees the same `cuju` and the extra folders."""
    home = os.path.dirname(os.path.dirname(importlib.util.find_spec('cuju').origin))
    environment = {**os.environ}
    environment.pop('CUJU_BATCH_HOOKS', None)
    environment.update(variables)
    environment['PYTHONPATH'] = os.pathsep.join([home, *map(str, folders)])
    return subprocess.run(
        [sys.executable, '-c', textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=300,
        env=environment,
    )


def test_a_portion_on_the_path_joins_the_package(site):
    result = run('import cuju.fakepkg as f; print(f.FAKE)', site)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ['True']


def test_a_whole_cuju_on_the_path_stays_out(tmp_path):
    stray = tmp_path / 'stray' / 'cuju'
    (stray / 'strayed').mkdir(parents=True)
    (stray / '__init__.py').write_text("raise SystemExit('the stray cuju was imported')\n")
    result = run('import cuju; print(any("stray" in p for p in cuju.__path__))', tmp_path / 'stray')
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ['False']


def test_a_missing_portion_suggests_its_package():
    code = """
        import importlib.util
        import cuju

        for attempt in (
            lambda: __import__('cuju.fakepkg'),
            lambda: importlib.util.find_spec('cuju.fakepkg'),
            lambda: __import__('cuju.no_such_one'),
        ):
            try:
                attempt()
            except ModuleNotFoundError as error:
                print(error.name)
                print(error)
        try:
            __import__('cuju.fakepkg.deeper')
        except ModuleNotFoundError as error:
            print(error.name)
        try:
            __import__('cuju._private_missing')
        except ModuleNotFoundError as error:
            print(error)
    """
    result = run(code)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == 'cuju.fakepkg' and 'pip install cuju-fakepkg' in lines[1]
    assert lines[2] == 'cuju.fakepkg' and 'cuju-fakepkg' in lines[3]
    assert lines[4] == 'cuju.no_such_one' and 'cuju-no-such-one' in lines[5]
    assert lines[6] == 'cuju.fakepkg'
    assert lines[7] == "No module named 'cuju._private_missing'"


def test_a_present_portion_is_found(site):
    result = run(
        "import importlib.util; print(importlib.util.find_spec('cuju.fakepkg') is None)", site
    )
    assert result.stdout.split() == ['False']


def test_a_command_of_another_package_runs(site):
    code = """
        import sys
        import cuju.__main__ as main

        sys.exit(main.main(['fakecmd', 'a', '--b']))
    """
    result = run(code, site)
    assert result.returncode == 3, result.stderr
    assert result.stdout.split() == ['fakecmd', 'ran', 'with', 'a', '--b']


def test_the_help_lists_the_commands_of_other_packages(site):
    code = """
        import sys
        import cuju.__main__ as main

        sys.exit(main.main(['help']))
    """
    result = run(code, site)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    heading = lines.index('fakecmd, cuju-fakehook 1.0')
    assert lines[heading + 1] == '  a fake package'
    assert sum(1 for line in lines if line.split()[:1] == ['check']) == 1
    plain = run(code)
    assert not any(line.split()[:1] == ['fakecmd'] for line in plain.stdout.splitlines())


def test_a_command_of_the_engine_wins_its_name(site):
    code = """
        import sys
        import cuju.__main__ as main

        sys.exit(main.main(['check', '--list']))
    """
    result = run(code, site)
    assert result.returncode == 0, result.stderr
    assert 'fakecmd' not in result.stdout


def test_an_unknown_command_suggests_an_installed_command(site):
    code = """
        import sys
        import cuju.__main__ as main

        sys.exit(main.main(['fakecm']))
    """
    result = run(code, site)
    assert result.returncode == 2
    assert result.stdout == ''
    assert result.stderr.splitlines() == [
        "cuju: error: no command 'fakecm'",
        'cuju: hint: run cuju help fakecmd, similar command names are fakecmd.',
    ]


def test_a_named_hook_receives_each_batch(site, tmp_path):
    log = tmp_path / 'log.txt'
    plain = run(BATCH, site, FAKE_LOG=str(log))
    result = run(BATCH, site, CUJU_BATCH_HOOKS='fake', FAKE_LOG=str(log))
    assert result.returncode == 0, result.stderr
    assert result.stdout == plain.stdout
    assert log.read_text().splitlines() == ['hook 2']


def test_a_missing_or_failing_hook_warns_and_the_batch_runs(site, tmp_path):
    log = tmp_path / 'log.txt'
    plain = run(BATCH, site, FAKE_LOG=str(log))
    code = "import logging; logging.basicConfig(format='%(message)s')\n" + textwrap.dedent(BATCH)
    result = run(code, site, CUJU_BATCH_HOOKS='absent, broken', FAKE_LOG=str(log))
    assert result.returncode == 0, result.stderr
    assert result.stdout == plain.stdout
    assert "names the hook 'absent', which no installed package provides" in result.stderr
    assert (
        "the hook 'broken' gave the batch nothing, RuntimeError: the hook failed" in result.stderr
    )


def test_a_plain_process_reads_no_entry_points(site):
    code = """
        import importlib.metadata as metadata
        import torch

        calls = []
        for name in ('entry_points', 'distributions', 'distribution', 'metadata', 'version'):
            def counted(*arguments, _original=getattr(metadata, name), _name=name, **keywords):
                calls.append(_name)
                return _original(*arguments, **keywords)
            setattr(metadata, name, counted)
        import cuju

        batch = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=2, device='cpu')
        batch.reset()
        batch.step(torch.zeros((2, 4), dtype=torch.int32))
        batch.close()
        print(len(calls))
    """
    result = run(code, site)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ['0']


def test_an_intact_engine_reports_itself():
    from cuju import engine_status

    assert engine_status.intact() and engine_status.missing() == []
    assert engine_status.repair_command().startswith(
        'pip install --force-reinstall --no-deps cuju-engine'
    )
    result = run('import sys, cuju.engine_status as s; sys.exit(s.main())')
    assert result.returncode == 0, result.stderr
    assert 'the engine is intact' in result.stdout


@pytest.mark.parametrize('distribution', [None, 'cuju-engine', 'cuju-engine-cpu'])
def test_a_damaged_engine_names_the_repair(tmp_path, distribution):
    # An upgrade from cuju 0.2.1 leaves the files with new names in the namespace.
    import importlib.util

    folder = tmp_path / 'damaged' / 'cuju'
    folder.mkdir(parents=True)
    for name in ('engine_status', '_cli_contract'):
        source = importlib.util.find_spec(f'cuju.{name}').origin
        (folder / f'{name}.py').write_text(open(source, encoding='utf-8').read())
    version = '0.3.7' if distribution else None
    if distribution:
        info = folder.parent / f'{distribution.replace("-", "_")}-{version}.dist-info'
        info.mkdir()
        (info / 'METADATA').write_text(
            f'Metadata-Version: 2.1\nName: {distribution}\nVersion: {version}\n'
        )
    code = (
        'import sys, cuju.engine_status as s; '
        'print(s.installed_version()); print(s.repair_command()); '
        'print(s.intact(), s.missing()); sys.exit(s.main())'
    )
    environment = {**os.environ, 'PYTHONPATH': str(tmp_path / 'damaged')}
    result = subprocess.run(
        [sys.executable, '-S', '-c', code], capture_output=True, text=True, env=environment
    )
    assert result.returncode == 1, result.stderr
    installed, command, status = result.stdout.splitlines()
    assert installed == str(version)
    assert status.startswith('False [')
    requirement = f'{distribution}=={version}' if distribution else 'cuju-engine'
    repair = f'pip install --force-reinstall --no-deps {requirement}'
    assert command == repair
    assert 'the engine is damaged' in result.stderr
    assert f'cuju-repair: hint: {repair}\n' in result.stderr


def test_the_repair_command_is_a_console_script():
    import importlib.util

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, 'pyproject.toml')
    if not os.path.isfile(path):
        pytest.skip('no source tree')
    text = open(path, encoding='utf-8').read()
    assert 'cuju-repair = "cuju.engine_status:main"' in text
    assert importlib.util.find_spec('cuju.engine_status') is not None
