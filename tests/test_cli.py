# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""Checks the engine command line and its installed entry points."""

import errno
import json
import os
import pickle
import re
import shlex
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest

from cuju import __main__ as cli
from cuju import _cli_contract as contract
from cuju import dumps, engine_status


@pytest.fixture
def packages(tmp_path, monkeypatch):
    def distribution(name, version, entries=''):
        info = tmp_path / f'{name.replace("-", "_")}-{version}.dist-info'
        info.mkdir()
        (info / 'METADATA').write_text(
            f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\nSummary: test package\n'
        )
        if entries:
            (info / 'entry_points.txt').write_text(f'[cuju.commands]\n{entries}\n')

    distribution('cuju-engine', '8.0.0')
    distribution('cuju', '9.0.0')
    distribution('cuju-look', '2.0.0', 'look = looker:main\nview = looker:main')
    distribution('cuju-broken', '1.0.0', 'broken = unavailable:main')
    (tmp_path / 'looker.py').write_text(
        """
import argparse

def parser():
    root = argparse.ArgumentParser(prog='cuju look')
    team = root.add_subparsers().add_parser('team', help='manage teams')
    add = team.add_subparsers().add_parser('add', help='add a team')
    add.add_argument('--device', choices=['cpu', 'cuda'])
    add.add_argument('--side').completion = ('builtin', 'builtin:0.3')
    return root

def main(arguments):
    if arguments == ['--help']:
        print('delegated help')
        return 0
    parser().parse_args(arguments)
"""
    )

    (tmp_path / 'unavailable.py').write_text("raise RuntimeError('entry point is broken')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    return tmp_path


def test_version_reads_each_distribution_once_without_loading_it(packages, capsys):
    assert cli.main(['-V']) == 0
    out, err = capsys.readouterr()
    assert out.splitlines() == [
        'cuju-engine 8.0.0',
        'cuju-broken 1.0.0',
        'cuju-look 2.0.0',
        'cuju 9.0.0',
    ]
    assert err == ''
    assert 'unavailable' not in sys.modules


def test_family_help_uses_the_parser_hook_and_falls_back_to_the_summary(packages, capsys):
    assert cli.main([]) == 0
    bare, err = capsys.readouterr()
    assert err == ''
    assert cli.main(['help']) == 0
    assert capsys.readouterr().out == bare
    assert 'look, cuju-look 2.0.0\n  team  manage teams' in bare
    assert 'broken, cuju-broken 1.0.0\n  test package' in bare
    assert cli.main(['help', 'look', 'team', 'add']) == 0
    nested = capsys.readouterr().out
    assert cli.main(['look', 'team', 'add', '--help']) == 0
    assert nested == capsys.readouterr().out
    assert '--device {cpu,cuda}' in nested
    assert cli.main(['help', 'look']) == 0
    assert 'manage teams' in capsys.readouterr().out
    assert cli.main(['look', '--help']) == 0
    assert capsys.readouterr().out == 'delegated help\n'


def test_a_typo_suggests_a_command_from_an_installed_package(packages, capsys):
    assert cli.main(['lok']) == 2
    out, err = capsys.readouterr()
    assert out == ''
    assert err.splitlines() == [
        "cuju: error: no command 'lok'",
        'cuju: hint: run cuju help look, similar command names are look.',
    ]


def test_a_broken_entry_point_is_an_environment_error(packages, capsys):
    assert cli.main(['broken']) == 3
    out, err = capsys.readouterr()
    assert out == ''
    assert err.splitlines() == [
        'cuju broken: error: could not load the command from cuju-broken, RuntimeError, '
        'entry point is broken',
        'cuju broken: hint: run pip show cuju-broken to check the installation.',
    ]


def test_unknown_words_and_missing_family_packages_have_separate_hints(capsys, monkeypatch):
    monkeypatch.setattr(metadata, 'entry_points', lambda **kwargs: [])
    assert cli.main(['chek']) == 2
    assert capsys.readouterr().err.splitlines() == [
        "cuju: error: no command 'chek'",
        'cuju: hint: run cuju help check, similar command names are check.',
    ]
    assert cli.main(['rnder']) == 2
    assert capsys.readouterr().err.splitlines() == [
        "cuju: error: no command 'rnder'",
        'cuju: hint: run cuju help render, similar command names are render.',
    ]
    assert cli.main(['render']) == 3
    assert capsys.readouterr().err.splitlines() == [
        'cuju: error: expected a command from cuju-render, found the package missing',
        'cuju: hint: pip install cuju-render',
    ]


def test_missing_family_command_has_a_json_error(capsys, monkeypatch):
    monkeypatch.setattr(metadata, 'entry_points', lambda **kwargs: [])
    assert cli.main(['arena', '--json']) == 3
    out, err = capsys.readouterr()
    assert json.loads(out) == {
        'schema': 'error/1',
        'error': 'expected a command from cuju-arena, found the package missing',
        'error_id': 'command.package_missing',
        'error_values': {'command': 'arena', 'package': 'cuju-arena'},
        'hint': 'pip install cuju-arena',
        'exit': 3,
    }
    assert err.splitlines()[-1] == 'cuju: hint: pip install cuju-arena'


@pytest.mark.parametrize('tail', [['--version'], ['check'], ['chek']])
def test_a_leading_unknown_root_option_is_reported(tail, capsys):
    assert cli.main(['--bad', *tail]) == 2
    out, err = capsys.readouterr()
    assert out == ''
    assert err.startswith('usage: cuju ')
    assert "cuju: error: expected a root option or a command, found '--bad'" in err


def test_an_invalid_replay_frame_rate_is_a_usage_error(capsys):
    assert cli.main(['replay', '--trace_file', 'game.dump', '--fps', 'no', '--json']) == 2
    out, err = capsys.readouterr()
    assert json.loads(out)['error'] == "expected an integer for --fps, found 'no'"
    assert err.startswith('usage: cuju replay ')


def test_replay_help_explains_when_to_set_the_frame_rate(capsys):
    assert cli.main(['replay', '--help']) == 0
    help_text = ' '.join(capsys.readouterr().out.split())
    assert 'Use --fps 10 only for a dump with 10 physics ticks per step.' in help_text
    assert 'Omit --fps to use the rate in the dump.' in help_text


@pytest.mark.parametrize('flags', [[], ['--verbose'], ['-qv']])
def test_unexpected_errors_exit_70_with_optional_traceback(capsys, monkeypatch, flags):
    def broken_metadata(**kwargs):
        raise RuntimeError('metadata failed\non two lines')

    monkeypatch.setattr(metadata, 'entry_points', broken_metadata)
    assert cli.main([*flags, '--version']) == 70
    out, err = capsys.readouterr()
    assert out == ''
    assert err.splitlines()[:2] == [
        'cuju: error: internal error, RuntimeError, metadata failed on two lines',
        'cuju: hint: rerun with --verbose and report the traceback.',
    ]
    assert ('Traceback' in err) == bool(flags)


def test_an_interrupt_exits_130(capsys, monkeypatch):
    def interrupted(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(metadata, 'entry_points', interrupted)
    assert cli.main(['help']) == 130
    assert capsys.readouterr() == ('', '')


@pytest.mark.parametrize('args', [['help'], ['--version'], ['check', '--json', '--bad']])
def test_a_closed_pipe_exits_quietly(args):
    reader, writer = os.pipe()
    os.close(reader)
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'cuju', *args],
            stdout=writer,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
        )
    finally:
        os.close(writer)
    assert result.returncode == 0
    assert 'BrokenPipeError' not in result.stderr
    if '--bad' not in args:
        assert result.stderr == ''


@pytest.mark.parametrize('when', ['write', 'flush'])
def test_a_windows_closed_pipe_exits_quietly(when, monkeypatch, capsys):
    class Output:
        def write(self, text):
            if when == 'write':
                raise OSError(errno.EINVAL, 'Invalid argument')

        def flush(self):
            if when == 'flush':
                raise OSError(errno.EINVAL, 'Invalid argument')

    redirected = []
    with monkeypatch.context() as windows:
        windows.setattr(sys, 'platform', 'win32')
        windows.setattr(sys, 'stdout', Output())
        windows.setattr(contract, '_close_output', lambda: redirected.append(True))
        status = contract.guarded('cuju check', lambda: print('answer'), [])
    assert status == 0
    assert redirected == [True]
    assert capsys.readouterr().err == ''


def test_einval_on_linux_is_an_internal_error(monkeypatch, capsys):
    def invalid():
        raise OSError(errno.EINVAL, 'Invalid argument')

    with monkeypatch.context() as linux:
        linux.setattr(sys, 'platform', 'linux')
        assert contract.guarded('cuju check', invalid, []) == 70
    assert 'cuju check: error: internal error, OSError' in capsys.readouterr().err


def test_the_windows_launcher_uses_the_command_name(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', [r'C:\Program Files\Cuju\cuju.exe', 'replay'])
    assert cli.main() == 2
    out, err = capsys.readouterr()
    assert out == ''
    assert err.startswith('usage: cuju replay ')
    assert 'cuju replay: error: ' in err
    assert 'cuju replay: hint: run cuju replay --help' in err


@pytest.mark.parametrize('command', ['replay', 'dump-to-txt'])
@pytest.mark.parametrize(
    ('suffix', 'escape'), [('', False), ('\u00a0', False), ("'", True), ('"', True), ('\n', True)]
)
def test_dump_input_errors_quote_windows_paths(command, suffix, escape, monkeypatch, capsys):
    path = r'C:\Users\Player\match.dump' + suffix
    quoted = repr(path) if escape else "'" + path + "'"

    def missing(file):
        assert file == path
        raise FileNotFoundError(errno.ENOENT, 'No such file or directory')

    monkeypatch.setattr(dumps, '_read_cli_dump', missing)
    extra = ['--output', 'output.txt'] if command == 'dump-to-txt' else []
    assert cli.main([command, '--trace_file', path, *extra, '--json']) == 1
    out, err = capsys.readouterr()
    error, _ = err.splitlines()
    assert f' at {quoted}, found ' in error
    assert json.loads(out)['error_values']['file'] == path


def test_check_errors_quote_a_windows_library_path(monkeypatch, capsys):
    path = r'C:\Users\Player\missing-cuju-library.dll'
    monkeypatch.setenv('CUJU_CPU_LIBRARY', path)
    assert cli.main(['check', '--device', 'cpu', '--quiet']) == 3
    err = capsys.readouterr().err
    assert f"CUJU_CPU_LIBRARY names '{path}', which is no file" in err


@pytest.mark.parametrize('command', ['replay', 'dump-to-txt'])
@pytest.mark.parametrize('content', [None, b'', b'bad pickle', pickle.dumps(1), b'\x80\x04'])
def test_dump_input_errors_have_one_error_and_one_hint(command, content, tmp_path, capsys):
    path = tmp_path / 'input.dump'
    output = tmp_path / 'output.txt'
    if content is not None:
        path.write_bytes(content)
    rest = ['--output', str(output)] if command == 'dump-to-txt' else []
    assert cli.main([command, '--trace_file', str(path), *rest]) == 1
    out, err = capsys.readouterr()
    assert out == '' and not output.exists()
    error, hint = err.splitlines()
    assert error.startswith(f'cuju {command}: error: expected ')
    assert str(path) in error
    assert hint.startswith(f'cuju {command}: hint: ')


def test_dump_text_prints_the_written_path_and_keeps_grf_flags(tmp_path, capsys):
    path = tmp_path / 'input.dump'
    output = tmp_path / 'output.txt'
    step = {'observation': {}, 'debug': {'time': 1}}
    path.write_bytes(pickle.dumps(step))
    options = ['dump-to-txt', '--trace_file', str(path), '--output', str(output)]
    assert cli.main([*options, '--include_debug=false']) == 0
    assert output.read_text() == "[{'observation': {}}]"
    assert capsys.readouterr() == (f'{output}\n', '')
    assert cli.main([*options, '--noinclude_debug']) == 0
    assert output.read_text() == "[{'observation': {}}]"
    assert capsys.readouterr() == (f'{output}\n', '')
    assert cli.main([*options, '--json']) == 0
    out, err = capsys.readouterr()
    assert json.loads(out) == {'schema': 'engine.dump_to_txt/1', 'files': [str(output)]}
    assert err == ''


def test_repair_examples_parse():
    tree = engine_status.parser()
    lines = tree.epilog.splitlines()
    assert lines[0] == 'Examples' and len(lines) > 1
    for line in lines[1:]:
        command, *arguments = shlex.split(line)
        assert command == 'cuju-repair'
        tree.parse_args(arguments)


@pytest.mark.parametrize('distribution', ['cuju-engine', 'cuju-engine-cpu'])
def test_repair_json_reports_the_installation(distribution, monkeypatch, capsys):
    def version(name):
        if name == distribution:
            return '8.0.0'
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, 'version', version)
    assert engine_status.main([]) == 0
    assert capsys.readouterr() == (f'the engine is intact, {distribution} 8.0.0\n', '')
    assert engine_status.main(['--json']) == 0
    out, err = capsys.readouterr()
    assert out.startswith('{\n  "schema"')
    assert json.loads(out) == {
        'schema': 'engine.repair/1',
        'intact': True,
        'version': '8.0.0',
        'missing': [],
        'repair_command': None,
    }
    assert err == ''
    monkeypatch.setattr(engine_status, 'missing', lambda: ['__init__.py'])
    assert engine_status.main(['--json']) == 1
    out, err = capsys.readouterr()
    answer = json.loads(out)
    assert answer['intact'] is False and answer['missing'] == ['__init__.py']
    assert answer['version'] == '8.0.0'
    repair = f'pip install --force-reinstall --no-deps {distribution}==8.0.0'
    assert answer['repair_command'] == repair
    assert 'cuju-repair: error: the engine is damaged' in err
    assert f'cuju-repair: hint: {repair}\n' in err


def test_repair_help_and_usage_errors(capsys):
    assert engine_status.main(['--help']) == 0
    out, err = capsys.readouterr()
    assert 'cuju-repair --json' in out and err == ''
    assert engine_status.main(['--json', '--nonesuch']) == 2
    out, err = capsys.readouterr()
    assert json.loads(out)['schema'] == 'error/1'
    assert 'cuju-repair: error: ' in err


@pytest.mark.parametrize(
    ('shell', 'registration'),
    [
        ('bash', 'complete -F _cuju cuju'),
        ('zsh', '#compdef cuju'),
        ('fish', 'complete -c cuju'),
        ('powershell', 'Register-ArgumentCompleter -Native -CommandName cuju'),
    ],
)
def test_completion_includes_installed_parsers(packages, capsys, shell, registration):
    from cuju.check import load_cases

    assert cli.main(['completion', shell]) == 0
    out, err = capsys.readouterr()
    assert err == ''
    assert '# Cuju completion for cuju-engine 8.0.0, cuju-broken 1.0.0, cuju-look 2.0.0' in out
    assert registration in out
    assert 'look team add' in out
    assert 'help look team add' in out
    assert 'builtin:0.3' in out
    assert 'cpu' in out and 'cuda' in out
    assert all(name in out for name in load_cases(long=True))


def _bash_for_completion():
    if sys.platform != 'win32':
        bash = shutil.which('bash')
        if bash is None:
            pytest.skip('bash is unavailable')
        return bash

    roots = (
        os.environ.get('ProgramW6432'),
        os.environ.get('ProgramFiles', r'C:\Program Files'),
        os.environ.get('ProgramFiles(x86)'),
    )
    candidates = [Path(root) / 'Git' / 'bin' / 'bash.exe' for root in dict.fromkeys(roots) if root]
    for path in candidates:
        if path.is_file():
            return str(path)
    pytest.skip(f'Git Bash is unavailable, looked for {", ".join(map(str, candidates))}')


@pytest.mark.parametrize('installed', ['ProgramW6432', 'ProgramFiles', 'ProgramFiles(x86)', None])
def test_windows_completion_selects_git_bash(installed, monkeypatch, tmp_path):
    candidates = {}
    for variable in ('ProgramW6432', 'ProgramFiles', 'ProgramFiles(x86)'):
        root = tmp_path / variable
        monkeypatch.setenv(variable, str(root))
        candidates[variable] = root / 'Git' / 'bin' / 'bash.exe'
    if installed is not None:
        path = candidates[installed]
        path.parent.mkdir(parents=True)
        path.touch()

    with monkeypatch.context() as windows:
        windows.setattr(sys, 'platform', 'win32')
        windows.setattr(shutil, 'which', lambda _: r'C:\Windows\System32\bash.exe')
        if installed is not None:
            assert _bash_for_completion() == str(candidates[installed])
        else:
            with pytest.raises(pytest.skip.Exception) as skipped:
                _bash_for_completion()
            assert all(str(path) in str(skipped.value) for path in candidates.values())


def test_bash_completes_nested_choices_and_files(packages, capsys):
    bash = _bash_for_completion()
    assert cli.main(['completion', 'bash']) == 0
    script = packages / 'cuju.bash'
    script.write_text(capsys.readouterr().out)
    (packages / 'game one.dump').touch()
    result = subprocess.run(
        [
            bash,
            '--noprofile',
            '--norc',
            '-c',
            """
            source "$1"
            compopt() { :; }
            COMP_WORDS=(cuju look team add --device c)
            COMP_CWORD=5
            _cuju
            printf '<%s>\n' "${COMPREPLY[@]}"
            COMP_WORDS=(cuju help look te)
            COMP_CWORD=3
            _cuju
            printf '<%s>\n' "${COMPREPLY[@]}"
            COMP_WORDS=(cuju look team add --side builtin : 0)
            COMP_CWORD=7
            _cuju
            printf '<%s>\n' "${COMPREPLY[@]}"
            COMP_WORDS=(cuju replay --trace_file ga)
            COMP_CWORD=3
            _cuju
            printf '<%s>\n' "${COMPREPLY[@]}"
            COMP_WORDS=(cuju check --device=c)
            COMP_CWORD=2
            _cuju
            printf '<%s>\n' "${COMPREPLY[@]}"
            COMP_WORDS=(cuju check --device = c)
            COMP_CWORD=4
            _cuju
            printf '<%s>\n' "${COMPREPLY[@]}"
            COMP_WORDS=(cuju check --case 11_vs_11_competition/simple115v)
            COMP_CWORD=3
            _cuju
            printf '<%s>\n' "${COMPREPLY[@]}"
            for device in au cp cu; do
                COMP_WORDS=(cuju replay --device "$device")
                COMP_CWORD=3
                _cuju
                printf '<%s>\n' "${COMPREPLY[@]}"
            done
            for value in fa tr; do
                COMP_WORDS=(cuju dump-to-txt --include_debug "$value")
                COMP_CWORD=3
                _cuju
                printf '<%s>\n' "${COMPREPLY[@]}"
            done
            COMP_WORDS=(cuju dump-to-txt --include_debug --out)
            COMP_CWORD=3
            _cuju
            printf '<%s>\n' "${COMPREPLY[@]}"
            """,
            'bash',
            str(script),
        ],
        cwd=packages,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stderr == ''
    assert result.stdout.splitlines() == [
        '<cpu>',
        '<cuda>',
        '<team>',
        '<0.3>',
        '<game one.dump>',
        '<--device=cpu>',
        '<--device=cuda>',
        '<cpu>',
        '<cuda>',
        '<11_vs_11_competition/simple115v2>',
        '<auto>',
        '<cpu>',
        '<cuda>',
        '<false>',
        '<true>',
        '<--output>',
    ]


@pytest.mark.skipif(shutil.which('zsh') is None, reason='zsh is unavailable')
def test_zsh_script_sources_in_a_fresh_shell(packages, capsys):
    assert cli.main(['completion', 'zsh']) == 0
    script = packages / '_cuju'
    script.write_text(capsys.readouterr().out)
    result = subprocess.run(
        ['zsh', '-f', '-c', 'source "$1"; rc=$?; exit "$rc"', 'zsh', str(script)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == result.stderr == ''


@pytest.mark.skipif(shutil.which('zsh') is None, reason='zsh is unavailable')
@pytest.mark.parametrize(
    ('setup', 'registration'),
    [
        pytest.param('compinit -D\nsource ./_cuju', '_cuju_complete', id='source'),
        pytest.param('fpath=("$PWD" "${fpath[@]}")\ncompinit -D', '_cuju', id='fpath'),
    ],
)
def test_zsh_completes_commands_options_and_choices(packages, capsys, setup, registration):
    assert cli.main(['completion', 'zsh']) == 0
    (packages / '_cuju').write_text(capsys.readouterr().out)
    (packages / 'setup.zsh').write_text(
        'autoload -Uz compinit\n'
        f'{setup}\n'
        """
        show_buffer() {
            print -r -- "<BUFFER>${BUFFER}</BUFFER>"
            BUFFER=
        }
        zle -N show_buffer
        bindkey -e
        bindkey '^X^B' show_buffer
        print -r -- "<READY>${_comps[cuju]}</READY>"
        """
    )
    completions = {
        'cuju ch': 'cuju check',
        'cuju check --lo': 'cuju check --long',
        'cuju check --device cu': 'cuju check --device cuda',
        'cuju check --device=cu': 'cuju check --device=cuda',
        'cuju check --case 11_vs_11_competition/simple115v': (
            'cuju check --case 11_vs_11_competition/simple115v2'
        ),
        'cuju replay --device au': 'cuju replay --device auto',
        'cuju replay --device cp': 'cuju replay --device cpu',
        'cuju replay --device cu': 'cuju replay --device cuda',
        'cuju dump-to-txt --include_debug fa': 'cuju dump-to-txt --include_debug false',
        'cuju dump-to-txt --include_debug tr': 'cuju dump-to-txt --include_debug true',
        'cuju lo': 'cuju look',
        'cuju help look te': 'cuju help look team',
        'cuju help look team ad': 'cuju help look team add',
        'cuju look team add --side builtin:0': 'cuju look team add --side builtin:0.3',
        'cuju look te': 'cuju look team',
        'cuju look team add --de': 'cuju look team add --device',
        'cuju look team add --device cu': 'cuju look team add --device cuda',
    }
    driver = r"""
        set -e
        zmodload zsh/zpty
        zpty completion zsh -f -i
        zpty -w completion 'source ./setup.zsh'
        zpty -r completion output '*<READY>*</READY>'
        print -r -- "$output"
        for prefix in "$@"; do
            zpty -w -n completion "$prefix"$'\t\x18\x02'
            zpty -r completion output '*<BUFFER>*</BUFFER>'
            print -r -- "$output"
        done
        zpty -d completion
    """
    result = subprocess.run(
        ['zsh', '-f', '-c', driver, 'zsh', *completions],
        cwd=packages,
        env={**os.environ, 'TERM': 'xterm', 'ZDOTDIR': str(packages)},
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    assert f'<READY>{registration}</READY>' in result.stdout, result.stdout
    buffers = re.findall(r'<BUFFER>([^\n]*)</BUFFER>', result.stdout)
    assert [buffer.rstrip() for buffer in buffers] == list(completions.values()), result.stdout


@pytest.mark.parametrize(
    'arguments',
    [
        ('--help',),
        ('help', 'check'),
        ('completion', 'bash'),
        ('check', '--list'),
        ('check', '--list', '--long', '--case', '5_vs_5/simple115v2', '--json'),
    ],
)
def test_read_only_cli_does_not_load_the_batch(arguments):
    probe = """
import sys
from cuju.__main__ import main
assert main(sys.argv[1:]) == 0
assert 'torch' not in sys.modules
assert not any(name.startswith('cuju._batch') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, '-c', probe, *arguments], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
