# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""Checks the engine against the family command line contract."""

import argparse
import dataclasses
import importlib
import json
import re
import shlex
from collections.abc import Iterator

import pytest


@dataclasses.dataclass(frozen=True)
class Profile:
    """Describes the contract that applies to this package."""

    package: str
    distribution: str
    module: str
    globals: tuple[str, ...]
    settings: bool
    no_json: dict[str, str]
    exempt: dict[str, str]
    allowed: dict[str, str]
    variables: dict[str, str]


PROFILE = Profile(
    package='',
    distribution='cuju-engine',
    module='cuju.__main__',
    globals=('--quiet', '--verbose'),
    settings=False,
    no_json={
        'completion': 'prints a shell script',
        'help': 'prints help text',
        'agents client': 'prints the source of a file',
    },
    exempt={},
    allowed={'--list': 'keeps the self-check flag from 0.2.1'},
    variables={},
)

SHARED = {
    'json': ('--json',),
    'quiet': ('-q', '--quiet'),
    'verbose': ('-v', '--verbose'),
    'home': ('--home',),
    'arena_home': ('--arena-home',),
    'config': ('--config',),
    'overrides': ('--set',),
    'output': ('-o', '--output'),
    'seed': ('--seed',),
    'scenario': ('--scenario',),
    'device': ('--device',),
    'format': ('--format',),
    'option': ('--option',),
    'preset': ('--preset',),
    'size': ('--size',),
    'port': ('--port',),
    'open': ('--open',),
    'limit': ('--limit',),
    'resolution': ('--resolution',),
    'overwrite': ('--overwrite',),
    'replace': ('--replace',),
}
FORBIDDEN = {
    '--out',
    '--to',
    '--submission-format',
    '--ability',
    '--left-ability',
    '--right-ability',
    '--eleven',
    '--no-open',
    '--delete-group',
    '--restore-group',
    '--package',
    '--list',
    '--runs',
    '--table',
    '--show',
    '--draw',
    '--reset',
    '--install',
    '--uninstall',
    '--at-login',
    '--max-size',
}
SHORT = {'-h', '-V', '-q', '-v', '-o'}
METAVARS = {
    'FILE',
    'DIR',
    'PATH',
    'ID',
    'TEAM',
    'REC',
    'NAME',
    'OUT',
    'CODE',
    'QUERY',
    'KEY=VALUE',
    'N',
    'SCENARIO',
    'DEVICE',
    'DIFFICULTY',
    'R',
    'SECONDS',
    'TIME',
    'A',
    'B',
    'LESSON',
    'SOURCE',
    'TARGET',
}
VERBS = {
    'add',
    'new',
    'list',
    'show',
    'set',
    'remove',
    'restore',
    'export',
    'convert',
    'check',
    'run',
    'resume',
    'cancel',
    'open',
    'serve',
    'play',
    'search',
    'draw',
    'forfeit',
    'preflight',
    'legs',
    'replay',
    'table',
    'tasks',
    'presets',
    'ratings',
    'reset',
    'install',
    'uninstall',
    'on',
    'off',
    'status',
    'parts',
    'client',
}
_PROSE = re.compile(r'[;()\[\]{}`\"!?]|: ')
_EXAMPLE = re.compile(r'^\s+(cuju .+)$', re.MULTILINE)


@dataclasses.dataclass(frozen=True)
class Node:
    """One parser of the tree, with its words below `cuju` and the summary its parent gives it."""

    words: tuple[str, ...]
    parser: argparse.ArgumentParser
    summary: str | None
    leaf: bool


def _root() -> argparse.ArgumentParser:
    return importlib.import_module(PROFILE.module).parser()


def _subparsers(parser: argparse.ArgumentParser) -> argparse._SubParsersAction | None:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    return None


def walk(parser: argparse.ArgumentParser, words: tuple[str, ...] = ()) -> Iterator[Node]:
    """Yields every parser of a tree, the root first, with its summary."""
    yield from _walk(parser, words, None)


def _walk(parser, words, summary):
    group = _subparsers(parser)
    yield Node(words, parser, summary, group is None)
    if group is None:
        return
    helps = {choice.dest: choice.help for choice in group._choices_actions}
    seen: set[int] = set()
    for name, child in group.choices.items():
        if id(child) in seen:
            continue
        seen.add(id(child))
        yield from _walk(child, (*words, name), helps.get(name))


def _base() -> tuple[str, ...]:
    return (PROFILE.package,) if PROFILE.package else ()


def _label(node: Node) -> str:
    return ' '.join(node.words[len(_base()) :])


NODES = list(walk(_root(), _base()))
LEAVES = [node for node in NODES if node.leaf]
CHECKED = [node for node in LEAVES if _label(node) not in PROFILE.exempt]


def _options(parser: argparse.ArgumentParser) -> Iterator[argparse.Action]:
    for action in parser._actions:
        if action.option_strings and not isinstance(action, argparse._HelpAction):
            yield action


def _flags(parser: argparse.ArgumentParser) -> set[str]:
    return {flag for action in _options(parser) for flag in action.option_strings}


@pytest.fixture(autouse=True)
def _homes(tmp_path, monkeypatch):
    for name, below in PROFILE.variables.items():
        monkeypatch.setenv(name, str(tmp_path / below) if below else '')


def _run(words: list[str], capsys) -> tuple[int, str, str]:
    main = importlib.import_module(PROFILE.module).main
    try:
        status = main(list(words[len(_base()) :]))
    except SystemExit as exit:
        status = exit.code if isinstance(exit.code, int) else 1
    captured = capsys.readouterr()
    return int(status or 0), captured.out, captured.err


def test_the_tree_is_at_most_two_words_below_the_package():
    deep = [_label(node) for node in NODES if len(node.words) - len(_base()) > 2]
    assert not deep


def test_no_command_has_an_alias():
    for node in NODES:
        group = _subparsers(node.parser)
        if group is None:
            continue
        parsers = list(group.choices.values())
        assert len(parsers) == len({id(item) for item in parsers}), _label(node)


def test_a_word_below_a_noun_is_a_known_verb():
    for node in NODES:
        depth = len(node.words) - len(_base())
        if depth == 2:
            assert node.words[-1] in VERBS, _label(node)


@pytest.mark.parametrize('node', [n for n in NODES if n.words != _base()], ids=_label)
def test_every_command_has_a_short_summary(node):
    assert node.summary, 'no summary'
    assert len(node.summary) <= 60, node.summary
    assert node.summary[:1].islower() and not node.summary.endswith('.'), node.summary
    assert not _PROSE.search(node.summary), node.summary


@pytest.mark.parametrize('node', NODES, ids=_label)
def test_every_command_has_a_description(node):
    text = node.parser.description or ''
    assert text.strip(), 'no description'
    assert text.rstrip().endswith('.'), text
    assert not _PROSE.search(text), text


@pytest.mark.parametrize('node', NODES, ids=_label)
def test_help_fits_an_eighty_column_terminal(node, monkeypatch):
    monkeypatch.setenv('COLUMNS', '80')
    lines = node.parser.format_help().splitlines()
    assert max(map(len, lines)) <= 80


@pytest.mark.parametrize('node', LEAVES, ids=_label)
def test_every_leaf_has_examples_that_parse(node):
    epilog = node.parser.epilog or ''
    assert epilog.lstrip().startswith('Examples'), 'no Examples block'
    examples = _EXAMPLE.findall(epilog)
    assert examples, 'no example'
    prefix = ['cuju', *node.words]
    for example in examples:
        words = shlex.split(example)
        assert words[: len(prefix)] == prefix, example
        try:
            _root().parse_args(words[1 + len(_base()) :])
        except SystemExit:
            pytest.fail(f'the example does not parse, {example}')


@pytest.mark.parametrize('node', LEAVES, ids=_label)
def test_option_help_is_prose_and_metavars_are_known(node):
    for action in node.parser._actions:
        if action.help in (None, argparse.SUPPRESS) or isinstance(action, argparse._HelpAction):
            continue
        if isinstance(action, argparse._SubParsersAction):
            continue
        assert not _PROSE.search(action.help), action.help
        assert not action.help.endswith('.'), action.help
        metavar = action.metavar
        if isinstance(metavar, str) and action.nargs != 0:
            assert metavar in METAVARS, f'{action.dest}, {metavar}'


@pytest.mark.parametrize('node', CHECKED, ids=_label)
def test_shared_options_keep_their_spelling(node):
    for action in _options(node.parser):
        flags = set(action.option_strings)
        replaced = flags & FORBIDDEN - set(PROFILE.allowed)
        if isinstance(action, argparse.BooleanOptionalAction) and '--open' in flags:
            replaced -= {'--no-open'}
        assert not replaced, sorted(replaced)
        assert {flag for flag in flags if not flag.startswith('--')} <= SHORT, sorted(flags)
        expected = SHARED.get(action.dest)
        if expected is not None:
            assert set(expected) <= flags, f'{action.dest}, {sorted(flags)}'


@pytest.mark.parametrize('node', CHECKED, ids=_label)
def test_shared_options_take_their_values_alike(node):
    for action in _options(node.parser):
        if action.dest == 'json':
            assert action.nargs == 0 and action.const is True, 'a flag, never a value'
        if action.dest in ('config', 'overrides'):
            assert isinstance(action, argparse._AppendAction)
            assert action.metavar == ('FILE' if action.dest == 'config' else 'KEY=VALUE')
        if action.dest == 'open':
            assert isinstance(action, argparse.BooleanOptionalAction)
        if action.dest == 'seed':
            assert action.type is int


@pytest.mark.parametrize('node', LEAVES, ids=_label)
def test_every_leaf_takes_the_global_options(node):
    flags = _flags(node.parser)
    missing = [flag for flag in PROFILE.globals if flag not in flags]
    assert not missing, missing


@pytest.mark.parametrize('node', CHECKED, ids=_label)
def test_a_leaf_that_prints_data_takes_json(node):
    if _label(node) in PROFILE.no_json:
        assert '--json' not in _flags(node.parser), 'listed as printing no data'
        return
    assert '--json' in _flags(node.parser)


def _bad_values(node: Node) -> Iterator[list[str]]:
    yield ['--no-such-option']
    for action in _options(node.parser):
        flag = action.option_strings[-1]
        if action.type is int:
            yield [flag, 'not-a-number']
        if action.choices:
            yield [flag, 'no-such-choice']


@pytest.mark.parametrize('node', LEAVES, ids=_label)
def test_bad_arguments_exit_2_with_one_error_line_and_no_traceback(node, capsys):
    prog = ' '.join(['cuju', *node.words])
    for bad in _bad_values(node):
        status, out, err = _run([*node.words, *bad], capsys)
        assert status == 2, (bad, status)
        assert 'Traceback' not in err and 'Traceback' not in out, bad
        errors = [line for line in err.splitlines() if ': error: ' in line]
        assert len(errors) == 1, err
        assert errors[0].startswith(f'{prog}: error: '), errors[0]
        assert out == '', 'a usage error writes nothing on standard output without --json'


@pytest.mark.parametrize(
    'node', [n for n in LEAVES if _label(n) not in PROFILE.no_json], ids=_label
)
def test_a_usage_error_under_json_is_one_json_object(node, capsys):
    status, out, err = _run([*node.words, '--json', '--no-such-option'], capsys)
    assert status == 2
    answer = json.loads(out)
    assert next(iter(answer)) == 'schema' and answer['schema'] == 'error/1'
    assert answer['exit'] == 2 and answer['error']
    assert set(answer) >= {'error', 'error_id', 'error_values', 'hint', 'exit'}
    assert ': error: ' in err, 'the human line goes to standard error as well'


@pytest.mark.parametrize('node', [n for n in NODES if not n.leaf], ids=_label)
def test_a_bare_noun_exits_2_with_its_usage(node, capsys):
    if node.words == _base():
        return
    status, _out, err = _run(list(node.words), capsys)
    assert status == 2 and 'Traceback' not in err
    assert ' '.join(['cuju', *node.words]) in err


def test_the_version_names_the_distribution(capsys):
    status, out, _ = _run([*_base(), '--version'], capsys)
    assert status == 0
    lines = out.splitlines()
    engine = re.escape(PROFILE.distribution)
    assert re.fullmatch(rf'{engine}(?:-cpu)? \d+\.\d+\.\d+\S*', lines[0]), out
    for line in lines[1:]:
        assert re.fullmatch(r'\S+ \S+', line), line


def answer(out: str, schema: str) -> dict:
    """Returns the one JSON object of a command's standard output, checked against the contract."""
    found = json.loads(out)
    assert isinstance(found, dict), 'one object, never a bare list'
    assert list(found)[:1] == ['schema'] and found['schema'] == schema, found.get('schema')
    assert out.startswith('{\n  "schema"'), 'indent 2'
    return found
