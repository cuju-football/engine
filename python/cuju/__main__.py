# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib
import sys
from difflib import get_close_matches
from importlib import metadata

from . import _cli_contract as contract
from . import _cli_parsers as parsers
from ._version import __version__

GROUP = 'cuju.commands'
ENGINE = 'cuju-engine'
FAMILY = {'arena': 'cuju-arena', 'render': 'cuju-render'}
_MODULES = {'check': '.check', 'replay': '._replay_command', 'dump': '._dump_command'}


def _entries() -> dict[str, metadata.EntryPoint]:
    entries = {}
    for entry in metadata.entry_points(group=GROUP):
        if entry.name not in parsers.BUILDERS:
            entries.setdefault(entry.name, entry)
    return dict(sorted(entries.items()))


def version_lines() -> list[str]:
    for name in (ENGINE, 'cuju-engine-cpu'):
        try:
            engine = f'{name} {metadata.version(name)}'
            break
        except metadata.PackageNotFoundError:
            continue
    else:
        engine = f'{ENGINE} {__version__}'

    seen = {ENGINE, 'cuju-engine-cpu', 'cuju'}
    extensions = {}
    for entry in metadata.entry_points(group=GROUP):
        dist = entry.dist
        if dist is not None:
            key = dist.name.lower().replace('_', '-').replace('.', '-')
            if key not in seen:
                extensions[key] = f'{dist.name} {dist.version}'
    lines = [engine, *(extensions[key] for key in sorted(extensions))]
    try:
        lines.append(f'cuju {metadata.version("cuju")}')
    except metadata.PackageNotFoundError:
        pass
    return lines


class _Version(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        print('\n'.join(version_lines()))
        parser.exit()


def parser() -> contract.Parser:
    root = contract.Parser(
        prog='cuju',
        description='Run the engine tools and the commands of installed packages.',
        epilog=contract.examples('cuju help', 'cuju check', 'cuju completion bash'),
    )
    root.add_argument(
        '-V', '--version', action=_Version, help='print the versions of the installed packages'
    )
    contract.add_globals(root)
    group = root.add_subparsers(dest='command', metavar='NAME')
    for build in parsers.BUILDERS.values():
        build(group)
    return root


def _package_parser(entry: metadata.EntryPoint) -> argparse.ArgumentParser | None:
    module = importlib.import_module(entry.module)
    build = getattr(module, 'parser', None)
    return build() if build is not None else None


def package_parsers(entries):
    for name, entry in entries.items():
        try:
            tree = _package_parser(entry)
        except Exception:
            tree = None
        yield name, entry, tree


def _overview(root: contract.Parser) -> None:
    lines = [root.format_help().rstrip()]
    for name, entry, tree in package_parsers(_entries()):
        dist = entry.dist
        heading = f'{name}, {dist.name} {dist.version}' if dist else name
        lines.extend(('', heading))
        summaries = (
            [
                (choice.dest, choice.help)
                for group in contract.subcommands(tree)
                for choice in group._choices_actions
            ]
            if tree is not None
            else []
        )
        if summaries:
            width = max(len(word) for word, _ in summaries)
            for word, summary in summaries:
                lines.append(f'  {word:<{width}}  {summary or ""}'.rstrip())
        elif dist and dist.metadata['Summary']:
            lines.append(f'  {dist.metadata["Summary"]}')
    print('\n'.join(lines))


def _unknown(name, entries, as_json, help_parser=None) -> int:
    prog = help_parser.prog if help_parser is not None else 'cuju'
    if name in FAMILY:
        return contract.fail(
            prog,
            f'expected a command from {FAMILY[name]}, found the package missing',
            hint=f'pip install {FAMILY[name]}',
            as_json=as_json,
            status=contract.EXIT_ENVIRONMENT,
            error_id='command.package_missing',
            values={'command': name, 'package': FAMILY[name]},
        )
    names = parsers.BUILDERS.keys() | entries.keys() | FAMILY.keys()
    close = get_close_matches(name, sorted(names), n=3)
    hint = 'run cuju help to list the commands.'
    if close:
        hint = f'run cuju help {close[0]}, similar command names are {", ".join(close)}.'
    if help_parser is not None:
        help_parser.print_usage(sys.stderr)
    return contract.fail(
        prog,
        f'no command {name!r}',
        hint=hint,
        as_json=as_json,
        status=contract.EXIT_USAGE,
        error_id='command.unknown',
        values={'command': name},
    )


def _run_entry(entry: metadata.EntryPoint, arguments: list[str]) -> int | None:
    try:
        function = entry.load()
        if not callable(function):
            raise TypeError('expected a callable entry point')
    except BrokenPipeError:
        raise
    except Exception as error:
        dist = entry.dist
        package = dist.name if dist else entry.name
        return contract.fail(
            f'cuju {entry.name}',
            f'could not load the command from {package}, {type(error).__name__}, {error}',
            hint=f'run pip show {package} to check the installation.',
            as_json='--json' in contract.options(arguments),
            status=contract.EXIT_ENVIRONMENT,
            error_id='command.load',
            values={'command': entry.name, 'package': package},
        )
    status = function(arguments)
    return 0 if status is None else int(status)


def _children(tree):
    return {
        name: child for group in contract.subcommands(tree) for name, child in group.choices.items()
    }


def _command_help(root, args):
    name, *words = args.words
    entries = _entries()
    tree = _children(root).get(name)
    if tree is None:
        entry = entries.get(name)
        if entry is None:
            return _unknown(name, entries, False, args._command_parser)
        try:
            tree = _package_parser(entry)
        except BrokenPipeError:
            raise
        except Exception:
            tree = None
        if tree is None:
            return _run_entry(entry, [*words, '--help'])
    path = ['cuju', name]
    for word in words:
        child = _children(tree).get(word)
        if child is None:
            tree.print_usage(sys.stderr)
            prog = ' '.join(path)
            return contract.fail(
                prog,
                f'expected a command below {prog}, found {word!r}',
                hint=f'run {prog} --help to list the commands.',
                status=contract.EXIT_USAGE,
            )
        tree = child
        path.append(word)
    tree.print_help()
    return 0


def _dispatch(arguments: list[str], command_index: int) -> int | None:
    tail = arguments[command_index:]
    if tail and not tail[0].startswith('-') and tail[0] not in parsers.BUILDERS:
        name, *rest = tail
        entries = _entries()
        if name in entries:
            return _run_entry(entries[name], [*arguments[:command_index], *rest])
        return _unknown(name, entries, '--json' in contract.options(rest))

    root = parser()
    if not tail or tail[0] in ('-h', '--help'):
        _overview(root)
        return 0
    if tail[0].startswith('-') and tail[0] not in ('-V', '--version'):
        root.error(f'expected a root option or a command, found {tail[0]!r}')
    args = root.parse_args(arguments)
    if args.command in _MODULES:
        return importlib.import_module(_MODULES[args.command], __package__)._run(args)
    if args.command == 'help':
        if args.words:
            return _command_help(root, args)
        _overview(root)
    elif args.command == 'completion':
        from . import _completion

        print(_completion.script(args.shell, root, package_parsers(_entries()), version_lines()))
    return 0


def main(arguments: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if arguments is None else arguments
    index = 0
    for flag in arguments:
        if not contract.is_global(flag):
            break
        index += 1
    name = arguments[index] if index < len(arguments) else ''
    prog = f'cuju {name}' if name and not name.startswith('-') else 'cuju'
    return contract.guarded(prog, lambda: _dispatch(arguments, index), arguments)


if __name__ == '__main__':
    sys.exit(main())
