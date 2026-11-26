# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import errno
import json
import os
import re
import sys
import traceback
import unicodedata
from collections.abc import Callable, Sequence

EXIT_NO = 1
EXIT_USAGE = 2
EXIT_ENVIRONMENT = 3
EXIT_INTERNAL = 70
EXIT_INTERRUPTED = 130

GLOBAL_FLAGS = ('-q', '--quiet', '-v', '--verbose')


def is_global(flag: str) -> bool:
    return flag in GLOBAL_FLAGS or (
        flag.startswith('-') and len(flag) > 1 and set(flag[1:]) <= {'q', 'v'}
    )


def options(arguments: Sequence[str]) -> Sequence[str]:
    return arguments[: arguments.index('--')] if '--' in arguments else arguments


def emit(schema: str, payload: dict) -> None:
    print(json.dumps({'schema': schema, **payload}, indent=2))


def quote_path(path: str | os.PathLike[str]) -> str:
    text = os.fspath(path)
    if (
        "'" in text
        or '"' in text
        or any(unicodedata.category(char) in ('Cc', 'Cf', 'Cs', 'Zl', 'Zp') for char in text)
    ):
        return repr(text)
    return f"'{text}'"


def fail(
    prog: str,
    error: str,
    *,
    hint: str | None = None,
    as_json: bool = False,
    status: int = EXIT_NO,
    error_id: str | None = None,
    values: dict | None = None,
) -> int:
    error = ' '.join(error.splitlines())
    print(f'{prog}: error: {error}', file=sys.stderr)
    if hint:
        print(f'{prog}: hint: {hint}', file=sys.stderr)
    if as_json:
        emit(
            'error/1',
            {
                'error': error,
                'error_id': error_id,
                'error_values': values or {},
                'hint': hint,
                'exit': status,
            },
        )
    return status


class _HelpFormatter(argparse.RawDescriptionHelpFormatter):
    def _fill_text(self, text, width, indent):
        if '\n' not in text:
            return argparse.HelpFormatter._fill_text(self, text, width, indent)
        return super()._fill_text(text, width, indent)


class Parser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('formatter_class', _HelpFormatter)
        kwargs.setdefault('allow_abbrev', False)
        super().__init__(*args, **kwargs)
        self._as_json = False

    def parse_known_args(self, args=None, namespace=None):
        flags = options(sys.argv[1:] if args is None else args)
        self._as_json = '--json' in self._option_string_actions and any(
            flag == '--json' or flag.startswith('--json=') for flag in flags
        )
        return super().parse_known_args(args, namespace)

    def parse_args(self, args=None, namespace=None):
        parsed, remaining = self.parse_known_args(args, namespace)
        if remaining:
            selected = getattr(parsed, '_command_parser', self)
            selected.error(
                f'expected arguments accepted by {selected.prog}, found '
                + ', '.join(map(repr, remaining))
            )
        return parsed

    def _check_value(self, action, value):
        if action.choices is not None and value not in action.choices:
            accepted = ', '.join(map(str, action.choices))
            name = action.option_strings[-1] if action.option_strings else action.metavar
            raise argparse.ArgumentError(
                action, f'expected {name} to be one of {accepted}, found {value!r}'
            )

    def error(self, message, *, hint=None):
        missing = 'the following arguments are required: '
        if message.startswith(missing):
            message = f'expected {message[len(missing) :]}, found missing arguments'
        elif match := re.fullmatch(r'argument (.+): ignored explicit argument (.+)', message):
            message = f'expected {match[1]} without a value, found {match[2]}'
        elif match := re.fullmatch(r'argument (.+?): (.+)', message):
            name, detail = match.groups()
            if detail == 'expected one argument':
                message = f'expected a value for {name}, found none'
            elif detail.startswith('invalid int value: '):
                value = detail.removeprefix('invalid int value: ')
                message = f'expected an integer for {name}, found {value}'
            elif detail.startswith('expected '):
                message = detail
            else:
                message = f'{name}, {detail}'
        self.print_usage(sys.stderr)
        raise SystemExit(
            fail(
                self.prog,
                message,
                hint=hint or f'run {self.prog} --help for the options.',
                as_json=self._as_json,
                status=EXIT_USAGE,
                error_id='usage',
            )
        )


def add_globals(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '-q',
        '--quiet',
        action='store_true',
        default=argparse.SUPPRESS,
        help='print nothing on standard error but errors',
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        default=argparse.SUPPRESS,
        help='print debug lines and the traceback of an error',
    )


def add_json(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--json', action='store_true', help='print one JSON object on standard output'
    )


def examples(*lines: str) -> str:
    return 'Examples\n' + '\n'.join(f'  {line}' for line in lines)


def _close_output() -> None:
    try:
        with open(os.devnull, 'wb') as sink:
            os.dup2(sink.fileno(), sys.stdout.fileno())
    except (OSError, ValueError):
        pass


def _is_closed_pipe(error: BaseException) -> bool:
    return isinstance(error, BrokenPipeError) or (
        sys.platform == 'win32' and isinstance(error, OSError) and error.errno == errno.EINVAL
    )


def guarded(prog: str, function: Callable[[], int | None], arguments: Sequence[str]) -> int:
    flags = options(arguments)
    try:
        try:
            status = function()
        except SystemExit as error:
            status = error.code
            if status is not None and not isinstance(status, int):
                status = fail(prog, str(status), as_json='--json' in flags)
        except Exception as error:
            if _is_closed_pipe(error):
                raise
            status = fail(
                prog,
                f'internal error, {type(error).__name__}, {error}',
                hint='rerun with --verbose and report the traceback.',
                as_json='--json' in flags,
                status=EXIT_INTERNAL,
                error_id='internal',
                values={'type': type(error).__name__},
            )
            if any(is_global(flag) and 'v' in flag for flag in flags):
                traceback.print_exc()
        sys.stdout.flush()
        return status or 0
    except OSError as error:
        if not _is_closed_pipe(error):
            raise
        _close_output()
        return 0
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED


def run_command(
    parser: Parser, execute: Callable[[argparse.Namespace], int], argv: list[str] | None
) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    return guarded(parser.prog, lambda: execute(parser.parse_args(arguments)), arguments)


def subcommands(tree: argparse.ArgumentParser):
    return (action for action in tree._actions if isinstance(action, argparse._SubParsersAction))
