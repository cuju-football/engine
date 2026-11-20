# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""Lightweight argument parsers for the engine commands."""

from __future__ import annotations

import argparse
from typing import Any

from ._cli_contract import Parser, add_globals, add_json, examples

# The summary that `cuju help` and the root parser give each command of the engine.
SUMMARIES = {
    'check': 'check the engine against known match outputs',
    'replay': 'replay a dump and check that it plays the same',
    'dump-to-txt': 'write a dump as readable text',
    'help': 'describe the commands of the family, or one of them',
    'completion': 'print the completion script of a shell',
}
SHELLS = ('bash', 'zsh', 'fish', 'powershell')


def _make(group: Any, name: str, description: str, epilog: str) -> Parser:
    """Returns the parser of the command `name`, a subparser of `group` or a stand-alone one."""
    if group is None:
        parser = Parser(prog=f'cuju {name}', description=description, epilog=epilog)
    else:
        parser = group.add_parser(
            name, prog=f'cuju {name}', help=SUMMARIES[name], description=description, epilog=epilog
        )
    parser.set_defaults(_command_parser=parser)
    return parser


def check(group: Any = None) -> Parser:
    """Returns the parser of `cuju check`, as a subparser of `group` if given."""
    parser = _make(
        group,
        'check',
        'Check the engine against known match outputs. The check verifies the package data '
        'and compares the matches played on each selected backend with the expected digests.',
        examples(
            'cuju check',
            'cuju check --device cpu',
            'cuju check --case 5_vs_5/simple115v2 --json',
            'cuju check --list',
        ),
    )
    parser.add_argument(
        '--device',
        choices=['all', 'cpu', 'cuda'],
        default='all',
        metavar='DEVICE',
        help='check all, cpu or cuda, all by default',
    )
    parser.add_argument(
        '--long', action='store_true', help='also play the long cases, which take several minutes'
    )
    parser.add_argument(
        '--case', action='append', metavar='NAME', help='play only the case NAME, may repeat'
    )
    parser.add_argument('--list', action='store_true', help='list the cases and play none')
    add_json(parser)
    add_globals(parser)
    return parser


def _flag(text: str) -> bool:
    """Returns the value of a boolean flag written as absl reads it."""
    lowered = text.lower()
    if lowered in ('true', 't', '1'):
        return True
    if lowered in ('false', 'f', '0'):
        return False
    raise argparse.ArgumentTypeError(f'expected true or false, found {text!r}')


def replay(group: Any = None) -> Parser:
    """Returns the parser of `cuju replay`, with the flags of GRF's script."""
    parser = _make(
        group,
        'replay',
        'Replay a dump and check that it plays the same. The replay compares each step with '
        'the recorded step, apart from wall-clock time.',
        examples(
            'cuju replay --trace_file /tmp/dumps/episode_done.dump',
            'cuju replay --trace_file /tmp/dumps/episode_done.dump --device cpu --json',
        ),
    )
    parser.add_argument('--trace_file', required=True, metavar='FILE', help='the dump to replay')
    parser.add_argument(
        '--fps',
        type=int,
        default=None,
        metavar='N',
        help='frames per second, the frame rate of the dump by default',
    )
    parser.add_argument(
        '--device',
        default='cpu',
        metavar='DEVICE',
        help='cpu, cuda, cuda:N or auto, cpu by default',
    )
    add_json(parser)
    add_globals(parser)
    return parser


def dump_to_txt(group: Any = None) -> Parser:
    """Returns the parser of `cuju dump-to-txt`, with the flags of GRF's script."""
    parser = _make(
        group,
        'dump-to-txt',
        'Write a dump as readable text. The command reads dumps written by GRF and cuju.',
        examples(
            'cuju dump-to-txt --trace_file /tmp/input.dump --output /tmp/output.txt --json',
            'cuju dump-to-txt --trace_file in.dump --output out.txt --noinclude_debug',
        ),
    )
    parser.add_argument('--trace_file', required=True, metavar='FILE', help='the dump to convert')
    parser.add_argument('--output', required=True, metavar='FILE', help='the text file to write')
    parser.add_argument(
        '--include_debug',
        nargs='?',
        const=True,
        default=True,
        type=_flag,
        metavar='NAME',
        help='write the debug information of each step, true by default',
    )
    parser.add_argument(
        '--noinclude_debug',
        dest='include_debug',
        action='store_false',
        help='leave out the debug information of each step',
    )
    add_json(parser)
    add_globals(parser)
    return parser


def help_command(group: Any = None) -> Parser:
    """Returns the parser of `cuju help`."""
    parser = _make(
        group,
        'help',
        'Describe the commands of the family. Without words it lists the commands of the engine '
        'and then those of each package installed beside it. With the words of a command it '
        'prints the help of that command, so cuju help arena team add equals cuju arena team '
        'add --help.',
        examples('cuju help', 'cuju help check', 'cuju help arena team add'),
    )
    parser.add_argument(
        'words',
        nargs='*',
        metavar='NAME',
        help='a command and the words below it, none to list the commands',
    )
    add_globals(parser)
    return parser


def completion(group: Any = None) -> Parser:
    """Returns the parser of `cuju completion`."""
    parser = _make(
        group,
        'completion',
        'Print the completion script of a shell, bash, zsh, fish or powershell. The script is '
        'static and holds the commands and options of every package installed now, so run this '
        'command again after installing or removing a package. Redirect the output into the '
        'folder of completion scripts of the shell, or source it from the startup file.',
        examples(
            'cuju completion bash',
            'cuju completion zsh',
            'cuju completion fish',
            'cuju completion powershell',
        ),
    )
    parser.add_argument('shell', choices=SHELLS, help='the shell to complete for', metavar='NAME')
    add_globals(parser)
    return parser


BUILDERS = {
    'check': check,
    'replay': replay,
    'dump-to-txt': dump_to_txt,
    'help': help_command,
    'completion': completion,
}
