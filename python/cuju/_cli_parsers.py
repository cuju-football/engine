# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from ._cli_contract import Parser, add_globals, add_json, examples

SUMMARIES = {
    'check': 'check the engine against known match outputs',
    'replay': 'check a GRF dump bit for bit, without rendering',
    'dump': 'write a dump as text, without running its code',
    'help': 'describe the commands of the family, or one of them',
    'completion': 'print the completion script of a shell',
}
SHELLS = ('bash', 'zsh', 'fish', 'powershell')


def _make(group: Any, name: str, description: str, epilog: str) -> Parser:
    if group is None:
        parser = Parser(prog=f'cuju {name}', description=description, epilog=epilog)
    else:
        parser = group.add_parser(
            name, prog=f'cuju {name}', help=SUMMARIES[name], description=description, epilog=epilog
        )
    parser.set_defaults(_command_parser=parser)
    return parser


def check(group: Any = None) -> Parser:
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
    case = parser.add_argument(
        '--case', action='append', metavar='NAME', help='play only the case NAME, may repeat'
    )
    case.completion = _check_case_names
    parser.add_argument('--list', action='store_true', help='list the cases and play none')
    add_json(parser)
    add_globals(parser)
    return parser


def _check_case_names() -> tuple[str, ...]:
    from .check import load_cases

    return tuple(load_cases(long=True))


def replay(group: Any = None) -> Parser:
    parser = _make(
        group,
        'replay',
        'Check a GRF dump bit for bit, without rendering. The replay plays the dump again and '
        'compares each step with the recorded step, apart from wall-clock time. The command '
        'reads dumps written by GRF and cuju, and runs no code from them.',
        examples(
            'cuju replay /tmp/dumps/episode_done.dump',
            'cuju replay /tmp/dumps/episode_done.dump --device cuda --json',
        ),
    )
    parser.add_argument('file', metavar='FILE', help='the dump to replay')
    device = parser.add_argument(
        '--device',
        default='cpu',
        metavar='DEVICE',
        help='cpu, cuda, cuda:N or auto, cpu by default',
    )
    device.completion = ('auto', 'cpu', 'cuda')
    add_json(parser)
    add_globals(parser)
    return parser


def dump(group: Any = None) -> Parser:
    parser = _make(
        group,
        'dump',
        'Write a dump as text, without running its code. The command reads dumps written by GRF '
        'and cuju. It prints the text, or writes it to the file of the option -o.',
        examples(
            'cuju dump /tmp/input.dump',
            'cuju dump /tmp/input.dump -o /tmp/output.txt --json',
        ),
    )
    parser.add_argument('file', metavar='FILE', help='the dump to convert')
    parser.add_argument(
        '-o',
        '--output',
        metavar='OUT',
        help='the text file to write, the standard output without it',
    )
    add_json(parser)
    add_globals(parser)
    return parser


def help_command(group: Any = None) -> Parser:
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
    'dump': dump,
    'help': help_command,
    'completion': completion,
}
