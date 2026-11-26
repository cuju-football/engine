# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from ._cli_contract import Parser, add_globals, add_json, examples

SUMMARIES = {
    'check': 'check the engine against known match outputs',
    'replay': 'check a GRF dump bit for bit, without rendering',
    'dump': 'write a dump as text, without running its code',
    'recording': 'show, check and convert recordings of matches',
    'agents': 'check and play controllers, and list their parts',
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


def recording(group: Any = None) -> Parser:
    parser = _make(
        group,
        'recording',
        'Show, check and convert recordings of matches. A recording is one episode in one file '
        'with the extension cjr, which holds the state, the actions and the rewards.',
        examples(
            'cuju recording show match.cjr',
            'cuju recording check match.cjr --replay',
            'cuju recording convert episode_done.dump -o recordings',
        ),
    )
    add_globals(parser)
    commands = parser.add_subparsers(dest='recording_command', metavar='COMMAND')

    def child(name: str, summary: str, description: str, epilog: str) -> Parser:
        found = commands.add_parser(name, help=summary, description=description, epilog=epilog)
        found.set_defaults(_command_parser=found)
        return found

    def device_option(found: Parser) -> None:
        option = found.add_argument(
            '--device',
            default='cpu',
            metavar='DEVICE',
            help='cpu, cuda, cuda:N or auto, cpu by default',
        )
        option.completion = ('auto', 'cpu', 'cuda')

    show = child(
        'show',
        'print what a recording holds',
        'Print what a recording holds. The output gives the header, the number of steps, the '
        'score at the end and the names of the step fields and arrays.',
        examples('cuju recording show match.cjr', 'cuju recording show match.cjr --json'),
    )
    show.add_argument('file', metavar='FILE', help='the recording to show')
    add_json(show)
    add_globals(show)

    check = child(
        'check',
        'validate a recording and replay it',
        'Validate every member of a recording. With the option --replay the command also plays '
        'the state and actions in the installed engine and compares each stored step, and exits '
        'with status 1 when the recording does not reproduce.',
        examples(
            'cuju recording check match.cjr',
            'cuju recording check match.cjr --replay --device cpu --json',
        ),
    )
    check.add_argument('file', metavar='FILE', help='the recording to check')
    check.add_argument('--replay', action='store_true', help='also replay it in the engine')
    device_option(check)
    add_json(check)
    add_globals(check)

    convert = child(
        'convert',
        'convert GRF dumps and Kaggle episodes',
        'Convert GRF dumps and Kaggle episodes to recordings. A dump is played again in the '
        'engine and converted only when every step equals the dump. A Kaggle episode keeps the '
        'steps that it holds and no state. Each recording is written to the folder under the '
        'name of its source and the extension cjr.',
        examples(
            'cuju recording convert episode_done.dump -o recordings',
            'cuju recording convert a.dump b.json -o recordings --device cpu --json',
        ),
    )
    convert.add_argument(
        'paths', nargs='+', metavar='PATH', help='the dumps and episodes to convert'
    )
    convert.add_argument(
        '-o', '--output', required=True, metavar='DIR', help='the folder to write to'
    )
    device_option(convert)
    convert.add_argument('--overwrite', action='store_true', help='replace a file that exists')
    add_json(convert)
    add_globals(convert)
    return parser


def agents(group: Any = None) -> Parser:
    parser = _make(
        group,
        'agents',
        'Check and play controllers. A controller sends the actions of the agents of a batch. '
        'It can be a folder with a main.py in the form of Kaggle, a TOML file, or any entry of '
        'the form module:object.',
        examples(
            'cuju agents check submission --scenario 5_vs_5',
            'cuju agents play submission builtin@0.5 --scenario 5_vs_5 --episodes 10',
            'cuju agents parts',
        ),
    )
    add_globals(parser)
    commands = parser.add_subparsers(dest='agents_command', metavar='COMMAND')

    def child(name: str, summary: str, description: str, epilog: str) -> Parser:
        found = commands.add_parser(name, help=summary, description=description, epilog=epilog)
        found.set_defaults(_command_parser=found)
        return found

    def common(found: Parser) -> None:
        found.add_argument(
            '--scenario',
            required=True,
            metavar='SCENARIO',
            help='the scenario to play',
        )
        found.add_argument('--device', default='cpu', metavar='DEVICE', help='cpu, cuda or auto')
        add_json(found)
        add_globals(found)

    check = child(
        'check',
        'play short matches and report what fails',
        'Play short matches with a controller on each side and report each failure with its '
        'reason, the actions taken and the time of its calls. A folder or a Python file runs in '
        'a process of its own, unless the option --in-process is given. The exit status is 0 '
        'when the controller plays and 1 when it does not.',
        examples(
            'cuju agents check submission --scenario 5_vs_5',
            'cuju agents check submission --scenario 5_vs_5 --twice --steps 50',
            "cuju agents check --command './child --flag' --scenario 5_vs_5",
        ),
    )
    check.add_argument('path', nargs='?', metavar='PATH', help='the folder, file or entry to check')
    check.add_argument('--steps', type=int, default=10, metavar='N', help='the steps of each match')
    check.add_argument('--twice', action='store_true', help='play twice and compare the actions')
    check.add_argument('--process', action='store_true', help='run any tree in a process')
    check.add_argument(
        '--in-process', action='store_true', help='run a Kaggle folder in this process'
    )
    check.add_argument(
        '--python', default=None, metavar='PATH', help='the interpreter of the process'
    )
    check.add_argument(
        '--command',
        dest='child',
        default=None,
        metavar='NAME',
        help='check a child that speaks the protocol, started by this command line',
    )
    common(check)

    play = child(
        'play',
        'play two controllers against each other',
        'Play episodes of two controllers, the first on the left and the second on the right, '
        'and print the wins, draws and losses of the left side with the interval of its win '
        'rate. A side is a folder, a file, an entry or builtin with an optional difficulty '
        'such as builtin@0.7.',
        examples(
            'cuju agents play submission builtin --scenario 5_vs_5 --episodes 10',
            'cuju agents play a.toml b.toml --scenario 5_vs_5 --seed 3 --json',
        ),
    )
    play.add_argument('a', metavar='A', help='the controller of the left side')
    play.add_argument('b', metavar='B', help='the controller of the right side')
    play.add_argument('--episodes', type=int, default=1, metavar='N', help='the episodes to play')
    play.add_argument(
        '--environments', type=int, default=64, metavar='N', help='the environments at most'
    )
    play.add_argument('--seed', type=int, default=0, metavar='N', help='the seed of the batch')
    play.add_argument('--record', default=None, metavar='FILE', help='write the first episode')
    play.add_argument(
        '--in-process', action='store_true', help='run Kaggle folders in this process'
    )
    common(play)

    parts = child(
        'parts',
        'list the names that a tree may use',
        'List the standard controllers and the parts that installed packages provide. A name '
        'in the column on the left is the entry of a node, and the group cuju.agents.parts '
        'gives an installed part its short name.',
        examples('cuju agents parts', 'cuju agents parts --json'),
    )
    add_json(parts)
    add_globals(parts)

    client = child(
        'client',
        'print the file that a child process runs',
        'Print the source of client.py, the one file that runs a controller in another Python '
        'environment, which needs only numpy. Copy it there, or pipe it into a container.',
        examples('cuju agents client', 'cuju agents client -o client.py'),
    )
    client.add_argument('-o', '--output', metavar='OUT', help='write the file here')
    add_globals(client)
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
    'recording': recording,
    'agents': agents,
    'help': help_command,
    'completion': completion,
}
