# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import shlex
import sys
from pathlib import Path
from typing import Any

from . import _cli_contract as contract
from .errors import ConfigurationError, DeviceError, Error, LibraryError


def _fail(arguments: argparse.Namespace, error: Exception, name: str, status: int) -> int:
    return contract.fail(
        f'cuju agents {arguments.agents_command}',
        ' '.join(str(error).split()),
        hint='run cuju agents parts to list the names that a tree may use.',
        as_json=getattr(arguments, 'json', False),
        status=status,
        error_id=f'agents.{name}',
        values={},
    )


def _isolated(text: str, in_process: bool) -> Any:
    from pathlib import Path

    from .agents._load import node_of

    node = node_of(text)
    # A folder or a Python file runs in a process of its own, unless asked otherwise.
    path = Path(text)
    if not in_process and isinstance(node, dict) and (path.is_dir() or path.suffix == '.py'):
        return {'entry': 'process', 'inner': node}
    return node


def _check(arguments: argparse.Namespace) -> int:
    from .agents._check import check

    if arguments.child is None and arguments.path is None:
        arguments._command_parser.error('expected PATH or the option --command, found neither')
    try:
        if arguments.child is None:
            from .agents._load import node_of

            node = node_of(arguments.path)
            path = Path(arguments.path)
            wanted = path.is_dir() or path.suffix == '.py'
            process = arguments.python or arguments.process or (wanted and not arguments.in_process)
            report = check(
                node,
                arguments.scenario,
                arguments.steps,
                arguments.twice,
                process=process,
                device=arguments.device,
            )
        else:
            report = check(
                None,
                arguments.scenario,
                arguments.steps,
                arguments.twice,
                command=shlex.split(arguments.child, posix=os.name != 'nt'),
                device=arguments.device,
            )
    except (ConfigurationError, DeviceError, LibraryError) as error:
        return _fail(arguments, error, 'load', contract.EXIT_USAGE)
    if arguments.json:
        contract.emit('agents.check/1', report.to_dict())
    else:
        for row in report.timing:
            line = f'{row["side"]}: {row["calls"]} calls, load {row["load_seconds"]:.2f} s'
            if row['calls']:
                line += ''.join(
                    f', {name} {row[name + "_seconds"] * 1000:.1f} ms'
                    for name in ('first', 'min', 'mean', 'max')
                )
            print(line)
        for failure in report.failures:
            print(f'{failure["side"] or "load"}: {failure["reason"]}, {failure["message"]}')
        print('The controller plays.' if report.ok else 'The controller does not play.')
    return 0 if report.ok else contract.EXIT_NO


def _play(arguments: argparse.Namespace) -> int:
    from .agents._match import play_batch

    try:
        left = _isolated(arguments.a, arguments.in_process)
        right = _isolated(arguments.b, arguments.in_process)
        result = play_batch(
            left,
            right,
            scenario=arguments.scenario,
            environments=min(arguments.episodes, arguments.environments),
            episodes=arguments.episodes,
            seed=arguments.seed,
            device=arguments.device,
            record=bool(arguments.record),
        )
    except (ConfigurationError, DeviceError, LibraryError, Error) as error:
        return _fail(arguments, error, 'play', contract.EXIT_USAGE)
    if arguments.record and result.recordings:
        result.recordings[0].write(arguments.record)
    summary = result.summary()
    if result.error is not None:
        summary['error'] = ' '.join(str(result.error).split())
    if arguments.json:
        summary['score'] = result.score.tolist()
        contract.emit('agents.play/1', summary)
    else:
        low, high = summary['win_rate_interval']
        print(
            f'{summary["episodes"]} episodes, {summary["wins"]} wins, {summary["draws"]} draws, '
            f'{summary["losses"]} losses, goal difference {summary["goal_difference"]}, win rate '
            f'{summary["win_rate"]:.3f} from {low:.3f} to {high:.3f}'
        )
        if result.error is not None:
            print(f'stopped by an error, {summary["error"]}', file=sys.stderr)
    return contract.EXIT_NO if result.error is not None else 0


def _parts(arguments: argparse.Namespace) -> int:
    from .agents._check import parts

    try:
        found = parts()
    except ConfigurationError as error:
        return _fail(arguments, error, 'parts', contract.EXIT_ENVIRONMENT)
    if arguments.json:
        contract.emit('agents.parts/1', {'parts': found})
        return 0
    width = max(len(row['name']) for row in found)
    for row in found:
        print(f'{row["name"]:<{width}}  {row["kind"]:<9}  {row["summary"]}'.rstrip())
    return 0


def _client(arguments: argparse.Namespace) -> int:
    from .agents._process import source

    if arguments.output:
        with open(arguments.output, 'wb') as handle:
            handle.write(source())
        return 0
    sys.stdout.buffer.write(source())
    sys.stdout.buffer.flush()
    return 0


_RUN = {'check': _check, 'play': _play, 'parts': _parts, 'client': _client}


def _run(arguments: argparse.Namespace) -> int:
    if arguments.agents_command is None:
        arguments._command_parser.error(
            'expected a command from check, play, parts and client, found none'
        )
    return _RUN[arguments.agents_command](arguments)
