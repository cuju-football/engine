# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
from pathlib import Path
from typing import Any

from . import _cli_contract as contract
from .errors import (
    ConfigurationError,
    DeviceError,
    LibraryError,
    MissingFileError,
)
from .recording._errors import RecordingError
from .recording._recording import Recording, load

_ENDS = {0: 'running', 1: 'terminated', 2: 'time limit'}


def _unreadable(arguments: argparse.Namespace, error: Exception, path: str) -> int:
    detail = (error.strerror or str(error)).lower() if isinstance(error, OSError) else error
    return contract.fail(
        f'cuju recording {arguments.recording_command}',
        f'expected a recording at {contract.quote_path(path)}, found {detail}',
        hint='run cuju recording show FILE with a file that cuju recording wrote.',
        as_json=arguments.json,
        error_id='recording.unreadable',
        values={'file': path},
    )


def _score(match: Recording) -> list[int] | None:
    score = match.step.get('score')
    return None if score is None else [int(goals) for goals in score[-1]]


def _shape(array: Any) -> dict[str, Any]:
    return {'dtype': str(array.dtype), 'shape': list(array.shape)}


def _show(arguments: argparse.Namespace) -> int:
    try:
        match, _ = load(arguments.file)
    except (OSError, RecordingError) as error:
        return _unreadable(arguments, error, arguments.file)
    score = _score(match)
    if arguments.json:
        contract.emit(
            'recording.show/1',
            {
                'file': arguments.file,
                'header': match.header,
                'end': _ENDS[match.end],
                'score': score,
                'state': match.state is not None,
                'step': {name: _shape(array) for name, array in match.step.items()},
                'array': {name: _shape(array) for name, array in match.array.items()},
            },
        )
        return 0
    header = match.header
    lines = [
        f'{arguments.file}',
        f'  scenario   {match.scenario}',
        f'  agents     {match.left_agents} left, {match.right_agents} right',
        f'  steps      {match.steps}, {_ENDS[match.end]}',
    ]
    if score is not None:
        lines.append(f'  score      {score[0]} to {score[1]}')
    if match.teams:
        lines.append(f'  teams      {match.teams[0]}, {match.teams[1]}')
    lines.append(f'  state      {"kept" if match.state is not None else "none"}')
    lines.append(f'  engine     {header["engine"]}')
    for title, table in (('step', match.step), ('array', match.array)):
        if table:
            lines.append(f'  {title}')
            width = max(len(name) for name in table)
            lines.extend(
                f'    {name:<{width}}  {array.dtype} {list(array.shape)}'
                for name, array in table.items()
            )
    for key, value in match.provenance.items():
        lines.append(f'  provenance {key} {value}')
    print('\n'.join(lines))
    return 0


def _backend_error(arguments: argparse.Namespace, error: Exception) -> int:
    usage = isinstance(error, ConfigurationError)
    return contract.fail(
        f'cuju recording {arguments.recording_command}',
        str(error).partition(';')[0],
        hint=f'run cuju recording {arguments.recording_command} with --device cpu.',
        as_json=arguments.json,
        status=contract.EXIT_USAGE if usage else contract.EXIT_ENVIRONMENT,
        error_id='recording.device' if usage else 'recording.backend_unavailable',
        values={'device': arguments.device},
    )


def _check(arguments: argparse.Namespace) -> int:
    try:
        match, unknown = load(arguments.file)
    except (OSError, RecordingError) as error:
        return _unreadable(arguments, error, arguments.file)
    if unknown:
        return contract.fail(
            'cuju recording check',
            f'expected only the members of the format, found {", ".join(unknown)}',
            hint='remove the members that the format does not name.',
            as_json=arguments.json,
            error_id='recording.members',
            values={'members': unknown},
        )
    difference = None
    if arguments.replay:
        try:
            match.replay(device=arguments.device)
        except (ConfigurationError, DeviceError, LibraryError, MissingFileError) as error:
            return _backend_error(arguments, error)
        except RecordingError as error:
            difference = str(error)
    if arguments.json:
        contract.emit(
            'recording.check/1',
            {
                'file': arguments.file,
                'steps': match.steps,
                'replayed': bool(arguments.replay),
                'difference': difference,
            },
        )
    elif difference is not None:
        print(difference)
    elif arguments.replay:
        print(f"The replay equals the recording's {match.steps} steps.")
    else:
        print(f'The recording holds {match.steps} valid steps.')
    return contract.EXIT_NO if difference is not None else 0


def _convert(arguments: argparse.Namespace) -> int:
    from .recording import _files, _grf, _kaggle

    folder = Path(arguments.output)
    converted, failed = [], []
    for path in arguments.paths:
        try:
            data = _files.read_bytes(path)
            kind = 'kaggle' if data.lstrip()[:1] == b'{' else 'grf-dump'
            match = (
                _kaggle.from_kaggle(data)
                if kind == 'kaggle'
                else _grf.from_grf_dump(data, device=arguments.device)
            )
            target = folder / (Path(path).stem + '.cjr')
            if target.exists() and not arguments.overwrite:
                raise RecordingError(
                    f'expected no file at {contract.quote_path(target)}, found one, so pass '
                    '--overwrite to replace it'
                )
            match.write(target)
        except (DeviceError, LibraryError) as error:
            return _backend_error(arguments, error)
        except (OSError, RecordingError) as error:
            detail = (error.strerror or str(error)).lower() if isinstance(error, OSError) else error
            failed.append({'source': path, 'error': str(detail)})
            if not arguments.json:
                print(f'{path}: {detail}', file=sys.stderr)
            continue
        converted.append({'source': path, 'file': str(target), 'kind': kind, 'steps': match.steps})
        if not arguments.json:
            print(f'{path} to {target}, {match.steps} steps')
    if arguments.json:
        contract.emit('recording.convert/1', {'converted': converted, 'failed': failed})
    return contract.EXIT_NO if failed else 0


_RUN = {'show': _show, 'check': _check, 'convert': _convert}


def _run(arguments: argparse.Namespace) -> int:
    if arguments.recording_command is None:
        arguments._command_parser.error(
            'expected a command from show, check and convert, found none'
        )
    return _RUN[arguments.recording_command](arguments)
