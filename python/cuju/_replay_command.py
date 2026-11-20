# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""`cuju replay`, GRF's script `replay` without rendering, which plays a dump again and checks it.

    cuju replay --trace_file=/tmp/dumps/episode_done_20261026-120000000000.dump

It replays cuju's and GRF's dumps alike and compares every step with the dump's bit for bit, apart
from the wall-clock time. It prints the first difference and exits 1 if there is one, and
`check(trace_file)` does the same from Python.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from . import _cli_contract as contract
from . import _cli_parsers, _replays, dumps
from ._library_path import _backend
from .errors import ChoiceError, DeviceError, LibraryError, MissingFileError

__all__ = ['check', 'main']


def check(trace_file: str, device: Any = 'cpu', threads: int = 1) -> _replays.ReplayResult:
    """Plays a dump again and compares every step with the dump's.

    Args:
        trace_file: A dump that starts at the start of its episode.
        device: The device that plays the replay. Every device plays the same match.
        threads: The CPU threads of the replay.

    Returns:
        `(steps, ended, difference)`, the steps found equal, whether the replay's episode ended
        with the dump's last step, and the first difference as a text, or `None` if the replay
        equals the dump.

    Raises:
        ValueError: If the dump holds no configuration, does not start at the start of its
            episode, or holds an action outside its action set.
    """
    return _replays.replay(trace_file, device=device, threads=threads)


def _run(arguments: argparse.Namespace) -> int:
    path = arguments.trace_file
    try:
        if arguments.device != 'auto':
            _backend(arguments.device)
    except ChoiceError:
        arguments._command_parser.error(
            f'expected --device to be cpu, cuda, cuda:N or auto, found {arguments.device!r}'
        )
    try:
        steps = dumps._read_cli_dump(path)
        if arguments.fps is not None:
            ticks = steps[0]['debug']['config']['physics_steps_per_frame']
            fps = arguments.fps
            if fps <= 0 or fps != ticks or 100 // fps != ticks:
                arguments._command_parser.error(
                    f'expected a frame rate for the dump of {ticks} ticks a step, found {fps}'
                )
        result = _replays._replay_steps(steps, path, device=arguments.device)
    except (DeviceError, LibraryError, MissingFileError) as error:
        expected = str(error).partition(';')[0]
        return contract.fail(
            'cuju replay',
            expected,
            hint=f'run cuju replay --trace_file {contract.quote_path(path)} --device cpu.'
            if arguments.device != 'cpu'
            else 'run cuju check --device cpu to check the installation.',
            as_json=arguments.json,
            status=contract.EXIT_ENVIRONMENT,
            error_id='replay.backend_unavailable',
            values={'device': arguments.device},
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
        detail = (error.strerror or str(error)).lower() if isinstance(error, OSError) else error
        return contract.fail(
            'cuju replay',
            f'expected a replayable dump at {contract.quote_path(path)}, found {detail}',
            hint='run cuju replay --trace_file FILE with a dump from the start of an episode.',
            as_json=arguments.json,
            error_id='replay.dump',
            values={'file': path},
        )
    if arguments.json:
        contract.emit(
            'engine.replay/1',
            {'steps': result.steps, 'ended': result.ended, 'difference': result.difference},
        )
        return contract.EXIT_NO if result.difference is not None else 0
    if result.difference is not None:
        print(f'The replay differs from the dump at {result.difference}')
        return 1
    ending = 'and ends the episode there' if result.ended else 'before the end of the episode'
    print(f"The replay equals the dump's {result.steps} steps, {ending}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    return contract.run_command(_cli_parsers.replay(), _run, argv)


if __name__ == '__main__':
    sys.exit(main())
