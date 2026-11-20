# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""`cuju dump-to-txt`, GRF's script `dump_to_txt` with GRF's flags, which writes a dump as text.

It reads the dump with `cuju.dumps`, so dumps of GRF, of `cuju.env`, of a batch and of
`cuju.gfootball` print alike, and runs no code from them.

    cuju dump-to-txt --trace_file=/tmp/input.dump --output=/tmp/output.txt
"""

from __future__ import annotations

import argparse
import sys

from . import _cli_contract as contract
from . import _cli_parsers, dumps
from ._script_helpers import _dump_text, _grf_steps

__all__ = ['main']


def _run(arguments: argparse.Namespace) -> int:
    path = arguments.trace_file
    try:
        text = _dump_text(_grf_steps(dumps._read_cli_dump(path)), arguments.include_debug)
    except (OSError, ValueError, KeyError, TypeError) as error:
        detail = (error.strerror or str(error)).lower() if isinstance(error, OSError) else error
        return contract.fail(
            'cuju dump-to-txt',
            f'expected a readable dump at {contract.quote_path(path)}, found {detail}',
            hint='run cuju dump-to-txt --trace_file FILE --output PATH with a GRF or Cuju dump.',
            as_json=arguments.json,
            error_id='dump_to_txt.dump',
            values={'file': path},
        )
    try:
        with open(arguments.output, 'w') as handle:
            handle.write(text)
    except OSError as error:
        return contract.fail(
            'cuju dump-to-txt',
            f'expected a writable output file at {contract.quote_path(arguments.output)}, '
            f'found {error.strerror}',
            hint='run cuju dump-to-txt --trace_file FILE --output PATH with a writable path.',
            as_json=arguments.json,
            status=contract.EXIT_ENVIRONMENT,
            error_id='dump_to_txt.output',
            values={'output': arguments.output},
        )
    if arguments.json:
        contract.emit('engine.dump_to_txt/1', {'files': [arguments.output]})
    else:
        print(arguments.output)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Runs the converter on the command line."""
    return contract.run_command(_cli_parsers.dump_to_txt(), _run, argv)


if __name__ == '__main__':
    sys.exit(main())
