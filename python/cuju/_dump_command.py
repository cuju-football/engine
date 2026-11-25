# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import sys

from . import _cli_contract as contract
from . import _cli_parsers, dumps
from ._script_helpers import _dump_text, _grf_steps

__all__ = ['main']


def _run(arguments: argparse.Namespace) -> int:
    path = arguments.file
    try:
        text = _dump_text(_grf_steps(dumps._read_cli_dump(path)), True)
    except (OSError, ValueError, KeyError, TypeError) as error:
        detail = (error.strerror or str(error)).lower() if isinstance(error, OSError) else error
        return contract.fail(
            'cuju dump',
            f'expected a readable dump at {contract.quote_path(path)}, found {detail}',
            hint='run cuju dump FILE with a GRF or cuju dump.',
            as_json=arguments.json,
            error_id='dump.dump',
            values={'file': path},
        )
    if arguments.output is None:
        if arguments.json:
            contract.emit('engine.dump/1', {'text': text})
        else:
            sys.stdout.write(text)
        return 0
    try:
        with open(arguments.output, 'w') as handle:
            handle.write(text)
    except OSError as error:
        return contract.fail(
            'cuju dump',
            f'expected a writable output file at {contract.quote_path(arguments.output)}, '
            f'found {error.strerror}',
            hint='run cuju dump FILE -o OUT with a writable path.',
            as_json=arguments.json,
            status=contract.EXIT_ENVIRONMENT,
            error_id='dump.output',
            values={'output': arguments.output},
        )
    if arguments.json:
        contract.emit('engine.dump/1', {'files': [arguments.output]})
    else:
        print(arguments.output)
    return 0


def main(argv: list[str] | None = None) -> int:
    return contract.run_command(_cli_parsers.dump(), _run, argv)


if __name__ == '__main__':
    sys.exit(main())
