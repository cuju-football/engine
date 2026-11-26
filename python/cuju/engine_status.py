# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""Tells an intact engine from a damaged one, and prints the command that repairs it."""

import argparse
import os
import sys
from importlib import metadata

from . import _cli_contract as contract

DISTRIBUTION = 'cuju-engine'
REQUIRED = ('__init__.py', '__main__.py', '_version.py', os.path.join('_batch', '__init__.py'))


def _folder() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def missing() -> list[str]:
    """Returns the required files of the engine that are missing, empty when it is intact."""
    folder = _folder()
    return [name for name in REQUIRED if not os.path.isfile(os.path.join(folder, name))]


def intact() -> bool:
    """Returns whether the files of the engine that an upgrade of cuju 0.2.1 deletes are present."""
    return not missing()


def _installed_engine() -> tuple[str, str | None]:
    for name in (DISTRIBUTION, 'cuju-engine-cpu'):
        try:
            return name, metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return DISTRIBUTION, None


def installed_version() -> str | None:
    """Returns the installed engine version, or None for a source tree."""
    return _installed_engine()[1]


def _repair_command(name: str, version: str | None) -> str:
    pin = f'=={version}' if version else ''
    return f'pip install --force-reinstall --no-deps {name}{pin}'


def repair_command() -> str:
    """Returns the command that reinstalls the engine distribution at its installed version."""
    return _repair_command(*_installed_engine())


def parser() -> contract.Parser:
    """Builds the installation check parser."""
    args = contract.Parser(
        prog='cuju-repair',
        description='Check the engine installation and print the repair command if it is damaged.',
        epilog=contract.examples('cuju-repair', 'cuju-repair --json'),
    )
    contract.add_globals(args)
    contract.add_json(args)
    return args


def _run(arguments: argparse.Namespace) -> int:
    lost = missing()
    name, version = _installed_engine()
    repair = _repair_command(name, version) if lost else None
    if arguments.json:
        contract.emit(
            'engine.repair/1',
            {'intact': not lost, 'version': version, 'missing': lost, 'repair_command': repair},
        )
    if lost:
        return contract.fail(
            'cuju-repair', f'the engine is damaged, it lacks {", ".join(lost)}', hint=repair
        )
    if not arguments.json:
        print(f'the engine is intact, {name} {version or "of a source tree"}')
    return 0


def main(arguments: list[str] | None = None) -> int:
    """Prints the engine state and returns its exit status."""
    return contract.run_command(parser(), _run, arguments)


if __name__ == '__main__':
    sys.exit(main())
