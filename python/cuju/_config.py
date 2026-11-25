# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""The machine config, the file `engine.toml` that holds the settings that change only speed.

    [run]
    threads = 0
    debug = false

`CUJU_CONFIG` names the file, else it is `~/.config/cuju/engine.toml` on Linux and macOS, with
`XDG_CONFIG_HOME` in place of `~/.config` where set, and `%APPDATA%\\cuju\\engine.toml` on Windows.
A missing file leaves every key at its default. The file may hold the table `[run]` and nothing
else, and `[run]` may hold only the keys of `DEFAULTS`. This module loads without PyTorch.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from .errors import ConfigurationError

__all__ = ['DEFAULTS', 'path', 'resolve']

# The keys of `[run]` with their defaults, in the order that `cuju check` lists them.
DEFAULTS: dict[str, Any] = {
    'threads': 0,
    'debug': False,
    'allocator': 'cuda',
    'pin_threads': False,
    'numa': False,
    'pin_memory': 'auto',
    'render_threads': 0,
    'render_workers': 0,
    'log': False,
}
_VARIABLE = 'CUJU_CONFIG'


def path() -> str:
    """Returns the path of the machine config, which may name no file."""
    given = os.environ.get(_VARIABLE)
    if given:
        return given
    if os.name == 'nt':
        base = os.environ.get('APPDATA') or os.path.join(
            os.path.expanduser('~'), 'AppData', 'Roaming'
        )
    else:
        base = os.environ.get('XDG_CONFIG_HOME') or os.path.join(os.path.expanduser('~'), '.config')
    return os.path.join(base, 'cuju', 'engine.toml')


def _load(text: str, name: str) -> dict[str, Any]:
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ConfigurationError(f'expected a TOML file at {name!r}, found {error}') from None


def _check(key: str, value: Any, where: str) -> Any:
    """Returns `value` for `key`, or raises `ConfigurationError` naming `where`."""
    default = DEFAULTS[key]
    if key == 'allocator':
        if value not in ('cuda', 'torch'):
            raise ConfigurationError(
                f'expected allocator to be "cuda" or "torch" in {where}, found {value!r}'
            )
    elif key == 'pin_memory':
        if value != 'auto' and not isinstance(value, bool):
            raise ConfigurationError(
                f'expected pin_memory to be "auto", true or false in {where}, found {value!r}'
            )
    elif key == 'numa':
        # A node of each processor, as `0,0,1,1`, stands in for what the system reports.
        if not isinstance(value, bool) and not (isinstance(value, str) and _nodes(value)):
            raise ConfigurationError(
                f'expected numa to be true, false or the node of each processor such as '
                f'"0,0,1,1" in {where}, found {value!r}'
            )
    elif isinstance(default, bool):
        if not isinstance(value, bool):
            raise ConfigurationError(
                f'expected {key} to be true or false in {where}, found {value!r}'
            )
    elif isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigurationError(
            f'expected {key} to be a whole number, 0 or more, in {where}, found {value!r}'
        )
    return value


def _nodes(text: str) -> bool:
    parts = text.split(',')
    return all(part.strip().isdigit() for part in parts)


def read(file: str | None = None) -> dict[str, Any]:
    """Returns the `[run]` values that the machine config sets, empty for a missing file.

    Raises:
        ConfigurationError: If the file is no TOML, holds a table other than `[run]` or a key
            outside a table, or sets an unknown key or a value of the wrong kind.
    """
    file = path() if file is None else file
    try:
        with open(file, encoding='utf-8') as handle:
            text = handle.read()
    except FileNotFoundError:
        return {}
    except OSError as error:
        raise ConfigurationError(
            f'expected a readable machine config at {file!r}, found {error.strerror}'
        ) from None
    document = _load(text, file)
    for name, value in document.items():
        if name != 'run' or not isinstance(value, dict):
            raise ConfigurationError(f'expected only the table [run] in {file!r}, found {name!r}')
    values = document.get('run', {})
    for key, value in values.items():
        if key not in DEFAULTS:
            raise ConfigurationError(
                f'expected a key of [run] from {", ".join(DEFAULTS)} in {file!r}, found {key!r}'
            )
        _check(key, value, repr(file))
    return values


def resolve(overrides: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Returns each key of `[run]` with its value and where the value came from.

    Args:
        overrides: The `config=` dict of a call, which wins over the file.

    Returns:
        A plain dict from each key to `{'value': ..., 'source': ...}`, where the source is
        `default`, `file <path>` or `call`.
    """
    file = path()
    given = read(file)
    resolved = {key: {'value': value, 'source': 'default'} for key, value in DEFAULTS.items()}
    for key, value in given.items():
        resolved[key] = {'value': value, 'source': f'file {file}'}
    if overrides:
        if not isinstance(overrides, dict):
            raise ConfigurationError(
                f'expected config to be a dict of [run] keys, found {overrides!r}'
            )
        for key, value in overrides.items():
            if key not in DEFAULTS:
                raise ConfigurationError(
                    f'expected a key of [run] from {", ".join(DEFAULTS)} in config, found {key!r}'
                )
            resolved[key] = {'value': _check(key, value, 'config'), 'source': 'call'}
    return resolved


def values(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Returns the resolved value of each key of `[run]`."""
    return {key: entry['value'] for key, entry in resolve(overrides).items()}


def differing(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Returns the keys whose resolved value differs from the default."""
    return {key: value for key, value in values(overrides).items() if value != DEFAULTS[key]}


def machine_options(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Returns the speed options of `Batch` that the machine config and `overrides` resolve to.

    The key `pin_memory`, `auto` in the file by default, becomes true where PyTorch sees a CUDA
    GPU, and the batch applies it to a CPU batch alone.
    """
    resolved = values(overrides)
    pin = resolved['pin_memory']
    if pin == 'auto':
        import torch

        pin = torch.cuda.is_available()
    return {
        'threads': resolved['threads'],
        'debug': resolved['debug'],
        'allocator': resolved['allocator'],
        'pin_threads': resolved['pin_threads'],
        'numa': resolved['numa'],
        'pin_memory': pin,
        'render_threads': resolved['render_threads'],
        'render_workers': resolved['render_workers'],
        'log': resolved['log'],
    }
