# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""Loads a tree of controllers from a file, a folder, a dictionary or an entry string."""

import os
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from ..errors import ConfigurationError
from . import client
from ._contract import Controller

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover, Python 3.10
    import tomli as tomllib

GROUP = 'cuju.agents.parts'
SCHEMA = 1
_PATH_KEYS = frozenset({'path', 'paths', 'file', 'folder', 'dir', 'directory', 'python', 'cwd'})
_PATH_SUFFIXES = ('_path', '_file', '_folder', '_dir')
_DOTTED = re.compile(r'\w+(\.\w+)*')


def standard() -> dict[str, Any]:
    """Returns the standard controllers by their reserved names."""
    from ._process import Process
    from .standard import NAMES

    return {**NAMES, 'process': Process}


def installed() -> dict[str, Any]:
    """Returns the installed parts by short name, refusing a clash with a reserved name."""
    reserved = set(standard())
    found: dict[str, Any] = {}
    for entry in metadata.entry_points(group=GROUP):
        if entry.name in reserved:
            raise ConfigurationError(
                f'expected a part name other than the reserved {entry.name!r}, found the '
                f'package {entry.dist.name if entry.dist else entry.value} providing it'
            )
        if entry.name in found:
            raise ConfigurationError(f'expected one part named {entry.name!r}, found two')
        found[entry.name] = entry
    return found


class _Names(dict):
    def __init__(self, table: dict[str, Any], entries: dict[str, Any]) -> None:
        super().__init__(table)
        self._entries = entries
        self.update(dict.fromkeys(entries))

    def __getitem__(self, name: str) -> Any:
        value = super().__getitem__(name)
        if value is None:
            value = self._entries[name].load()
            self[name] = value
        return value


def names() -> dict[str, Any]:
    """Returns every short name that a tree may use, loading an installed part when used."""
    return _Names(standard(), installed())


def _absolute(value: Any, base: Path | None) -> Any:
    if not isinstance(value, str) or os.path.isabs(value) or base is None:
        return value
    return str((base / value).resolve())


def _resolve(node: Any, base: Path | None) -> Any:
    if isinstance(node, list):
        return [_resolve(item, base) for item in node]
    if not isinstance(node, dict):
        return node
    found = {}
    for key, value in node.items():
        if key in _PATH_KEYS or key.endswith(_PATH_SUFFIXES):
            value = (
                [_absolute(item, base) for item in value]
                if isinstance(value, list)
                else _absolute(value, base)
            )
        elif key == 'inner':
            value = _resolve(value, base)
        found[key] = value
    return found


def _read(path: Path) -> dict[str, Any]:
    try:
        with open(path, 'rb') as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigurationError(f'expected a TOML file at {str(path)!r}, found {error}') from None
    schema = data.pop('agents', SCHEMA)
    if schema != SCHEMA:
        raise ConfigurationError(f'expected agents = {SCHEMA} in {str(path)!r}, found {schema!r}')
    return data


def node_of(spec: Any, base: Path | None = None) -> dict[str, Any] | Controller:
    """Returns the node or the controller that a form of a tree names, with absolute paths."""
    if isinstance(spec, Controller) or (hasattr(spec, 'act') and not isinstance(spec, type)):
        return spec
    if isinstance(spec, dict):
        return _resolve(dict(spec), base)
    if callable(spec) and not isinstance(spec, (str, os.PathLike)):
        return {'entry': 'function', 'function': spec}
    if isinstance(spec, os.PathLike):
        spec = os.fspath(spec)
    if not isinstance(spec, str):
        raise ConfigurationError(
            f'expected a path, an entry, a dictionary or a controller, found {spec!r}'
        )
    if spec == 'builtin' or spec.startswith('builtin@'):
        difficulty = None if spec == 'builtin' else _difficulty(spec[len('builtin@') :])
        return {'entry': 'builtin', 'difficulty': difficulty}
    path = Path(spec) if base is None else (base / spec)
    if path.exists():
        path = path.resolve()
        if path.is_dir() or path.suffix == '.py':
            return {'entry': 'kaggle', 'path': str(path)}
        return _resolve(_read(path), path.parent)
    if ':' in spec and not os.path.isabs(spec) and _DOTTED.fullmatch(spec.partition(':')[0]):
        return {'entry': spec}
    if spec in standard() or spec in installed():
        return {'entry': spec}
    raise ConfigurationError(
        f'expected an existing path, a name from {", ".join(sorted(names()))} or module:object, '
        f'found {spec!r}'
    )


def _difficulty(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise ConfigurationError(
            f'expected a difficulty from 0 to 1 after @, found {text!r}'
        ) from None


def load(spec: Any, base: str | os.PathLike[str] | None = None) -> Controller:
    """Builds the controller that a TOML file, a folder, a dictionary or an entry names."""
    node = node_of(spec, None if base is None else Path(base))
    if isinstance(node, Controller) or not isinstance(node, dict):
        return node
    return _build(node)


def _build(node: dict[str, Any]) -> Controller:
    try:
        return client.build(node, names())
    except client.ConfigurationError as error:
        raise ConfigurationError(str(error)) from None
