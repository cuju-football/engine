# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""Reads many recordings into concatenated arrays, for offline learning."""

import json
import os
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import _files
from ._errors import RecordingError

__all__ = ['load']

_STEP_FIELDS = ('actions', 'rewards', 'done')
_DEFAULT = _STEP_FIELDS


def _paths(paths: Any) -> list[Path]:
    items = [paths] if isinstance(paths, (str, os.PathLike)) else list(paths)
    found: list[Path] = []
    for item in items:
        path = Path(os.fspath(item))
        found.extend(sorted(path.glob('*.cjr')) if path.is_dir() else [path])
    if not found:
        raise RecordingError('expected at least one recording, found none')
    return found


def _read(path: Path, fields: Sequence[str]) -> dict[str, np.ndarray]:
    found: dict[str, np.ndarray] = {}
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            _files._checked(json.loads(archive.read(_files.HEADER)))
            for field in fields:
                member = f'{field}.npy'
                if member not in names:
                    raise RecordingError(f'expected the field {field!r}, found none')
                with archive.open(member) as handle:
                    found[field] = np.lib.format.read_array(handle, allow_pickle=False)
    except RecordingError as error:
        raise RecordingError(f'{path}, {error}') from None
    except (OSError, zipfile.BadZipFile, ValueError, EOFError) as error:
        raise RecordingError(f'expected a recording at {str(path)!r}, found {error}') from error
    return found


def load(paths: Any, fields: Sequence[str] | None = None) -> dict[str, np.ndarray]:
    """Reads recordings into arrays that join the episodes, decoding only the fields asked for.

    Args:
        paths: A file, a folder of `.cjr` files, or a list of them.
        fields: The members to read, from `actions`, `rewards` and `done`, and `step/NAME` and
            `array/NAME` for the tables of a recording. By default `actions`, `rewards` and `done`.

    Returns:
        A dictionary of arrays with the episodes one after the other along the first axis, and
        `episode_start`, the first row of each episode with the total at the end, so episode `i`
        is the rows from `episode_start[i]` to `episode_start[i + 1]`. The tables `step/NAME` and
        `array/NAME` hold one row more than the steps of each episode, so with them `table_start`
        gives their offsets in the same way.
    """
    chosen = list(_DEFAULT if fields is None else fields)
    for field in chosen:
        if field not in _STEP_FIELDS and not field.startswith(('step/', 'array/')):
            raise RecordingError(
                f'expected a field from {", ".join(_STEP_FIELDS)}, step/NAME or array/NAME, '
                f'found {field!r}'
            )
    files = _paths(paths)
    columns: dict[str, list[np.ndarray]] = {field: [] for field in chosen}
    starts, table_starts = [0], [0]
    wants_tables = any(field not in _STEP_FIELDS for field in chosen)
    for path in files:
        needed = list(dict.fromkeys([*chosen, 'done']))
        episode = _read(path, needed)
        steps = len(episode['done'])
        for field in chosen:
            columns[field].append(episode[field])
        starts.append(starts[-1] + steps)
        table_starts.append(table_starts[-1] + steps + 1)
    joined: dict[str, np.ndarray] = {}
    for field, parts in columns.items():
        try:
            joined[field] = np.concatenate(parts, axis=0)
        except ValueError as error:
            raise RecordingError(
                f'expected episodes of one shape for {field!r}, found {error}'
            ) from None
    joined['episode_start'] = np.asarray(starts, dtype=np.int64)
    if wants_tables:
        joined['table_start'] = np.asarray(table_starts, dtype=np.int64)
    return joined
