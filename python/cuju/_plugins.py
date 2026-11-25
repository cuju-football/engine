# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import sys
from typing import Any

VARIABLE = 'CUJU_BATCH_HOOKS'
GROUP = 'cuju.batch_hooks'
_LOGGER = logging.getLogger('cuju')
_warned: set[str] = set()


def hint(module: str) -> str:
    package = 'cuju-' + module.partition('.')[2].partition('.')[0].replace('_', '-')
    return (
        f'No module named {module!r}. A package that installs beside cuju provides it, and '
        f'{package} may be that package: pip install {package}'
    )


class _Missing:
    def find_spec(self, name: str, path: Any = None, target: Any = None) -> None:
        parts = name.split('.')
        if len(parts) == 2 and parts[0] == 'cuju' and not parts[1].startswith('_'):
            raise ModuleNotFoundError(hint(name), name=name)
        return None


def install_finder() -> None:
    if not any(isinstance(finder, _Missing) for finder in sys.meta_path):
        sys.meta_path.append(_Missing())


def is_missing(error: BaseException) -> bool:
    name = getattr(error, 'name', None) or ''
    return isinstance(error, ModuleNotFoundError) and name.count('.') == 1 and name[:5] == 'cuju.'


def requested() -> list[str]:
    return [name.strip() for name in os.environ.get(VARIABLE, '').split(',') if name.strip()]


def on_batch_created(batch: Any) -> None:
    names = requested()
    if not names:
        return
    from importlib import metadata

    found = {entry.name: entry for entry in metadata.entry_points(group=GROUP)}
    for name in names:
        entry = found.get(name)
        if entry is None:
            if name not in _warned:
                _warned.add(name)
                _LOGGER.warning(
                    'cuju: %s names the hook %r, which no installed package provides. '
                    'Every batch runs without it.',
                    VARIABLE,
                    name,
                )
            continue
        try:
            entry.load()(batch)
        except Exception as error:
            _LOGGER.warning(
                'cuju: the hook %r gave the batch nothing, %s: %s',
                name,
                type(error).__name__,
                error,
            )
