# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import Any

VARIABLE = 'CUJU_BATCH_HOOKS'
GROUP = 'cuju.batch_hooks'
_LOGGER = logging.getLogger('cuju')
_warned: set[str] = set()


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
