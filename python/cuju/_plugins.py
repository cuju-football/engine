# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""What cuju knows of the packages that install beside it, as `cuju.<name>`, and one hook.

A package that installs beside cuju puts its module into the folder `cuju`, and an editable
install of it keeps the module in its own `cuju` folder, which `import cuju` joins to the
package. Nothing here names such a package. A `cuju.<name>` that no finder has gets a message that
suggests the package `cuju-<name>`, from one finder at the end of `sys.meta_path`, which Python
asks only after every other finder has found nothing.

Every batch that cuju creates while `CUJU_BATCH_HOOKS` names hooks hands itself to each, and a hook
is the entry point of that name in the group `cuju.batch_hooks`, a function that takes the batch. A
process that sets nothing imports nothing and reads no entry points.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

# The variable that names the hooks of every batch of a process, separated by commas.
VARIABLE = 'CUJU_BATCH_HOOKS'
# The group of entry points that holds the hooks.
GROUP = 'cuju.batch_hooks'
_LOGGER = logging.getLogger('cuju')
# The hooks that a batch has already warned of as missing, once each.
_warned: set[str] = set()


def hint(module: str) -> str:
    """Returns the message that suggests the package of a module `cuju.<name>` that is missing."""
    package = 'cuju-' + module.partition('.')[2].partition('.')[0].replace('_', '-')
    return (
        f'No module named {module!r}. A package that installs beside cuju provides it, and '
        f'{package} may be that package: pip install {package}'
    )


class _Missing:
    """Suggests the package of `cuju.<name>` when it is imported and no other finder has it.

    It answers only for a module of one level below cuju whose name is public, so an import of a
    module deeper down fails through its package, and every other module that no finder has fails as
    it always does.
    """

    def find_spec(self, name: str, path: Any = None, target: Any = None) -> None:
        parts = name.split('.')
        if len(parts) == 2 and parts[0] == 'cuju' and not parts[1].startswith('_'):
            raise ModuleNotFoundError(hint(name), name=name)
        return None


def install_finder() -> None:
    """Adds the finder of a missing package at the end of `sys.meta_path`, once."""
    if not any(isinstance(finder, _Missing) for finder in sys.meta_path):
        sys.meta_path.append(_Missing())


def is_missing(error: BaseException) -> bool:
    """Returns whether an import failed because a module `cuju.<name>` itself is missing."""
    name = getattr(error, 'name', None) or ''
    return isinstance(error, ModuleNotFoundError) and name.count('.') == 1 and name[:5] == 'cuju.'


def requested() -> list[str]:
    """Returns the hooks that `CUJU_BATCH_HOOKS` names."""
    return [name.strip() for name in os.environ.get(VARIABLE, '').split(',') if name.strip()]


def on_batch_created(batch: Any) -> None:
    """Hands a new batch to each hook that `CUJU_BATCH_HOOKS` names, and never fails the batch.

    A hook that no package provides gives one warning through the logger `cuju` per process, and a
    hook that fails gives one warning for the batch, which then runs as it would without the hook.
    """
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
