# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""gym for GRF's Python, the installed gym wherever GRF runs on it, else gym 0.21's own core.

GRF requires `gym<=0.21.0`, and it runs on every gym up to 0.24, which `import gfootball` imports
and registers its ids with. Gym 0.25 runs it with numpy 1, and with numpy 2 its step checker reads
`np.bool8`, which numpy 2 removed, so every step fails. Gym 0.26 took gymnasium's interface, and
GRF's `reset` fails there. The port imports this module as `gym`, and it chooses once, at its
import. An installed gym on which GRF runs, and that imports, serves as itself, so the port's
environments are instances of its `gym.Env`, its `gym.make` finds GRF's ids, and its warnings are
GRF's. On every other gym the module serves the core of gym 0.21.0 vendored in `_vendored`, under
gym's MIT license, whose registry belongs to the port alone. `VENDORED` tells which of the two
serves.
"""

import importlib
import importlib.metadata
import re
from types import ModuleType


def _version(name: str) -> tuple[int, int] | None:
    """Returns the major and minor version of an installed distribution, or `None`."""
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    match = re.match(r'(\d+)\.(\d+)', version)
    return (int(match[1]), int(match[2])) if match else None


def _grf_runs_on(gym: tuple[int, int], numpy: tuple[int, int] | None) -> bool:
    """Returns whether GRF's Python runs on a gym of version `gym` with a numpy of `numpy`."""
    if gym <= (0, 24):
        return True
    return gym[:2] == (0, 25) and numpy is not None and numpy < (2, 0)


def _installed_gym() -> ModuleType | None:
    """Returns the installed gym where GRF runs on it and it imports, else `None`."""
    gym = _version('gym')
    if gym is None or not _grf_runs_on(gym, _version('numpy')):
        return None
    # A gym that fails to import, with any error, leaves the choice to the vendored core.
    try:
        return importlib.import_module('gym')
    except Exception:
        return None


_source = _installed_gym()
VENDORED = _source is None
if _source is None:
    from . import _vendored as _source

Env = _source.Env
Wrapper = _source.Wrapper
ObservationWrapper = _source.ObservationWrapper
ActionWrapper = _source.ActionWrapper
RewardWrapper = _source.RewardWrapper
Space = _source.Space
spaces = _source.spaces
envs = _source.envs
wrappers = _source.wrappers
make = _source.make
spec = _source.spec
register = _source.register
logger = _source.logger
error = _source.error
__version__ = _source.__version__

__all__ = [
    'VENDORED',
    'ActionWrapper',
    'Env',
    'ObservationWrapper',
    'RewardWrapper',
    'Space',
    'Wrapper',
    'envs',
    'error',
    'logger',
    'make',
    'register',
    'spaces',
    'spec',
    'wrappers',
]
