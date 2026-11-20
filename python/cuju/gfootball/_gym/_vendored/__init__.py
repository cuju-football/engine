# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""The core of gym 0.21.0 that `cuju.gfootball._gym` serves where no gym of GRF's version is found.

It holds the names of gym's own `gym/__init__.py` that GRF's Python reads, from gym's files
vendored next to it with their imports pointed inside this package. Its registry is its own, so
an id registered here reaches `make` of this package alone.
"""

from cuju.gfootball._gym._vendored.core import (
    ActionWrapper,
    Env,
    GoalEnv,
    ObservationWrapper,
    RewardWrapper,
    Wrapper,
)

# isort: split
# The wrappers subclass `Wrapper` as they import, so they follow `core`.
from cuju.gfootball._gym._vendored import envs, error, logger, spaces, version, wrappers
from cuju.gfootball._gym._vendored.envs import make, register, spec
from cuju.gfootball._gym._vendored.spaces import Space

__version__ = version.VERSION

__all__ = [
    'ActionWrapper',
    'Env',
    'GoalEnv',
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
