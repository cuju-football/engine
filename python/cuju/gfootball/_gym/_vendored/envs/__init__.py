# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""The registry of the vendored gym, the names of gym 0.21.0's `gym/envs/__init__.py`.

gym's own module registers its bundled environments and loads the plugins of the entry point
group `gym.envs` at import. This one registers nothing, since the registry holds GRF's ids alone.
Plugins, written against an installed gym, register with that gym, so `load_env_plugins` stays a
function to call by hand.
"""

from cuju.gfootball._gym._vendored.envs.registration import (
    load_env_plugins,
    make,
    register,
    registry,
    spec,
)

__all__ = ['load_env_plugins', 'make', 'register', 'registry', 'spec']
