# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Batched Google Research Football (GRF) environments on the GPU or the CPU, bit for bit."""

import importlib
import os
import sys

from . import errors
from ._assets import scenario_names
from ._version import __version__
from .errors import Error

__all__ = [
    'BatchEnvironment',
    'Error',
    '__version__',
    'engine_status',
    'errors',
    'scenario_names',
    'types',
]

scenario_names.__module__ = __name__

_LAZY = {
    'BatchEnvironment': '_environment',
    'engine_status': 'engine_status',
    'types': 'types',
}


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = importlib.import_module(f'.{module}', __name__)
    if name != module:
        value = getattr(value, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


def _extend_path(path: list[str]) -> list[str]:
    # What `pkgutil.extend_path` does, without its imports of `re` and `typing`, which cost more
    # than the rest of `import cuju`: a folder `cuju` on `sys.path` joins the package.
    seen = {os.path.normcase(os.path.abspath(folder)) for folder in path}
    for entry in sys.path:
        if not isinstance(entry, str):
            continue
        folder = os.path.abspath(os.path.join(entry, __name__))
        key = os.path.normcase(folder)
        if key not in seen and os.path.isdir(folder):
            seen.add(key)
            path.append(folder)
    return path


__path__ = _extend_path(list(__path__))
