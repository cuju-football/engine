# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Batched Google Research Football (GRF) environments on the GPU or the CPU, bit for bit."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from typing import ClassVar

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
]

scenario_names.__module__ = __name__

_LAZY = {
    'BatchEnvironment': '_environment',
    'engine_status': 'engine_status',
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


class _RegisteringLoader:
    def __init__(self, loader, name):
        self._loader = loader
        self._name = name

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        _RegistrationFinder.waiting.discard(self._name)
        _register_ids(self._name)

    def __getattr__(self, name):
        return getattr(self._loader, name)


class _RegistrationFinder:
    waiting: ClassVar[set[str]] = set()
    finding: ClassVar[set[str]] = set()

    def find_spec(self, name, path=None, target=None):
        if name not in self.waiting or name in self.finding:
            return None
        self.finding.add(name)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            self.finding.discard(name)
        if spec is not None and spec.loader is not None:
            spec.loader = _RegisteringLoader(spec.loader, name)
        return spec


def _register_ids(module):
    try:
        from .env import _register_with
    except ImportError:
        return
    _register_with(module)


def _register_gymnasium_ids():
    if sys.modules.get('gymnasium') is not None:
        _register_ids('gymnasium')
    else:
        _RegistrationFinder.waiting.add('gymnasium')
    if not any(isinstance(finder, _RegistrationFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _RegistrationFinder())


_register_gymnasium_ids()


def _with_portions(path):
    import os

    folders = list(path)
    seen = {os.path.normcase(os.path.abspath(folder)) for folder in folders}
    for entry in sys.path:
        if not isinstance(entry, str):
            continue
        folder = os.path.abspath(os.path.join(entry, __name__))
        key = os.path.normcase(folder)
        if key in seen or os.path.isfile(os.path.join(folder, '__init__.py')):
            continue
        if os.path.isdir(folder):
            seen.add(key)
            folders.append(folder)
    return folders


__path__ = _with_portions(__path__)

from . import _plugins  # noqa: E402

_plugins.install_finder()
