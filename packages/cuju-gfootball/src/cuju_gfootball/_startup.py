# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Serves `cuju.gfootball` and `cuju.gfootball_engine` under GRF's module names.

Python's `site` runs `cuju_gfootball.pth` at startup, which imports this module. It puts one
finder at the front of `sys.meta_path` and imports nothing else. The finder decides at the first
import of a name of GRF or of the port whether the alias is active, and it stays out of the way for
every other name.

The alias maps `gfootball`, its submodules, `gfootball_engine` and `_gameplayfootball` onto the
modules of cuju, so each name of GRF is the same module object as its name in cuju. The classes of
those modules carry GRF's module names, so a pickle of them, a dump or a state, names GRF's modules
and opens in GRF. Where a `gfootball` of any other origin can be imported, the alias activates
nothing and warns once, so GRF's own install keeps its names.
"""

import importlib.machinery
import importlib.util
import sys
import threading

_ROOTS = {
    'gfootball': 'cuju.gfootball',
    'gfootball_engine': 'cuju.gfootball_engine',
    '_gameplayfootball': 'cuju.gfootball_engine._gameplayfootball',
}
_ENGINE = 'cuju.gfootball_engine._gameplayfootball'
_PREFIXES = ('gfootball', '_gameplayfootball', 'cuju.gfootball')
_lock = threading.RLock()
# `None` until the first import of a name of GRF or of the port decides, then the reason the alias
# stays inactive, or the empty string where it is active.
_reason = None
_warned = False


def _target(name):
    """Returns the module of cuju that serves GRF's module `name`, or `None`."""
    if name in _ROOTS:
        return _ROOTS[name]
    if name.startswith(('gfootball.', 'gfootball_engine.')):
        return f'cuju.{name}'
    return None


def _grf_name(name):
    """Returns GRF's name of the module of cuju `name`, or `None` for a module outside GRF's tree.

    The modules of cuju's own inside the port, whose names start with an underscore, have none.
    """
    if name == _ENGINE:
        return '_gameplayfootball'
    for root in ('cuju.gfootball', 'cuju.gfootball_engine'):
        if name == root or name.startswith(f'{root}.'):
            grf = name[len('cuju.') :]
            return None if any(part.startswith('_') for part in grf.split('.')) else grf
    return None


def _origin(spec):
    if spec.origin not in (None, 'namespace'):
        return spec.origin
    return next(iter(spec.submodule_search_locations or ()), spec.name)


def _decide():
    """Returns why the alias stays inactive, or the empty string where it activates."""
    for name in _ROOTS:
        for finder in sys.meta_path:
            if isinstance(finder, _Finder) or not hasattr(finder, 'find_spec'):
                continue
            try:
                spec = finder.find_spec(name, None)
            except Exception:
                spec = None
            if spec is not None:
                return (
                    f'`{name}` imports from {_origin(spec)}, an install of GRF or of another '
                    f'package, which cuju-gfootball leaves in place'
                )
    from importlib import metadata

    try:
        found = metadata.distribution('gfootball')
    except metadata.PackageNotFoundError:
        found = None
    if found is not None:
        return (
            f"GRF's distribution `gfootball` {found.version} is installed, which "
            f'cuju-gfootball leaves in place'
        )
    try:
        cuju = importlib.util.find_spec('cuju')
    except (ImportError, ValueError):
        cuju = None
    if cuju is None:
        return 'cuju is not installed'
    return ''


def reason():
    """Returns why the alias stays inactive, or the empty string where it is active."""
    global _reason
    with _lock:
        if _reason is None:
            _reason = _decide()
        return _reason


def _warn():
    global _warned
    with _lock:
        if _warned:
            return
        _warned = True
    import warnings

    warnings.warn(
        f"cuju-gfootball is inactive: {reason()}. GRF's names import what is installed under "
        'them, and cuju.gfootball keeps its own names. Uninstall one of the two to choose.',
        RuntimeWarning,
        stacklevel=3,
    )


def _rename(module):
    """Gives the classes that `module` defines GRF's module name, as GRF's pickles hold it."""
    name = _grf_name(module.__name__)
    for value in list(vars(module).values()):
        if isinstance(value, type) and value.__module__ == module.__name__:
            try:
                value.__module__ = name
            except (AttributeError, TypeError):
                pass


class _AliasLoader:
    """Loads one of GRF's names as the module of cuju that serves it."""

    def __init__(self, target):
        self._target = target
        self._spec = None

    def create_module(self, spec):
        module = importlib.import_module(self._target)
        self._spec = module.__spec__
        return module

    def exec_module(self, module):
        # The import system set the alias's spec on cuju's module, which keeps its own.
        module.__spec__ = self._spec


class _RenamingLoader:
    """Loads a module of the port with its own loader, then gives its classes GRF's names."""

    def __init__(self, loader):
        self._loader = loader

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        _rename(module)

    def __getattr__(self, name):
        return getattr(self._loader, name)


class _Finder:
    """Finds GRF's names as cuju's modules and the port's modules with their classes renamed."""

    def find_spec(self, name, path=None, target=None):
        if not name.startswith(_PREFIXES):
            return None
        alias = _target(name)
        if alias is None and _grf_name(name) is None:
            return None
        if reason():
            if alias is not None:
                _warn()
            return None
        if alias is not None:
            return self._alias_spec(name, alias)
        return self._renamed_spec(name, path, target)

    @staticmethod
    def _alias_spec(name, alias):
        found = importlib.util.find_spec(alias)
        if found is None:
            return None
        spec = importlib.machinery.ModuleSpec(name, _AliasLoader(alias), origin=found.origin)
        if found.submodule_search_locations is not None:
            spec.submodule_search_locations = list(found.submodule_search_locations)
        return spec

    def _renamed_spec(self, name, path, target):
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, 'find_spec'):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                if spec.loader is not None and hasattr(spec.loader, 'exec_module'):
                    spec.loader = _RenamingLoader(spec.loader)
                return spec
        return None


def install():
    """Puts the finder at the front of `sys.meta_path` once, and renames the port's loaded modules.

    The modules of the port load after the finder where `site` reads `cuju_gfootball.pth`. Under
    `python -S`, or with an import of the port before `cuju_gfootball`, some loaded first.
    """
    with _lock:
        if not any(isinstance(finder, _Finder) for finder in sys.meta_path):
            sys.meta_path.insert(0, _Finder())
    loaded = [module for name, module in list(sys.modules.items()) if _grf_name(name)]
    if loaded and not reason():
        for module in loaded:
            _rename(module)


install()
