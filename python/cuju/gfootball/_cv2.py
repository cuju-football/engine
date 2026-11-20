# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""OpenCV for GRF's Python, imported at the first use of one of its names.

GRF imports `cv2` at the top of `football_env_core.py`, `observation_processor.py`, `wrappers.py`
and `remote_football_env.py`, and uses it for frames, videos, pixel observations and `imshow`
alone. The port imports this module as `cv2` in its place, so `cuju.gfootball.env` imports without
OpenCV. The first name read from it imports the real `cv2` and serves that module's attribute.
Where OpenCV is missing, that read raises `ModuleNotFoundError: No module named 'cv2'`, the error of
GRF's own import, moved from the import to the first use.
"""

import importlib
from types import ModuleType

_module: ModuleType | None = None


def _cv2() -> ModuleType:
    """Returns the real `cv2`, imported once."""
    global _module
    if _module is None:
        _module = importlib.import_module('cv2')
    return _module


def __getattr__(name: str):
    if name.startswith('__') and name.endswith('__'):
        # Probes such as `hasattr(module, '__path__')` expect an `AttributeError` for a name that
        # is missing, so a special name gets one where OpenCV is missing.
        try:
            module = _cv2()
        except ImportError:
            raise AttributeError(f'module {__name__!r} has no attribute {name!r}') from None
        return getattr(module, name)
    return getattr(_cv2(), name)


def __dir__() -> list[str]:
    try:
        return sorted(set(globals()) | set(dir(_cv2())))
    except ImportError:
        return sorted(globals())
