# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Records, reads, replays and converts matches of Cuju football."""

from importlib import import_module
from typing import Any

from .._version import __version__
from ._errors import RecordingError
from ._recording import Recording, read

__all__ = [
    'Recorder',
    'Recording',
    'RecordingError',
    '__version__',
    'from_grf_dump',
    'from_kaggle',
    'load',
    'read',
    'record',
]

# These need PyTorch, so importing the package stays light.
_LAZY = {
    'Recorder': '_recorder',
    'record': '_recorder',
    'from_grf_dump': '_grf',
    'from_kaggle': '_kaggle',
    'load': '_dataset',
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        value = getattr(import_module(f'{__name__}.{_LAZY[name]}'), name)
        globals()[name] = value
        return value
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
