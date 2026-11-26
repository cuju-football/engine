# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Controllers that send actions for the agents of a batch, and the matches they play."""

from importlib import import_module
from typing import Any

__all__ = [
    'Actions',
    'Controller',
    'ControllerError',
    'Match',
    'Observation',
    'Result',
    'Spec',
    'check',
    'load',
    'parts',
    'play',
    'play_batch',
    'to_gymnasium',
]

# These import PyTorch or the engine, so importing the package stays light.
_LAZY = {
    'Actions': '_contract',
    'Controller': '_contract',
    'ControllerError': '_contract',
    'Observation': '_contract',
    'Spec': '_contract',
    'load': '_load',
    'parts': '_check',
    'Match': '_match',
    'Result': '_match',
    'play': '_match',
    'play_batch': '_match',
    'check': '_check',
    'to_gymnasium': '_gym',
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        value = getattr(import_module(f'{__name__}.{_LAZY[name]}'), name)
        globals()[name] = value
        return value
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__() -> list[str]:
    return sorted(__all__)
