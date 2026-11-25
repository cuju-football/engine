# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

__all__ = ['Batch']


def __getattr__(name: str) -> Any:
    if name == 'Batch':
        from .environment import Batch

        globals()['Batch'] = Batch
        return Batch
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
