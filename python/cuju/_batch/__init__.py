# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any

__all__ = ['Batch', 'unwrap']


def unwrap(batch: Any) -> Any:
    """Returns the private `Batch` of a `BatchEnvironment`, and a `Batch` itself."""
    inner = getattr(batch, '_batch', None)
    return batch if inner is None else inner


def __getattr__(name: str) -> Any:
    if name == 'Batch':
        from .environment import Batch

        globals()['Batch'] = Batch
        return Batch
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
