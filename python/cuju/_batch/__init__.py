# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""The implementation of the batch: `Batch`, with every option of the library.

`cuju.BatchEnvironment` is the public class and maps onto `Batch`. The engine's commands and the
packages of the family create their batches here. `Batch` loads on first use, so that importing
this package starts without PyTorch.
"""

from __future__ import annotations

from typing import Any

__all__ = ['Batch']


def __getattr__(name: str) -> Any:
    if name == 'Batch':
        from .environment import Batch

        globals()['Batch'] = Batch
        return Batch
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
