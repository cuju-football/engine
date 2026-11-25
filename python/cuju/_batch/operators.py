# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
import itertools
import weakref

import torch

_BATCHES: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
_KEYS = itertools.count(1)


def register(batch) -> int:
    key = next(_KEYS)
    _BATCHES[key] = batch
    return key


def _batch(key):
    batch = _BATCHES.get(key)
    if batch is None:
        raise RuntimeError('the batch of this compiled step is gone')
    return batch


AVAILABLE = hasattr(torch.library, 'custom_op')

if AVAILABLE:

    @torch.library.custom_op('cuju::step', mutates_args=('outputs',))
    def step(batch: int, actions: torch.Tensor, outputs: list[torch.Tensor]) -> None:
        """Steps the batch of key `batch` with `actions` and writes its output tensors `outputs`."""
        _batch(batch)._operator_step(actions, outputs)

    @step.register_fake
    def _step_fake(batch, actions, outputs):
        return None

    @torch.library.custom_op('cuju::reset_done', mutates_args=('outputs',))
    def reset_done(batch: int, outputs: list[torch.Tensor]) -> None:
        """Resets the ended environments of the batch of key `batch` and writes `outputs`."""
        _batch(batch)._operator_reset_done(outputs)

    @reset_done.register_fake
    def _reset_done_fake(batch, outputs):
        return None
