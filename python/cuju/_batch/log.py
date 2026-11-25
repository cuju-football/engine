# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

LOGGER = logging.getLogger('cuju')


class _StandardError(logging.StreamHandler):
    @property
    def stream(self):
        return sys.stderr

    @stream.setter
    def stream(self, value):
        pass


def enabled(override: bool | None = None) -> bool:
    if override is not None:
        return bool(override)
    from .. import _config

    return bool(_config.values()['log'])


def _handler():
    LOGGER.setLevel(logging.INFO)
    if LOGGER.handlers or logging.getLogger().handlers:
        return
    handler = _StandardError()
    handler.setFormatter(logging.Formatter('cuju: %(message)s'))
    LOGGER.addHandler(handler)


def info(message: str) -> None:
    if enabled():
        _handler()
        LOGGER.info(message)


def start(
    device: str, torch_device: torch.device | None = None, log: bool | None = None
) -> tuple[float, int | None] | None:
    if not enabled(log):
        return None
    _handler()
    free = None
    if device == 'cuda':
        import torch

        if torch.cuda.is_available():
            index = None if torch_device is None else torch_device.index
            free = torch.cuda.mem_get_info(index)[0]
    return time.perf_counter(), free


def created(batch, assets, library, started) -> None:
    if started is None:
        return
    began, free = started
    seconds = time.perf_counter() - began
    if batch.device == 'cuda':
        import torch

        index = batch._torch_device.index
        major, minor = torch.cuda.get_device_capability(index)
        place = (
            f'cuda:{index} ({torch.cuda.get_device_name(index)}, compute capability '
            f'{major}.{minor})'
        )
        if free is not None:
            taken = (free - torch.cuda.mem_get_info(index)[0]) / 2**20
            place += f', about {taken:.0f} MiB of device memory'
    else:
        place = f'the CPU with {batch.threads} threads'
    LOGGER.info(
        f'created {batch.environment_count} environments of {assets} on {place} through '
        f'{library} in {seconds:.2f} s: {batch.agents} agents, {batch.players} players a team, '
        f'{batch._representation}'
    )
