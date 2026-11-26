# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

import ctypes
import warnings

import torch

from ..errors import _from_library


def _free_device_memory(device, allocator):
    try:
        free, _ = torch.cuda.mem_get_info(device)
    except RuntimeError:
        return None
    if allocator == 'torch':
        free += torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    return free


_LOCAL_MEMORY_HELD = set()


def _local_memory_bytes(library, device):
    if device.type != 'cuda' or device.index in _LOCAL_MEMORY_HELD:
        return 0
    local = ctypes.c_ulonglong()
    with torch.cuda.device(device):
        if library.cuju_local_memory_bytes(ctypes.byref(local)) != 0:
            raise _from_library(library.cuju_last_error().decode())
    return local.value


def _warn_memory(requested, free, device, local=0):
    if requested <= free:
        return
    kernels = ''
    if local:
        kernels = f', {local / 2**20:.1f} MiB of it for the local memory of the kernels,'
    warnings.warn(
        f'the batch requested {requested / 2**20:.1f} MiB of {device}{kernels} and the GPU had '
        f'{free / 2**20:.1f} MiB free before its creation. On Windows and WSL the WDDM driver may '
        'then place the batch in shared system memory, which runs many times slower and raises '
        'no error. Use fewer environments',
        stacklevel=3,
    )
