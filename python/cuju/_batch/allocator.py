# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ctypes

import torch

ALLOCATE = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
RELEASE = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)


class TorchAllocator:
    def __init__(self, device: torch.device) -> None:
        stream = torch.cuda.default_stream(device)
        errors: list[BaseException] = []

        def allocate(size, _context):
            try:
                return torch.cuda.caching_allocator_alloc(size, device, stream)
            except Exception as error:
                errors.append(error)
                return None

        def release(pointer, _context):
            torch.cuda.caching_allocator_delete(pointer)

        self.allocate = ALLOCATE(allocate)
        self.release = RELEASE(release)
        self.errors = errors

    def pointers(self) -> tuple[int, int]:
        return (
            ctypes.cast(self.allocate, ctypes.c_void_p).value,
            ctypes.cast(self.release, ctypes.c_void_p).value,
        )
