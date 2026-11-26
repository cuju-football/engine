# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
import torch

_PAGE = 4096


def register(ranges: list[tuple[int, int]]) -> list[int]:
    runtime = torch.cuda.cudart()
    registered: list[int] = []
    try:
        for pointer, size in ranges:
            status = runtime.cudaHostRegister(pointer, (size + _PAGE - 1) // _PAGE * _PAGE, 0)
            if int(status) != 0:
                raise RuntimeError(
                    f'cudaHostRegister refused {size} bytes of the batch, with error {int(status)}'
                )
            registered.append(pointer)
    except BaseException:
        unregister(registered)
        raise
    return registered


def unregister(pointers: list[int]) -> None:
    runtime = torch.cuda.cudart()
    for pointer in pointers:
        runtime.cudaHostUnregister(pointer)
