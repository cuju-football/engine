# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""Reads single fields of a GRF state, such as the joints of a player or the spin of the ball."""

from __future__ import annotations

import ctypes
import functools
import struct
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from .errors import ArgumentError, ChoiceError, LibraryError, StateError

__all__ = ['FRAMES', 'Field', 'gather', 'layout', 'read']

FRAMES = {'rest': 0, 'pitch': 1}

# GRF: the state names each type as C++'s `typeid` does, and stores it little-endian.
_DTYPES = {
    'f': ('<f4', ()),
    'i': ('<i4', ()),
    'j': ('<u4', ()),
    'b': ('u1', ()),
    'c': ('i1', ()),
    'm': ('<u8', ()),
    'N7blunted7Vector3E': ('<f4', (3,)),
    'N7blunted10QuaternionE': ('<f4', (4,)),
}


class Field(NamedTuple):
    """One field of the match in memory."""

    path: str
    offset: int
    size: int
    type: str


@functools.cache
def _library():
    from ._batch import library as _libraries

    failures = []
    for device in ('cpu', 'cuda'):
        try:
            library = _libraries._load_library(None, device)
        except Exception as error:
            failures.append(f'{device}: {error}')
            continue
        library.cuju_state_layout.restype = ctypes.c_void_p
        library.cuju_state_layout.argtypes = [ctypes.POINTER(ctypes.c_ulonglong)]
        library.cuju_state_gather.argtypes = [
            ctypes.c_char_p,
            ctypes.c_ulonglong,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        return library
    raise LibraryError('expected a library of cuju that loads, found ' + '; '.join(failures))


@functools.cache
def _fields() -> dict[str, Field]:
    library = _library()
    size = ctypes.c_ulonglong()
    pointer = library.cuju_state_layout(ctypes.byref(size))
    if not pointer:
        raise LibraryError(library.cuju_last_error().decode())
    fields = {}
    for line in ctypes.string_at(pointer, size.value).decode().splitlines():
        path, offset, length, kind = line.split('\t')
        fields[path] = Field(path, int(offset), int(length), kind)
    return fields


def layout() -> dict[str, Field]:
    """Returns every field of the match in memory, by path, in the order of GRF's state format."""
    return dict(_fields())


def _match_bytes(state: bytes) -> bytes:
    if len(state) < 4 or not 0 < struct.unpack_from('<i', state)[0] <= len(state) - 4:
        raise StateError(
            f'expected the bytes of a state: a pickle length, the pickle and the match, found '
            f'{len(state)} bytes that hold none'
        )
    (length,) = struct.unpack_from('<i', state)
    return state[4 + length :]


def _resolve(fields: Sequence[str | Field]) -> list[Field]:
    known = _fields()
    resolved = []
    for field in fields:
        if isinstance(field, Field):
            resolved.append(field)
        elif field in known:
            resolved.append(known[field])
        else:
            raise ArgumentError(f'expected a path of cuju.state.layout(), found {field!r}')
    return resolved


def gather(state: bytes, fields: Sequence[str | Field], frame: str = 'rest') -> bytes:
    """Loads a state and returns the bytes of the chosen fields, one after another."""
    if frame not in FRAMES:
        raise ChoiceError(f'expected the frame rest or pitch, found {frame!r}')
    match = _match_bytes(state)
    resolved = _resolve(fields)
    offsets = np.array([field.offset for field in resolved], np.uint64)
    sizes = np.array([field.size for field in resolved], np.uint32)
    out = ctypes.create_string_buffer(int(sizes.sum()))
    library = _library()
    if library.cuju_state_gather(
        match,
        len(match),
        FRAMES[frame],
        offsets.ctypes.data,
        sizes.ctypes.data,
        len(resolved),
        ctypes.addressof(out),
    ):
        raise StateError(library.cuju_last_error().decode())
    return out.raw


def read(state: bytes, fields: Sequence[str | Field], frame: str = 'rest') -> list[np.ndarray]:
    """Loads a state and returns the value of each chosen field as a numpy array."""
    resolved = _resolve(fields)
    data = gather(state, resolved, frame)
    values, position = [], 0
    for field in resolved:
        chunk = data[position : position + field.size]
        position += field.size
        dtype, shape = _DTYPES.get(field.type, (None, ()))
        # GRF: an enumeration is a C++ `enum`, four bytes on every platform it runs on.
        if dtype is None and field.size == 4 and field.type[:1].isdigit():
            dtype = '<i4'
        if dtype is not None and np.dtype(dtype).itemsize * int(np.prod(shape)) == field.size:
            values.append(np.frombuffer(chunk, dtype).reshape(shape).copy())
        else:
            values.append(np.frombuffer(chunk, np.uint8).copy())
    return values
