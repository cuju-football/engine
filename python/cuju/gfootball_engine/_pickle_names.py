# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""Translates the module names inside the pickle of a state between cuju and GRF, unpickling none.

A state of `cuju.gfootball` pickles GRF's classes under `cuju.gfootball`, and a state of GRF under
`gfootball`. The engine stores GRF's names, so states move between GRF and cuju either way. The
rewrite walks the opcodes, replaces the module names of `GLOBAL` and `INST` and every string that
is exactly such a name, and keeps the frames, whose lengths it updates. A name that is no module of
the port stays as it is.
"""

from __future__ import annotations

import functools
import os
import pickletools
import struct

_PACKAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENGINE = ('cuju.gfootball_engine._gameplayfootball', '_gameplayfootball')
_ENGINE_PACKAGE = ('cuju.gfootball_engine', 'gfootball_engine')
_STRINGS = ('SHORT_BINUNICODE', 'BINUNICODE', 'BINUNICODE8', 'UNICODE')


@functools.lru_cache(maxsize=1)
def _names():
    """Returns the module names of the port and of the engine, cuju's mapped to GRF's."""
    names = dict((_ENGINE, _ENGINE_PACKAGE))
    root = os.path.join(_PACKAGE, 'gfootball')
    for folder, directories, files in os.walk(root):
        directories[:] = sorted(name for name in directories if name.isidentifier())
        relative = os.path.relpath(folder, _PACKAGE).replace(os.sep, '.')
        names[f'cuju.{relative}'] = relative
        for name in files:
            stem, extension = os.path.splitext(name)
            if extension == '.py' and stem != '__init__' and stem.isidentifier():
                names[f'cuju.{relative}.{stem}'] = f'{relative}.{stem}'
    return names


@functools.lru_cache(maxsize=1)
def _reverse():
    return {grf: cuju for cuju, grf in _names().items()}


def to_grf(data: bytes) -> bytes:
    """Returns the pickle `data` with cuju's module names of the port turned into GRF's."""
    return _translate(data, _names())


def from_grf(data: bytes) -> bytes:
    """Returns the pickle `data` with GRF's module names turned into cuju's."""
    return _translate(data, _reverse())


def _string(kind: str, value: str) -> bytes:
    """Encodes `value` with the operation `kind` that held the name, as the pickler would."""
    if kind == 'UNICODE':
        return b'V' + value.encode('raw-unicode-escape') + b'\n'
    encoded = value.encode('utf-8', 'surrogatepass')
    if kind == 'SHORT_BINUNICODE' and len(encoded) < 256:
        return b'\x8c' + bytes([len(encoded)]) + encoded
    if kind == 'BINUNICODE8':
        return b'\x8d' + struct.pack('<Q', len(encoded)) + encoded
    return b'X' + struct.pack('<I', len(encoded)) + encoded


def _translate(data: bytes, mapping: dict) -> bytes:
    if not data:
        return data
    try:
        operations = list(pickletools.genops(data))
    except Exception:
        # Bytes that hold no pickle reach the engine as they are, as GRF's do.
        return data
    if not any(_rewrites(opcode.name, argument, mapping) for opcode, argument, _ in operations):
        return data
    # Each piece is the bytes of one operation, and each frame holds the pieces up to its end.
    pieces = []
    frames = []
    frame_end = None
    for index, (opcode, argument, position) in enumerate(operations):
        end = operations[index + 1][2] if index + 1 < len(operations) else len(data)
        raw = data[position:end]
        if frame_end is not None and position >= frame_end:
            frame_end = None
        if opcode.name == 'FRAME':
            frames.append([len(pieces), 0])
            frame_end = end + argument
            pieces.append(None)
            continue
        replaced = _rewrites(opcode.name, argument, mapping)
        if replaced is not None:
            raw = replaced
        if frame_end is not None:
            frames[-1][1] += len(raw)
        pieces.append(raw)
    for index, size in frames:
        pieces[index] = b'\x95' + struct.pack('<Q', size)
    return b''.join(pieces)


def _rewrites(name, argument, mapping):
    """Returns the new bytes of an operation that names a mapped module, or `None`."""
    if name in ('GLOBAL', 'INST'):
        module, _, attribute = argument.partition(' ')
        if module in mapping:
            code = b'c' if name == 'GLOBAL' else b'i'
            return code + mapping[module].encode() + b'\n' + attribute.encode() + b'\n'
        return None
    if name in _STRINGS and isinstance(argument, str) and argument in mapping:
        return _string(name, mapping[argument])
    return None
