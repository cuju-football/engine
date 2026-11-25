# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

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
    return _translate(data, _names())


def from_grf(data: bytes) -> bytes:
    return _translate(data, _reverse())


def _string(kind: str, value: str) -> bytes:
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
        return data
    if not any(_rewrites(opcode.name, argument, mapping) for opcode, argument, _ in operations):
        return data
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
    if name in ('GLOBAL', 'INST'):
        module, _, attribute = argument.partition(' ')
        if module in mapping:
            code = b'c' if name == 'GLOBAL' else b'i'
            return code + mapping[module].encode() + b'\n' + attribute.encode() + b'\n'
        return None
    if name in _STRINGS and isinstance(argument, str) and argument in mapping:
        return _string(name, mapping[argument])
    return None
