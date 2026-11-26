# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""The host side of the controller protocol, for transports that user parts write.

The frames and messages are those of `docs/protocol.md`. `Connection` speaks them over any
readable and writable binary stream, such as a pipe or the file of a socket, and `process` uses
the same code. `client.py` keeps its own copy of the codec, so a child needs no cuju.
"""

import json
import math
import struct
import sys
from typing import Any, BinaryIO

import numpy as np

from .client import STUB as STUB

__all__ = [
    'MAXIMUM_BODY',
    'MAXIMUM_HEADER',
    'PROTOCOL',
    'STUB',
    'Connection',
    'FrameError',
    'act',
    'actions',
    'close',
    'decode',
    'encode',
    'error',
    'hello',
    'read',
    'ready',
    'start',
    'write',
]

PROTOCOL = 1
MAXIMUM_HEADER = 16 * 1024 * 1024
MAXIMUM_BODY = 1024 * 1024 * 1024
_LENGTHS = struct.Struct('<II')
_DTYPES = (
    'bool',
    'uint8',
    'int8',
    'int16',
    'uint16',
    'int32',
    'uint32',
    'int64',
    'uint64',
    'float16',
    'float32',
    'float64',
)
_CODES = {np.dtype(name).newbyteorder('<').str: np.dtype(name) for name in _DTYPES}
_CODE_OF = {dtype: code for code, dtype in _CODES.items()}
_ARRAY = '$array'
_NATIVE = sys.byteorder == 'little'


class FrameError(ValueError):
    """A frame that does not read, or a value that no frame can carry."""


def _tree(value: Any, arrays: list[np.ndarray]) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if hasattr(value, 'detach') and hasattr(value, 'cpu'):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        arrays.append(value)
        return {_ARRAY: len(arrays) - 1}
    if isinstance(value, np.generic):
        return _tree(value.item(), arrays)
    if isinstance(value, dict):
        found = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise FrameError(f'a key of a mapping must be text, found {key!r}')
            if key == _ARRAY:
                raise FrameError(f'the key {key!r} is the frames own')
            found[key] = _tree(item, arrays)
        return found
    if isinstance(value, list | tuple):
        return [_tree(item, arrays) for item in value]
    raise FrameError(f'a value of type {type(value).__name__} cannot cross')


def encode(kind: str, value: Any = None, **fields: Any) -> bytes:
    """Returns one frame: two lengths, a JSON header and the raw bytes of every array."""
    arrays: list[np.ndarray] = []
    tree = _tree(value, arrays)
    shapes, parts = [], []
    for array in arrays:
        dtype = array.dtype
        code = _CODE_OF.get(dtype.newbyteorder('=') if not dtype.isnative else dtype)
        if code is None:
            raise FrameError(f'an array of {dtype} cannot cross, only {", ".join(_DTYPES)}')
        if not (_NATIVE and dtype.isnative) and dtype.byteorder != '|':
            array = array.astype(dtype.newbyteorder('<'))
        parts.append(np.ascontiguousarray(array).tobytes())
        shapes.append([code, list(array.shape)])
    header = {'protocol': PROTOCOL, 'kind': kind, **fields, 'value': tree, 'arrays': shapes}
    head = json.dumps(header, separators=(',', ':')).encode('utf-8')
    body = b''.join(parts)
    if len(head) > MAXIMUM_HEADER or len(body) > MAXIMUM_BODY:
        raise FrameError(f'a frame of {len(head) + len(body)} bytes is too large to cross')
    return _LENGTHS.pack(len(head), len(body)) + head + body


def _value(tree: Any, arrays: list[np.ndarray]) -> Any:
    if isinstance(tree, dict):
        if len(tree) == 1 and _ARRAY in tree:
            index = tree[_ARRAY]
            if not isinstance(index, int) or not 0 <= index < len(arrays):
                raise FrameError(f'no array {index!r} in the frame')
            return arrays[index]
        return {key: _value(item, arrays) for key, item in tree.items()}
    if isinstance(tree, list):
        return [_value(item, arrays) for item in tree]
    return tree


def decode(head: bytes, body: bytes | bytearray) -> dict[str, Any]:
    """Returns the header of a frame with its value under `value`, the arrays in their places."""
    try:
        header = json.loads(head.decode('utf-8'))
    except (UnicodeDecodeError, ValueError) as error:
        raise FrameError(f'a frame header does not read, {error}') from None
    if not isinstance(header, dict) or not isinstance(header.get('kind'), str):
        raise FrameError('a frame header names no kind')
    if header.get('protocol') != PROTOCOL:
        raise FrameError(f'a frame of protocol {header.get("protocol")!r}, expected {PROTOCOL}')
    arrays = []
    offset = 0
    view = memoryview(body)
    for entry in header.pop('arrays', None) or ():
        try:
            code, shape = entry
            dtype = _CODES[code]
            count = math.prod(shape)
            size = count * dtype.itemsize
        except (KeyError, TypeError, ValueError) as error:
            raise FrameError(f'an array of the frame does not read, {error!r}') from None
        if count < 0 or offset + size > len(body):
            raise FrameError('a frame holds fewer bytes than its arrays')
        flat = np.frombuffer(view[offset : offset + size], dtype=dtype, count=count)
        arrays.append((flat if _NATIVE else flat.astype(dtype.newbyteorder('='))).reshape(shape))
        offset += size
    if offset != len(body):
        raise FrameError('a frame holds more bytes than its arrays')
    header['value'] = _value(header.get('value'), arrays)
    return header


def _read_exactly(stream: BinaryIO, size: int) -> bytearray:
    buffer = bytearray(size)
    view = memoryview(buffer)
    done = 0
    while done < size:
        count = stream.readinto(view[done:])
        if not count:
            raise EOFError('the stream ended inside a frame')
        done += count
    return buffer


def read(stream: BinaryIO) -> dict[str, Any]:
    """Reads one frame from a binary stream and returns its header with its value."""
    lengths = stream.read(_LENGTHS.size)
    if not lengths:
        raise EOFError('the stream ended')
    if len(lengths) < _LENGTHS.size:
        lengths = bytes(lengths) + bytes(_read_exactly(stream, _LENGTHS.size - len(lengths)))
    head_size, body_size = _LENGTHS.unpack(lengths)
    if head_size > MAXIMUM_HEADER or body_size > MAXIMUM_BODY:
        raise FrameError(f'a frame of {head_size + body_size} bytes is too large to read')
    head = _read_exactly(stream, head_size)
    body = _read_exactly(stream, body_size)
    return decode(bytes(head), body)


def write(stream: BinaryIO, kind: str, value: Any = None, **fields: Any) -> None:
    """Writes one frame to a binary stream and flushes it."""
    stream.write(encode(kind, value, **fields))
    stream.flush()


# The messages, as frames to send.


def hello(inner: dict[str, Any]) -> bytes:
    """Returns the frame that gives the child the unbuilt tree to build."""
    return encode('hello', {'inner': inner})


def start(spec: Any) -> bytes:
    """Returns the frame that starts a session, from a `Spec` or its dictionary."""
    return encode('start', spec if isinstance(spec, dict) else spec.to_dict())


def act(observation: Any) -> bytes:
    """Returns the frame that gives the child an observation."""
    value = {
        name: getattr(observation, name)
        for name in (
            'data',
            'first',
            'active',
            'left_players',
            'right_players',
            'previous',
            'step',
            'side',
            'meta',
            'environments',
        )
    }
    return encode('act', value)


def close() -> bytes:
    """Returns the frame that asks the child to end."""
    return encode('close')


def ready(
    representation: str = 'raw',
    control: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
    **fields: Any,
) -> bytes:
    """Returns the child's answer to `hello`, or to `start` with no arguments but the kind."""
    value = {'representation': representation, 'control': control, 'options': options}
    return encode('ready', value, **fields)


def actions(answer: Any, extras: dict[str, Any] | None = None) -> bytes:
    """Returns the child's answer to `act`."""
    return encode('actions', {'actions': answer, 'extras': extras or {}})


def error(reason: str, message: str, traceback: str = '') -> bytes:
    """Returns the frame that a child sends in place of an answer."""
    return encode('error', None, reason=reason, message=message, traceback=traceback)


class Connection:
    """The frames of the protocol over a readable and a writable binary stream."""

    def __init__(self, reader: BinaryIO, writer: BinaryIO | None = None) -> None:
        """Takes a stream to read, and one to write that may be the same, such as a socket file."""
        self.reader = reader
        self.writer = reader if writer is None else writer

    def send(self, frame: bytes) -> None:
        """Writes one frame, from `hello`, `start`, `act` or `close`, and flushes."""
        self.writer.write(frame)
        self.writer.flush()

    def receive(self) -> dict[str, Any]:
        """Reads the next frame and returns its header with its value."""
        return read(self.reader)
