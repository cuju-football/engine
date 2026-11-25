# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""GRF's `FrameStack` in the pickle of a stacked state, a deque of four numpy observations."""

from __future__ import annotations

import collections
import importlib
import pickle
import warnings

# GRF: `FrameStack(env, 4)` keeps the last four observations, the oldest first.
FRAMES = 4

_GLOBALS = {
    ('collections', 'deque'),
    ('_codecs', 'encode'),
    ('builtins', 'bytes'),
    ('numpy', 'ndarray'),
    ('numpy', 'dtype'),
    ('numpy._core.multiarray', '_reconstruct'),
    ('numpy.core.multiarray', '_reconstruct'),
    ('numpy._core.numeric', '_frombuffer'),
    ('numpy.core.numeric', '_frombuffer'),
}


def _numpy():
    try:
        import numpy
    except ImportError:
        raise ModuleNotFoundError(
            'expected numpy for a stacked state, whose frames GRF pickles as numpy arrays'
        ) from None
    return numpy


def frames_of(stack, agents):
    numpy = _numpy()
    array = stack.numpy()
    width = array.shape[-1] // FRAMES
    frames = [
        numpy.ascontiguousarray(array[..., width * frame : width * (frame + 1)])
        for frame in range(FRAMES)
    ]
    return [frame[0] if agents == 1 else frame for frame in frames]


def pickled_frames(frames):
    data = pickle.dumps(collections.deque(frames, maxlen=FRAMES), protocol=2)
    assert data[:2] == b'\x80\x02' and data[-1:] == b'.'
    return data[2:-1]


def find_class(module, name):
    if (module, name) not in _GLOBALS:
        return None
    if not module.startswith('numpy'):
        return getattr(importlib.import_module(module), name)
    _numpy()
    other = (
        module.replace('numpy._core', 'numpy.core')
        if '._core' in module
        else module.replace('numpy.core', 'numpy._core')
    )
    for candidate in (module, other):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return getattr(importlib.import_module(candidate), name)
        except (ImportError, AttributeError):
            continue
    raise pickle.UnpicklingError(f'cannot resolve `{module}.{name}` with this numpy')


def stack_bytes(frames, agents, shape, dtype):
    numpy = _numpy()
    frames = list(frames) if isinstance(frames, (collections.deque, list, tuple)) else None
    expected = tuple(shape) if agents == 1 else (agents, *shape)
    if frames is None or len(frames) != FRAMES:
        raise ValueError(f'expected the {FRAMES} frames of GRF in the FrameStack of the state')
    arrays = []
    for frame in frames:
        array = numpy.asarray(frame)
        if array.shape != expected or array.dtype != numpy.dtype(dtype):
            raise ValueError(
                f'expected frames of the shape {expected} and the type {dtype} in the state, '
                f'found {array.shape} and {array.dtype}'
            )
        arrays.append(array.reshape((agents, *shape)))
    return numpy.ascontiguousarray(numpy.concatenate(arrays, axis=-1)).tobytes()
