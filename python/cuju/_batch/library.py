# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import os
from typing import Any

import torch

from .._cli_contract import quote_path
from .._library_path import (  # noqa: F401, the names that this module has always held
    _LIBRARIES,
    _PACKAGE_DIRECTORY,
    _automatic,
    _backend,
    _check_processor,
    _missing_processor_features,
    library_path,
)
from ..errors import DeviceError, LibraryError, MissingFileError


def _placement(device: Any) -> torch.device:
    if _backend(device) == 'cpu':
        return torch.device('cpu')
    if not torch.cuda.is_available():
        raise DeviceError(
            f'expected a CUDA GPU for device={str(device)!r}, found none that PyTorch can use; '
            "pass device='cpu' to run the batch on the CPU"
        )
    index = torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    count = torch.cuda.device_count()
    if not 0 <= index < count:
        raise DeviceError(
            f'expected a CUDA device from cuda:0 to cuda:{count - 1}, found cuda:{index}; '
            'PyTorch sees only these GPUs (CUDA_VISIBLE_DEVICES may hide others)'
        )
    return torch.device('cuda', index)


class _Configuration(ctypes.Structure):
    _fields_ = [
        ('version', ctypes.c_int),
        ('assets', ctypes.POINTER(ctypes.c_char_p)),
        ('asset_count', ctypes.c_int),
        ('assignment', ctypes.c_void_p),
        ('animations', ctypes.c_char_p),
        ('environment_count', ctypes.c_int),
        ('seed', ctypes.c_ulonglong),
        ('action_set', ctypes.c_int),
        ('observation', ctypes.c_int),
        ('checkpoints', ctypes.c_int),
        ('numpy_legacy', ctypes.c_int),
        ('auto_reset', ctypes.c_int),
        ('threads', ctypes.c_int),
        ('ticks_per_step', ctypes.c_int),
        ('smm_width', ctypes.c_int),
        ('smm_height', ctypes.c_int),
        ('stack', ctypes.c_int),
        ('fixed_seeds', ctypes.c_void_p),
        ('agent_counts', ctypes.c_void_p),
        ('episode_number', ctypes.c_int),
        ('first_index', ctypes.c_ulonglong),
        ('allocate', ctypes.c_void_p),
        ('release', ctypes.c_void_p),
        ('allocator_context', ctypes.c_void_p),
        ('pin_threads', ctypes.c_int),
        ('numa', ctypes.c_int),
        ('page_aligned', ctypes.c_int),
        ('render_mode', ctypes.c_int),
        ('referee_animations', ctypes.c_int),
    ]


_CONFIGURATION_VERSION = 3


def _load_library(path=None, device='cuda'):
    variable = _LIBRARIES[device][1]
    if path is None and os.environ.get(variable) and not os.path.isfile(os.environ[variable]):
        raise MissingFileError(
            f'the environment variable {variable} names {quote_path(os.environ[variable])}, '
            'which is no file'
        )
    path = path or library_path(device)
    if not os.path.isfile(path):
        advice = (
            "pass device='cpu', or build the package with a CUDA compiler"
            if device == 'cuda'
            else 'build the package with that backend or pick the other device'
        )
        raise MissingFileError(
            f'expected the library of the backend {device} at {quote_path(path)}, '
            f'which this installation lacks; {advice}'
        )
    if device == 'cpu':
        _check_processor(path)
    library = ctypes.CDLL(path)
    pointer, integer, string = ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p
    library.cuju_backend.restype = string
    backend = library.cuju_backend().decode()
    if backend != device:
        raise LibraryError(
            f'expected the library of the backend {device} at {quote_path(path)}, found {backend}'
        )
    library.cuju_last_error.restype = string
    library.cuju_environment_create.restype = pointer
    library.cuju_environment_create.argtypes = [ctypes.POINTER(_Configuration)]
    library.cuju_environment_destroy.argtypes = [pointer]
    library.cuju_memory_estimate.argtypes = [
        ctypes.POINTER(_Configuration),
        ctypes.POINTER(ctypes.c_ulonglong),
    ]
    library.cuju_local_memory_bytes.argtypes = [ctypes.POINTER(ctypes.c_ulonglong)]
    library.cuju_environment_memory_bytes.restype = ctypes.c_ulonglong
    library.cuju_environment_memory_bytes.argtypes = [pointer]
    library.cuju_environment_count.argtypes = [pointer]
    library.cuju_environment_threads.argtypes = [pointer]
    library.cuju_environment_agents.argtypes = [pointer, pointer, pointer]
    library.cuju_environment_players.argtypes = [pointer, pointer, pointer]
    library.cuju_environment_set_difficulty.argtypes = [pointer, pointer]
    library.cuju_environment_difficulty.argtypes = [pointer, pointer]
    library.cuju_environment_difficulty.restype = None
    library.cuju_environment_set_render.argtypes = [pointer, pointer, integer, integer]
    library.cuju_engine_render_state_version.restype = ctypes.c_uint
    library.cuju_engine_render_state_version.argtypes = []
    library.cuju_engine_render_state_size.restype = ctypes.c_ulonglong
    library.cuju_engine_render_state_size.argtypes = []
    library.cuju_engine_internal_render_state.argtypes = [pointer, ctypes.c_ulonglong, integer]
    library.cuju_environment_render_event_size.restype = ctypes.c_ulonglong
    library.cuju_environment_render_event_size.argtypes = []
    library.cuju_environment_enable_render_events.argtypes = [pointer, integer]
    library.cuju_environment_render_events.argtypes = [pointer, integer, pointer, integer, pointer]
    library.cuju_environment_render_states.argtypes = [
        pointer,
        pointer,
        integer,
        pointer,
        ctypes.c_ulonglong,
        pointer,
    ]
    library.cuju_environment_reset.argtypes = [pointer, pointer, pointer, pointer]
    library.cuju_environment_reset_done.argtypes = [pointer, pointer]
    library.cuju_environment_reset_masked.argtypes = [
        pointer,
        pointer,
        integer,
        pointer,
        pointer,
        pointer,
        integer,
    ]
    library.cuju_environment_step.argtypes = [pointer, pointer, integer, pointer, integer]
    library.cuju_environment_step_async.argtypes = [pointer, pointer, integer, pointer]
    library.cuju_environment_step_wait.argtypes = [pointer]
    library.cuju_environment_synchronize.argtypes = [pointer, pointer]
    library.cuju_environment_get_state.restype = pointer
    library.cuju_environment_get_state.argtypes = [pointer, integer, pointer, pointer]
    library.cuju_environment_set_state.argtypes = [
        pointer,
        integer,
        ctypes.c_char_p,
        ctypes.c_ulonglong,
        pointer,
    ]
    library.cuju_environment_set_stack.argtypes = [
        pointer,
        integer,
        string,
        ctypes.c_char_p,
        ctypes.c_ulonglong,
    ]
    library.cuju_environment_roles.argtypes = [pointer, integer, pointer, integer]
    library.cuju_environment_buffer.restype = pointer
    library.cuju_environment_buffer.argtypes = [pointer, string]
    library.cuju_environment_raw_field.restype = string
    library.cuju_environment_raw_field.argtypes = [pointer, integer, pointer, pointer, pointer]
    return library
