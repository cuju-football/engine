# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import os
import sys
import threading
from typing import Any

import numpy as np

from . import _media
from .errors import ArgumentError, LibraryError, MissingFileError, StateError

API_VERSION = 2
# GRF's pixels and pixels_gray observations, `SMM_WIDTH` by `SMM_HEIGHT`.
PIXELS_WIDTH = 96
PIXELS_HEIGHT = 72

if os.name == 'nt':
    _FILE = 'cuju_render.dll'
elif sys.platform == 'darwin':
    _FILE = 'libcuju_render.dylib'
else:
    _FILE = 'libcuju_render.so'
_VARIABLE = 'CUJU_RENDER_LIBRARY'
_ERRORS = {1: ArgumentError, 2: MissingFileError, 3: StateError, 4: LibraryError}
_lock = threading.Lock()
_library = None


def library_path() -> str:
    return os.environ.get(_VARIABLE) or os.path.join(os.path.dirname(__file__), _FILE)


def _load() -> ctypes.CDLL:
    global _library
    with _lock:
        if _library is not None:
            return _library
        path = library_path()
        if not os.path.isfile(path):
            raise MissingFileError(
                f"cuju's renderer library {path} does not exist; build cuju with the renderer, "
                f'or set {_VARIABLE} to its path'
            )
        try:
            handle = ctypes.CDLL(path)
        except OSError as error:
            raise LibraryError(f'cannot load the renderer library {path}: {error}') from None
        handle.cuju_render_api_version.restype = ctypes.c_uint32
        if handle.cuju_render_api_version() != API_VERSION:
            raise LibraryError(
                f'the renderer library {path} speaks C API {handle.cuju_render_api_version()}, '
                f'and this cuju speaks {API_VERSION}; reinstall cuju'
            )
        pointer = ctypes.c_void_p
        size = ctypes.c_size_t
        handle.cuju_render_state_version.restype = ctypes.c_uint32
        handle.cuju_render_state_size.restype = size
        handle.cuju_render_last_error.restype = ctypes.c_char_p
        handle.cuju_render_create.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.POINTER(pointer),
        ]
        handle.cuju_render_destroy.argtypes = [pointer]
        handle.cuju_render_destroy.restype = None
        for name in ('reset_begin', 'stop_match', 'loading_page', 'state_processed'):
            getattr(handle, 'cuju_render_' + name).argtypes = [pointer]
        handle.cuju_render_render_scene.argtypes = [pointer, ctypes.c_int]
        handle.cuju_render_set_render.argtypes = [pointer, ctypes.c_int]
        handle.cuju_render_start_match.argtypes = [pointer, pointer, size, ctypes.c_uint32]
        handle.cuju_render_render.argtypes = [pointer, pointer, size, ctypes.c_int]
        handle.cuju_render_frame.argtypes = [pointer, ctypes.c_int, pointer, size]
        handle.cuju_render_pixels.argtypes = [pointer, ctypes.c_int, ctypes.c_int, pointer, size]
        handle.cuju_render_pixels_gray.argtypes = handle.cuju_render_pixels.argtypes
        handle.cuju_render_has_frame.argtypes = [pointer, ctypes.POINTER(ctypes.c_int)]
        handle.cuju_render_digests.argtypes = [pointer, ctypes.c_char_p, size]
        handle.cuju_render_digests.restype = size
        _library = handle
        return handle


def _check(status: int) -> None:
    if status:
        message = _load().cuju_render_last_error().decode('utf-8', 'replace')
        raise _ERRORS.get(status, LibraryError)(message)


def state_size() -> int:
    return _load().cuju_render_state_size()


def state_version() -> int:
    return _load().cuju_render_state_version()


def _state(state: Any) -> tuple[Any, int, Any]:
    try:
        view = memoryview(state).cast('B')
    except TypeError:
        raise ArgumentError('a RenderState must be contiguous bytes') from None
    keep = (ctypes.c_char * view.nbytes).from_buffer_copy(view)
    return ctypes.cast(keep, ctypes.c_void_p), view.nbytes, keep


def _output(out, shape):
    if out is None:
        return np.empty(shape, np.uint8)
    if out.shape != shape or out.dtype != np.uint8 or not out.flags.c_contiguous:
        raise ArgumentError(
            f'expected a C-contiguous uint8 array of shape {shape}, got {out.dtype} {out.shape}'
        )
    if not out.flags.writeable:
        raise ArgumentError('expected a writeable array')
    return out


class Renderer:
    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        threads: int = 1,
        media: str | os.PathLike | None = None,
    ):
        library = _load()
        folder = os.fspath(media) if media is not None else str(_media.root())
        handle = ctypes.c_void_p()
        _check(
            library.cuju_render_create(
                int(width), int(height), int(threads), folder.encode('utf-8'), ctypes.byref(handle)
            )
        )
        self._library = library
        self._handle = handle
        self.width = int(width)
        self.height = int(height)
        self.threads = int(threads)

    def _call(self, name: str, *arguments: Any) -> None:
        if not self._handle:
            raise StateError('the renderer is closed')
        _check(getattr(self._library, 'cuju_render_' + name)(self._handle, *arguments))

    def reset_begin(self) -> None:
        self._call('reset_begin')

    def stop_match(self) -> None:
        self._call('stop_match')

    def loading_page(self) -> None:
        self._call('loading_page')

    def render_scene(self, swap: bool = True) -> None:
        self._call('render_scene', int(bool(swap)))

    def start_match(self, state: Any, seed: int) -> None:
        pointer, size, keep = _state(state)
        self._call('start_match', pointer, size, int(seed) & 0xFFFFFFFF)
        del keep

    def state_processed(self) -> None:
        self._call('state_processed')

    def set_render(self, render: bool) -> None:
        """GRF's `game_config.render`, on at first."""
        self._call('set_render', int(bool(render)))

    def render(self, state: Any, swap: bool = True) -> None:
        pointer, size, keep = _state(state)
        self._call('render', pointer, size, int(bool(swap)))
        del keep

    def has_frame(self) -> bool:
        value = ctypes.c_int()
        self._call('has_frame', ctypes.byref(value))
        return bool(value.value)

    def frame(self, top_down: bool = False, out: np.ndarray | None = None) -> np.ndarray:
        out = _output(out, (self.height, self.width, 3))
        self._call('frame', int(bool(top_down)), out.ctypes.data, out.nbytes)
        return out

    def pixels(
        self, width: int = PIXELS_WIDTH, height: int = PIXELS_HEIGHT, out: np.ndarray | None = None
    ) -> np.ndarray:
        out = _output(out, (height, width, 3))
        self._call('pixels', int(width), int(height), out.ctypes.data, out.nbytes)
        return out

    def pixels_gray(
        self, width: int = PIXELS_WIDTH, height: int = PIXELS_HEIGHT, out: np.ndarray | None = None
    ) -> np.ndarray:
        out = _output(out, (height, width))
        self._call('pixels_gray', int(width), int(height), out.ctypes.data, out.nbytes)
        return out

    def digests(self) -> dict[str, str]:
        if not self._handle:
            raise StateError('the renderer is closed')
        size = self._library.cuju_render_digests(self._handle, None, 0)
        buffer = ctypes.create_string_buffer(size + 1)
        self._library.cuju_render_digests(self._handle, buffer, size + 1)
        out = {}
        for line in buffer.value.decode('ascii').splitlines():
            name, _, value = line.rpartition(' ')
            out[name] = value
        return out

    def close(self) -> None:
        if self._handle:
            self._library.cuju_render_destroy(self._handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self) -> Renderer:
        return self

    def __exit__(self, *exception: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
