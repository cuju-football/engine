# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""Loads libcuju_render, cuju's software renderer of GRF's frames, through ctypes.

One `Renderer` stands for one rendering GRF engine, a GameEnv, and its methods follow where
GameEnv touches graphics: the steps of a reset, the start of a match, and each render of a
RenderState, the bytes of `include/cuju/render/render_state.h`. The media come from cuju-grf-media
through `cuju._media` unless a folder is given. `CUJU_RENDER_LIBRARY` overrides the library's path.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from typing import Any

import numpy as np

from . import _media
from .errors import ArgumentError, LibraryError, MissingFileError, StateError

# The C API this module speaks, `CUJU_RENDER_API_VERSION` of src/render/render_api.h.
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
    """Returns the path of libcuju_render, from `CUJU_RENDER_LIBRARY` or beside this module."""
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
    """Returns the bytes of one RenderState, as the library reads it."""
    return _load().cuju_render_state_size()


def state_version() -> int:
    """Returns `RENDER_STATE_VERSION`, the RenderState layout the library reads."""
    return _load().cuju_render_state_version()


def _state(state: Any) -> tuple[Any, int, Any]:
    """Returns a pointer to the bytes of a RenderState, their size, and what keeps them alive."""
    try:
        view = memoryview(state).cast('B')
    except TypeError:
        raise ArgumentError('a RenderState must be contiguous bytes') from None
    keep = (ctypes.c_char * view.nbytes).from_buffer_copy(view)
    return ctypes.cast(keep, ctypes.c_void_p), view.nbytes, keep


def _output(out, shape):
    """Returns `out` checked to hold an image of `shape`, or a new array."""
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
    """One rendering GRF engine's graphics, drawn by cuju's software renderer.

    Args:
        width: The frame's width in pixels, a multiple of 4.
        height: The frame's height in pixels.
        threads: The threads that draw each frame. The frame is the same for every count.
        media: The folder of GRF's media, by default the installed cuju-grf-media.
    """

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
        """GameEnv::reset's first graphics step: the unused resources go, the pages clear."""
        self._call('reset_begin')

    def stop_match(self) -> None:
        """GameTask::StopMatch: the running match leaves the scene."""
        self._call('stop_match')

    def loading_page(self) -> None:
        """The loading page that the first reset of an engine shows."""
        self._call('loading_page')

    def render_scene(self, swap: bool = True) -> None:
        """GraphicsTask::Render alone, as the reset calls it between two matches."""
        self._call('render_scene', int(bool(swap)))

    def start_match(self, state: Any, seed: int) -> None:
        """GameTask::StartMatch with the new match's RenderState and its engine seed."""
        pointer, size, keep = _state(state)
        self._call('start_match', pointer, size, int(seed) & 0xFFFFFFFF)
        del keep

    def state_processed(self) -> None:
        """GameEnv::ProcessState of the rendering match, at each get_state and set_state."""
        self._call('state_processed')

    def set_render(self, render: bool) -> None:
        """GRF's `game_config.render`, on at first.

        Off, `start_match` builds GRF's match without rendering, on the pitch alone, and
        captions keep their text.
        """
        self._call('set_render', int(bool(render)))

    def render(self, state: Any, swap: bool = True) -> None:
        """GameEnv::render of one RenderState; with `swap` the frame is read back."""
        pointer, size, keep = _state(state)
        self._call('render', pointer, size, int(bool(swap)))
        del keep

    def has_frame(self) -> bool:
        """Whether a frame was read back."""
        value = ctypes.c_int()
        self._call('has_frame', ctypes.byref(value))
        return bool(value.value)

    def frame(self, top_down: bool = False, out: np.ndarray | None = None) -> np.ndarray:
        """Returns the last frame, `[height, width, 3]` RGB bytes.

        Rows run from the bottom as glReadPixels gives them, or from the top as GRF's
        observation `frame` holds them when `top_down` is set. With `out`, a C-contiguous
        `uint8` array of that shape, the bytes go there and `out` is returned.
        """
        out = _output(out, (self.height, self.width, 3))
        self._call('frame', int(bool(top_down)), out.ctypes.data, out.nbytes)
        return out

    def pixels(
        self, width: int = PIXELS_WIDTH, height: int = PIXELS_HEIGHT, out: np.ndarray | None = None
    ) -> np.ndarray:
        """Returns GRF's pixels of the last frame, `[height, width, 3]`, OpenCV's INTER_AREA."""
        out = _output(out, (height, width, 3))
        self._call('pixels', int(width), int(height), out.ctypes.data, out.nbytes)
        return out

    def pixels_gray(
        self, width: int = PIXELS_WIDTH, height: int = PIXELS_HEIGHT, out: np.ndarray | None = None
    ) -> np.ndarray:
        """Returns GRF's pixels_gray of the last frame, `[height, width]`, RGB2GRAY then area."""
        out = _output(out, (height, width))
        self._call('pixels_gray', int(width), int(height), out.ctypes.data, out.nbytes)
        return out

    def digests(self) -> dict[str, str]:
        """Returns the SHA-256 of the renderer's fixed data by name, for checks."""
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
        """Frees the renderer. Closing twice does nothing."""
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
