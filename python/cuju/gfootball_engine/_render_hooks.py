# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import os

from .. import _config, _media, _pixels
from ..errors import Error, LibraryError, MissingFileError
from . import _native

MESSAGE = (
    "GRF's rendering in cuju.gfootball needs cuju's renderer, libcuju_render, and GRF's media, "
    "the package cuju-grf-media; install them with pip install 'cuju-engine[pixels]'"
)
# GRF: `start_game` resets to `ScenarioConfig::make()`, whose seed is 42.
INTERNAL_SEED = 42
# GRF: the first rendering match of a process generates the pitch, once.
_pitch_generated = False


class RendererMissingError(Error, ModuleNotFoundError):
    """The renderer or the media that GRF's rendering needs is not installed."""


def _threads():
    value = _config.values()['render_threads']
    if value:
        return value
    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = os.cpu_count() or 1
    return max(1, min(4, available))


def _media_folder():
    folder = os.environ.get('CUJU_GRF_MEDIA')
    if folder:
        return folder
    return str(_media.root())


class Session:
    def __init__(self, env):
        game = env.game_config
        try:
            if not os.path.isfile(_pixels.library_path()):
                raise MissingFileError(f'the renderer library {_pixels.library_path()} is missing')
            self.renderer = _pixels.Renderer(
                game.render_resolution_x, game.render_resolution_y, _threads(), _media_folder()
            )
        except (MissingFileError, LibraryError) as error:
            raise RendererMissingError(f'{MESSAGE}: {error}') from None
        library = _native.library()
        self.size = library.cuju_engine_render_state_size()
        self.state = ctypes.create_string_buffer(self.size)
        # GRF: the loading page shows at the first reset of an engine, when no match stopped.
        self.started = False
        self._library = library
        # GRF: `start_game`'s own reset builds the first match of the graphics.
        global _pitch_generated
        self._check(
            library.cuju_engine_internal_render_state(
                self.state, self.size, 0 if _pitch_generated else 1
            )
        )
        _pitch_generated = True
        self._reset_steps(True)
        self.renderer.start_match(self.state, INTERNAL_SEED)

    def _check(self, status):
        if status != 0:
            raise LibraryError(_native.last_error())

    def _reset_steps(self, rendering):
        """GRF's `GameEnv::reset` up to `StartMatch`."""
        self.renderer.reset_begin()
        self.renderer.stop_match()
        if not self.started:
            self.renderer.loading_page()
            self.started = True
        if rendering:
            self.renderer.render_scene(True)

    def reset(self, env, animations):
        """GRF's `GameEnv::reset` after the engine's own reset: its graphics, then StartMatch."""
        rendering = bool(env.game_config.render)
        self._reset_steps(rendering)
        self.renderer.set_render(rendering)
        self._check(self._library.cuju_engine_render_start(env._engine, self.state, self.size))
        self.renderer.start_match(self.state, env._config.game_engine_random_seed)

    def render(self, env, swap_buffer):
        """GRF's `GameEnv::render` after `PrepareRender`."""
        self._check(
            self._library.cuju_engine_render_state(env._engine, self.state, self.size, None)
        )
        self.renderer.set_render(True)
        self.renderer.render(self.state, swap_buffer)

    def state_processed(self, env):
        """GRF's `GameEnv::ProcessState` of a save or a load."""
        self.renderer.state_processed()

    def frame(self):
        """GRF's `GameEnv::get_frame`, rows from the bottom."""
        return self.renderer.frame(top_down=False).tobytes()

    def close(self):
        self.renderer.close()


class _Provider:
    @staticmethod
    def open_session(env):
        return Session(env)


def provider():
    return _Provider


def open_session(env):
    return provider().open_session(env)


def require(session):
    if session is None:
        raise RuntimeError(
            'GRF renders only an engine whose game_config.render was true at start_game()'
        )
    return session
