# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""GRF's frames for a batch in render mode: one renderer per environment, which draws the renders
that each reset and step recorded, in GRF's order, and the `pixels` observations of the frames.

The environments draw side by side on `CUJU_RENDER_WORKERS` threads, each renderer with
`CUJU_RENDER_THREADS` threads of its own, and every frame is the same for every split."""

from __future__ import annotations

import concurrent.futures
import ctypes
import os

import numpy as np
import torch

from .. import _media, _pixels
from ..errors import ConfigurationError, LibraryError, MissingFileError

# The render events one call records per environment at most. A stoppage with a card renders
# about 150 times before play restarts.
CAPACITY = 512
# GRF: `start_game` resets to `ScenarioConfig::make()`, whose seed is 42.
INTERNAL_SEED = 42
_START, _RENDER = 1, 2
MESSAGE = (
    "the representations pixels and pixels_gray and render_frames need cuju's renderer, "
    "libcuju_render, and GRF's media, the package cuju-grf-media; install them with "
    "pip install 'cuju-engine[pixels]'"
)


def _cpus():
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _count(name):
    """The positive count that the environment variable `name` sets, or None."""
    value = os.environ.get(name, '')
    if not value:
        return None
    try:
        count = int(value)
    except ValueError:
        raise ConfigurationError(f'expected {name} to be a whole number, got {value!r}') from None
    return max(1, count)


def split(environments, cpus=None):
    """Returns the renderers drawn at once and the threads of each for a batch of `environments`.

    `CUJU_RENDER_WORKERS` and `CUJU_RENDER_THREADS` set either; by default the workers take one
    environment each up to the processors there are, and share the processors out.
    """
    cpus = cpus or _cpus()
    workers, threads = _count('CUJU_RENDER_WORKERS'), _count('CUJU_RENDER_THREADS')
    if workers is None:
        workers = cpus // threads if threads else cpus
    workers = max(1, min(workers, environments))
    if threads is None:
        threads = max(1, cpus // workers)
    return workers, threads


class Frames:
    """The renderers of a batch and their last frames."""

    def __init__(self, batch, kind, width, height, frame_size=(1280, 720)):
        library = batch._library
        self._batch = batch
        self._library = library
        self.kind = kind  # 'pixels', 'pixels_gray' or None for frames alone
        self.width, self.height = width, height
        self.frame_width, self.frame_height = frame_size
        self.event_size = library.cuju_environment_render_event_size()
        self.state_size = library.cuju_engine_render_state_size()
        count = batch.environment_count
        batch._check(library.cuju_environment_enable_render_events(batch._handle(), CAPACITY))
        self._buffer = ctypes.create_string_buffer(self.event_size * CAPACITY)
        self.workers, self.threads = split(count)
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.workers, thread_name_prefix='cuju-render'
        )
        self.renderers = []
        try:
            if not os.path.isfile(_pixels.library_path()):
                raise MissingFileError(f'the renderer library {_pixels.library_path()} is missing')
            media = os.environ.get('CUJU_GRF_MEDIA') or str(_media.root())
            self.renderers = [
                _pixels.Renderer(self.frame_width, self.frame_height, self.threads, media)
                for _ in range(count)
            ]
        except (MissingFileError, LibraryError) as error:
            self.close()
            raise MissingFileError(f'{MESSAGE}: {error}') from None
        state = ctypes.create_string_buffer(self.state_size)
        # GRF: a rendering engine's `start_game` builds the first match of its graphics.
        library.cuju_engine_internal_render_state(state, self.state_size, 1)
        state = state.raw
        self._started = [False] * count

        def start(environment):
            renderer = self.renderers[environment]
            self._reset_steps(renderer, environment)
            renderer.start_match(state, INTERNAL_SEED)

        self._each(start, range(count))
        channels = 3 if kind == 'pixels' else 1
        agents = batch.agents
        self.observation = torch.zeros((count, agents, height, width, channels), dtype=torch.uint8)
        self.final_observation = torch.zeros_like(self.observation)
        self.frame = torch.zeros((count, self.frame_height, self.frame_width, 3), dtype=torch.uint8)
        self.final_frame = torch.zeros_like(self.frame)

    def _each(self, work, environments):
        """Runs `work(environment)` for each environment on the workers, raising the first error."""
        environments = list(environments)
        if self.workers <= 1 or len(environments) <= 1:
            for environment in environments:
                work(environment)
            return
        for future in [self._pool.submit(work, environment) for environment in environments]:
            future.result()

    def _reset_steps(self, renderer, environment):
        """GRF's `GameEnv::reset` with rendering, up to `StartMatch`."""
        renderer.reset_begin()
        renderer.stop_match()
        if not self._started[environment]:
            renderer.loading_page()
            self._started[environment] = True
        renderer.render_scene(True)

    def _capture(self, environment, observation, frame):
        renderer = self.renderers[environment]
        # The pixels go straight into the first agent's observation, and the others copy them.
        # An environment without agents has no observation to fill.
        agents = observation.shape[1]
        if self.kind == 'pixels' and agents:
            renderer.pixels(self.width, self.height, out=observation[environment, 0].numpy())
        elif self.kind == 'pixels_gray' and agents:
            renderer.pixels_gray(
                self.width, self.height, out=observation[environment, 0, ..., 0].numpy()
            )
        if self.kind is not None and agents > 1:
            observation[environment, 1:] = observation[environment, :1]
        renderer.frame(top_down=True, out=frame[environment].numpy())

    def _events(self, environment):
        """The render events of the last call of `environment`, as bytes."""
        count = ctypes.c_int()
        batch = self._batch
        batch._check(
            self._library.cuju_environment_render_events(
                batch._handle(), environment, self._buffer, CAPACITY, ctypes.byref(count)
            )
        )
        if count.value < 0:
            raise LibraryError(
                f'environment {environment} rendered more than {CAPACITY} times in one call'
            )
        return count.value, self._buffer.raw[: count.value * self.event_size]

    def draw(self, environments):
        """Draws the events of the last call of `environments` and keeps their frames."""
        environments = list(environments)
        events = {environment: self._events(environment) for environment in environments}
        self._each(lambda environment: self._draw(environment, *events[environment]), environments)

    def _draw(self, environment, count, data):
        size, state_size = self.event_size, self.state_size
        renderer = self.renderers[environment]
        kinds = [int(np.frombuffer(data, '<i4', 1, index * size)[0]) for index in range(count)]
        rendered = False
        for index in range(count):
            at = index * size
            state = data[at + 8 : at + 8 + state_size]
            if kinds[index] == _START:
                if rendered:
                    # The step ended the episode and the next began: its last frame is final.
                    self._capture(environment, self.final_observation, self.final_frame)
                seed = int(np.frombuffer(data, '<u4', 1, at + 4)[0])
                self._reset_steps(renderer, environment)
                renderer.start_match(state, seed)
            else:
                # Speed: only the last render before a capture is read, and the renderer
                # rasterizes none of the others, whose pixels the next render draws again.
                read = index + 1 == count or kinds[index + 1] == _START
                renderer.render(state, read)
                rendered = True
        if rendered:
            self._capture(environment, self.observation, self.frame)

    def state_processed(self, environment):
        """GRF: a save or a load makes the next render draw the nets afresh."""
        self.renderers[environment].state_processed()

    def close(self):
        for renderer in self.renderers:
            renderer.close()
        self.renderers = []
        self._pool.shutdown(wait=True)
