# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""GRF's frames for a batch in render mode, one renderer per environment."""

import concurrent.futures
import ctypes
import os

import numpy as np
import torch

from .. import _media, _pixels
from ..errors import LibraryError, MissingFileError

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


def split(environments, cpus=None, workers=0, threads=0):
    cpus = cpus or _cpus()
    workers, threads = workers or None, threads or None
    if workers is None:
        workers = cpus // threads if threads else cpus
    workers = max(1, min(workers, environments))
    if threads is None:
        threads = max(1, cpus // workers)
    return workers, threads


class Frames:
    def __init__(self, batch, kind, width, height, frame_size=(1280, 720), workers=0, threads=0):
        library = batch._library
        self._batch = batch
        self._library = library
        self.kind = kind
        self.width, self.height = width, height
        self.frame_width, self.frame_height = frame_size
        self.event_size = library.cuju_environment_render_event_size()
        self.state_size = library.cuju_engine_render_state_size()
        count = batch.environment_count
        batch._check(library.cuju_environment_enable_render_events(batch._handle(), CAPACITY))
        self._buffer = ctypes.create_string_buffer(self.event_size * CAPACITY)
        self.workers, self.threads = split(count, workers=workers, threads=threads)
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.workers, thread_name_prefix='cuju-pixels'
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
                    self._capture(environment, self.final_observation, self.final_frame)
                seed = int(np.frombuffer(data, '<u4', 1, at + 4)[0])
                self._reset_steps(renderer, environment)
                renderer.start_match(state, seed)
            else:
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
