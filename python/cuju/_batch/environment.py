# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import ctypes
import importlib
import importlib.util
import io
import numbers
import operator
import os
import pickle
import struct
import sys
import tempfile
import warnings
from typing import Any

import torch

from ..errors import (
    ArgumentError,
    ChoiceError,
    ClosedError,
    ConfigurationError,
    DeviceError,
    LibraryError,
    StateError,
    _from_library,
)
from . import frame_stack as _frame_stack
from . import log as _log
from .assets import _DATA_DIRECTORY, _resolve_asset
from .library import (
    _CONFIGURATION_VERSION,
    _automatic,
    _backend,
    _Configuration,
    _load_library,
    _placement,
    library_path,
)
from .memory import (
    _LOCAL_MEMORY_HELD,
    _free_device_memory,
    _local_memory_bytes,
    _warn_memory,
)
from .state import _pickle_episode, _StateUnpickler

_REPRESENTATIONS = {'simple115': 1, 'simple115v2': 2, 'extracted': 4, 'raw': 8}
# GRF's frames, which the renderer draws, beside the cheapest representation of the library.
_PIXEL_REPRESENTATIONS = ('pixels', 'pixels_gray')
_RAW_TYPES = (torch.float32, torch.int32, torch.bool, torch.uint8)
_MAXIMUM_AGENTS = 22
_TYPE_STRINGS = {torch.float32: '<f4', torch.int32: '<i4', torch.bool: '|b1', torch.uint8: '|u1'}
_ACTION_SETS = {'default': 0, 'v2': 1, 'full': 2}
_ACTION_COUNTS = {'default': 19, 'v2': 20, 'full': 33}
_CUSTOM_OPERATORS = hasattr(torch.library, 'custom_op')


class _Native:
    def __init__(self, library, device):
        self.library, self.device = library, device
        self.environment = None
        self.scenario_directory = None
        self.allocator = None
        self.pinned = None

    def free(self):
        if self.pinned:
            from . import pinned as _pinned

            _pinned.unregister(self.pinned)
            self.pinned = None
        if self.environment:
            context = (
                torch.cuda.device(self.device)
                if self.device.type == 'cuda'
                else contextlib.nullcontext()
            )
            with context:
                self.library.cuju_environment_destroy(self.environment)
            self.environment = None
            self.allocator = None
        if self.scenario_directory is not None:
            self.scenario_directory.cleanup()
            self.scenario_directory = None

    def __del__(self, _finalizing=sys.is_finalizing):
        if _finalizing():
            return
        self.free()


class _CudaArray:
    def __init__(self, pointer, shape, dtype, owner):
        self.owner = owner
        self.__cuda_array_interface__ = {
            'data': (pointer, False),
            'shape': tuple(shape),
            'typestr': _TYPE_STRINGS[dtype],
            'version': 2,
            'strides': None,
        }


def _view(pointer, shape, dtype, device, owner):
    shape = tuple(shape)
    if not all(shape):
        return torch.zeros(shape, dtype=dtype, device=device)
    if device.type == 'cuda':
        return torch.as_tensor(_CudaArray(pointer, shape, dtype, owner), device=device)
    elements = 1
    for size in shape:
        elements *= size
    memory = (ctypes.c_char * (elements * dtype.itemsize)).from_address(pointer)
    memory.owner = owner
    return torch.frombuffer(memory, dtype=dtype, count=elements).view(shape)


def _spaces():
    try:
        import gymnasium as gym
    except ImportError:
        try:
            import gym
        except ImportError:
            raise ModuleNotFoundError(
                'expected gymnasium or gym for the spaces of a batch'
            ) from None
    return gym.spaces, importlib.import_module(gym.__name__ + '.vector.utils').batch_space


def _host(values, dtype):
    return torch.as_tensor(values, dtype=dtype).to('cpu').contiguous()


def _observer_failed(observer, error):
    _log.LOGGER.warning(
        'cuju: removed the step observer %r, which raised %s: %s',
        observer,
        type(error).__name__,
        error,
    )


def _per_environment(values, dtype, name, count, error=ConfigurationError):
    array = _host(values, dtype)
    if array.dim() == 0:
        array = array.repeat(count)
    if tuple(array.shape) != (count,):
        raise error(
            f'expected {name} with one value, or one for each of the {count} environments, '
            f'found the shape {tuple(array.shape)}'
        )
    return array.contiguous()


def _integers(values, dtype, name, count, error):
    array = torch.as_tensor(values)
    # GRF: the seed of a scenario is a C++ `unsigned int`, which refuses a float, 5.0 included.
    if array.is_floating_point() or array.is_complex():
        raise TypeError(f'expected integer {name}, found {array.dtype}')
    return _per_environment(array, dtype, name, count, error)


def _is_pair(value):
    return isinstance(value, (tuple, list)) and len(value) == 2


@contextlib.contextmanager
def _node_map(nodes):
    if nodes is None:
        yield
        return
    previous = os.environ.get('CUJU_NUMA_NODES')
    os.environ['CUJU_NUMA_NODES'] = nodes
    try:
        yield
    finally:
        if previous is None:
            del os.environ['CUJU_NUMA_NODES']
        else:
            os.environ['CUJU_NUMA_NODES'] = previous


def _choice(kind, value, choices, hints=None):
    if value in choices:
        return choices[value]
    hint = (hints or {}).get(value, '')
    raise ChoiceError(
        f'expected the {kind} {", ".join(map(repr, choices))}, found {value!r}'
        + (f'; {hint}' if hint else '')
    )


class Batch:
    _observers: tuple[Any, ...] = ()
    _closing = False
    _async_actions = None

    def __init__(
        self,
        assets: str | list[str],
        animations: str | None = None,
        environment_count: int = 1024,
        representation: str = 'simple115v2',
        rewards: str = 'scoring',
        action_set: str = 'default',
        seed: int = 0,
        numpy_legacy: bool = True,
        auto_reset: bool = True,
        library: str | None = None,
        assignment: Any = None,
        with_raw: bool = False,
        difficulty: list[tuple[float | None, float | None]] | None = None,
        device: str | torch.device = 'cuda',
        threads: int = 0,
        physics_steps_per_frame: int = 10,
        channel_dimensions: tuple[int, int] = (96, 72),
        stacked: bool = False,
        game_engine_random_seed: Any = None,
        squeeze: bool = False,
        agent_counts: list[tuple[int | None, int | None]] | None = None,
        episode_number: int = 0,
        first_index: int = 0,
        allocator: str = 'cuda',
        pin_threads: bool = False,
        numa: bool = False,
        pin_memory: bool = False,
        check_actions: bool = False,
        debug: bool = False,
        render_frames: bool = False,
        render_resolution_x: int = 1280,
        render_resolution_y: int = 720,
        signed_actions: bool = True,
        render_threads: int = 0,
        render_workers: int = 0,
        log: bool | None = None,
    ) -> None:
        device = _automatic(device)
        backend = _backend(device)
        pixel_kind = representation if representation in _PIXEL_REPRESENTATIONS else None
        if pixel_kind and stacked:
            raise ConfigurationError(f'expected no stack with {representation}, found stacked=True')
        representation_flag = _choice(
            'representation',
            'simple115' if pixel_kind else representation,
            _REPRESENTATIONS,
            {
                'extracted_stacked': "pass representation='extracted' with stacked=True",
                'simple115_stacked': "pass representation='simple115' with stacked=True",
            },
        )
        _choice('action set', action_set, _ACTION_SETS)
        # GRF: an assertion requires `scoring` in the rewards.
        if 'scoring' not in rewards.split(','):
            raise ConfigurationError(
                f'expected the rewards scoring or scoring,checkpoints, found {rewards!r}'
            )
        # GRF: the other parts are ignored, so a typo silently drops a reward.
        unknown = [part for part in rewards.split(',') if part not in ('scoring', 'checkpoints')]
        if unknown:
            warnings.warn(
                f'the rewards {rewards!r} hold {unknown}, which GRF ignores, as cuju does; '
                "write them without spaces, as 'scoring,checkpoints'",
                stacklevel=2,
            )
        if pixel_kind or render_frames:
            for name, value in (('x', render_resolution_x), ('y', render_resolution_y)):
                if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                    raise ConfigurationError(
                        f'expected render_resolution_{name} to be a whole number, found {value!r}'
                    )
                if not 0 < value <= 16384:
                    raise ConfigurationError(
                        f'expected render_resolution_{name} from 1 to 16384, found {value}'
                    )
            if render_resolution_x % 4:
                raise ConfigurationError(
                    'expected render_resolution_x to be a multiple of 4, found '
                    f'{render_resolution_x}'
                )
        if not 0 <= seed < 2**64:
            raise ConfigurationError(f'expected a seed from 0 to 2**64 - 1, found {seed}')
        if environment_count < 1:
            raise ConfigurationError(
                f'expected at least one environment, found {environment_count}'
            )
        # GRF: the index and the seed share one key, so an index of 2**32 would be another seed's.
        if not 0 <= first_index <= 2**32 - environment_count:
            raise ConfigurationError(
                f'expected a first index from 0 to 2**32 - environment_count, found {first_index}'
            )
        if threads < 0:
            raise ConfigurationError(
                f'expected 0 threads for one per processor, or more, found {threads}'
            )
        if allocator not in ('cuda', 'torch'):
            raise ChoiceError(f'expected the allocator cuda or torch, found {allocator!r}')
        if pin_memory and backend == 'cpu' and not torch.cuda.is_available():
            raise DeviceError(
                'expected a CUDA GPU for pin_memory=True, found none that PyTorch can use'
            )
        numa_nodes = numa if isinstance(numa, str) else None
        numa = bool(numa)
        self.device = backend
        self._torch_device = _placement(device)
        logged = _log.start(backend, self._torch_device, log)
        self._library = _load_library(library, backend)
        self._native = _Native(self._library, self._torch_device)
        requested = [assets] if isinstance(assets, str) or callable(assets) else list(assets)
        self._requested = requested
        resolved = [_resolve_asset(asset) for asset in requested]
        paths = [path for path, _ in resolved]
        if animations is None:
            animations = os.path.join(_DATA_DIRECTORY, 'animations.bin')
        for index, path in enumerate(paths):
            if path.endswith('.py'):
                from .. import scenarios

                cached = scenarios._batch_bundle(path, backend, library, animations)
                if cached is not None:
                    paths[index] = cached
                    continue
                if self._native.scenario_directory is None:
                    self._native.scenario_directory = tempfile.TemporaryDirectory(prefix='cuju-')
                name = os.path.splitext(os.path.basename(path))[0]
                paths[index] = scenarios.build_bundle(
                    path,
                    os.path.join(self._native.scenario_directory.name, f'{index}_{name}_l0_r0.cjb'),
                    device=backend,
                    library=library,
                    animations=animations,
                )
        if agent_counts is not None and (
            len(agent_counts) != len(paths) or not all(map(_is_pair, agent_counts))
        ):
            raise ConfigurationError(
                f'expected a (left, right) pair of agent counts for each of the {len(paths)} '
                f'bundles, such as [(4, 0)] for one, found {agent_counts!r}'
            )
        counts = [
            [
                named if given is None else given
                for named, given in zip(named_pair, given_pair, strict=True)
            ]
            for (_, named_pair), given_pair in zip(
                resolved, agent_counts or [(None, None)] * len(paths), strict=True
            )
        ]
        counts_array = (ctypes.c_int * (2 * len(paths)))(
            *(count for pair in counts for count in pair)
        )
        if difficulty is not None:
            if len(difficulty) != len(paths) or not all(map(_is_pair, difficulty)):
                raise ConfigurationError(
                    f'expected a (left, right) pair of difficulties for each of the {len(paths)} '
                    f'bundles, such as [(None, 0.5)] for one, found {difficulty!r}'
                )
            if any(
                value is not None and not 0 <= value <= 1 for pair in difficulty for value in pair
            ):
                raise ConfigurationError(
                    f"expected difficulties from 0 to 1, or None for the scenario's, found "
                    f'{difficulty!r}'
                )
        if assignment is None and len(paths) > 1:
            warnings.warn(
                f'the batch has {len(paths)} bundles and no assignment, so every environment plays '
                'the first; pass assignment, the bundle index of each environment',
                stacklevel=2,
            )
        if with_raw:
            representation_flag |= _REPRESENTATIONS['raw']
        if stacked and representation == 'raw':
            raise ConfigurationError(
                'expected simple115, simple115v2 or extracted for a stack, found raw'
            )
        path_array = (ctypes.c_char_p * len(paths))(*(path.encode() for path in paths))
        assignment_array = (
            None
            if assignment is None
            else _per_environment(assignment, torch.int32, 'assignment', environment_count)
        )
        fixed_seeds = None
        if game_engine_random_seed is not None:
            fixed_seeds = _integers(
                game_engine_random_seed,
                torch.int64,
                'game_engine_random_seed',
                environment_count,
                ConfigurationError,
            )
        width, height = channel_dimensions
        configuration = _Configuration(
            version=_CONFIGURATION_VERSION,
            assets=path_array,
            asset_count=len(paths),
            assignment=None if assignment_array is None else assignment_array.data_ptr(),
            animations=animations.encode(),
            environment_count=environment_count,
            seed=seed,
            action_set=_ACTION_SETS[action_set],
            observation=representation_flag,
            checkpoints=int('checkpoints' in rewards.split(',')),
            numpy_legacy=int(numpy_legacy),
            auto_reset=int(auto_reset),
            threads=threads,
            ticks_per_step=physics_steps_per_frame,
            smm_width=width,
            smm_height=height,
            stack=4 if stacked else 1,
            fixed_seeds=None if fixed_seeds is None else fixed_seeds.data_ptr(),
            agent_counts=ctypes.addressof(counts_array),
            episode_number=episode_number,
            first_index=first_index,
            pin_threads=int(pin_threads),
            numa=int(numa),
            page_aligned=int(pin_memory and backend == 'cpu'),
        )
        if allocator == 'torch' and backend == 'cuda':
            from .allocator import TorchAllocator

            self._native.allocator = TorchAllocator(self._torch_device)
            configuration.allocate, configuration.release = self._native.allocator.pointers()
        local_memory = _local_memory_bytes(self._library, self._torch_device)
        free_memory = None
        if backend == 'cuda':
            free_memory = _free_device_memory(self._torch_device, allocator)
        with self._on_device(), _node_map(numa_nodes):
            environment = self._library.cuju_environment_create(ctypes.byref(configuration))
        if not environment:
            message = self._library.cuju_last_error().decode()
            for path, asset in zip(paths, requested, strict=True):
                message = message.replace(path, repr(asset))
            if self._native.allocator is not None and self._native.allocator.errors:
                raise self._native.allocator.errors[0]
            error = _from_library(message)
            if isinstance(error, DeviceError) and 'out of memory' in message:
                error = DeviceError(
                    self._out_of_memory(message, configuration, free_memory, local_memory)
                )
            raise error
        self._native.environment = environment
        if backend == 'cuda':
            _LOCAL_MEMORY_HELD.add(self._torch_device.index)
        if free_memory is not None:
            _warn_memory(
                self.memory_bytes + local_memory, free_memory, self._torch_device, local_memory
            )
        self.threads = self._library.cuju_environment_threads(environment)
        self.environment_count = environment_count
        self.left_agents = torch.zeros(environment_count, dtype=torch.int32)
        self.right_agents = torch.zeros(environment_count, dtype=torch.int32)
        self.agents = self._library.cuju_environment_agents(
            environment, self.left_agents.data_ptr(), self.right_agents.data_ptr()
        )
        self.left_players = torch.zeros(environment_count, dtype=torch.int32)
        self.right_players = torch.zeros(environment_count, dtype=torch.int32)
        self.players = self._library.cuju_environment_players(
            environment, self.left_players.data_ptr(), self.right_players.data_ptr()
        )
        slots = torch.arange(self.agents)
        self.agent_mask = slots < (self.left_agents + self.right_agents)[:, None]
        self.action_count = _ACTION_COUNTS[action_set]
        if squeeze and not bool(((self.left_agents + self.right_agents) == 1).all()):
            self.close()
            raise ConfigurationError(
                'expected one agent in every environment to squeeze the agent axis'
            )
        self.squeeze = squeeze
        self.stacked = stacked
        self.channel_dimensions = (width, height)
        bundles = (
            torch.zeros(environment_count, dtype=torch.int64)
            if assignment_array is None
            else assignment_array.long()
        )
        chosen = torch.full((environment_count, 2), float('nan'))
        for bundle, pair in enumerate(difficulty or []):
            for side, value in enumerate(pair):
                if value is not None:
                    chosen[bundles == bundle, side] = value
        self._chosen_difficulty = chosen
        if not torch.isnan(chosen).all():
            self._apply_difficulty(chosen)
        self.difficulty = torch.empty((environment_count, 2), dtype=torch.float32)
        self._library.cuju_environment_difficulty(self._handle(), self.difficulty.data_ptr())
        self.auto_reset = auto_reset
        self.check_actions, self.debug = check_actions, debug
        self.signed_actions = signed_actions
        self._checks_device_actions = backend == 'cuda' and (check_actions or debug)
        self._agent_mask_on_device = None
        self._checkpoints = 'checkpoints' in rewards.split(',')
        self._representation = representation
        self._raw_fields = (
            self._describe_raw() if representation_flag & _REPRESENTATIONS['raw'] else None
        )
        stack = 4 if stacked else 1
        if representation == 'raw':
            self._observation_name, self._observation_shape, self._observation_type = (
                'raw',
                None,
                None,
            )
        elif representation == 'extracted':
            self._observation_name, self._observation_shape, self._observation_type = (
                'smm',
                (environment_count, self.agents, height, width, 4 * stack),
                torch.uint8,
            )
        else:
            self._observation_name, self._observation_shape, self._observation_type = (
                'simple115',
                (environment_count, self.agents, 115 * stack),
                torch.float32,
            )
        self._cached_outputs = None
        self._score_reward = None
        _log.created(self, requested, library or library_path(backend), logged)
        self._operator_key = None
        self._operator_outputs = None
        self._operator_static = False
        self._step_in_flight = False
        if pin_memory and backend == 'cpu':
            from . import pinned as _pinned

            try:
                self._native.pinned = _pinned.register(self._output_ranges())
            except BaseException:
                self.close()
                raise
        self._frames = None
        self._render_frames = bool(render_frames)
        if pixel_kind or render_frames:
            from .pixels import Frames

            # GRF: a rendering engine plays render mode and every reset with animations.
            self._set_render(render_mode=1, referee_animations=True)
            try:
                self._frames = Frames(
                    self,
                    pixel_kind,
                    width,
                    height,
                    (int(render_resolution_x), int(render_resolution_y)),
                    render_workers,
                    render_threads,
                )
            except BaseException:
                self.close()
                raise
        if os.environ.get('CUJU_BATCH_HOOKS'):
            from .. import _plugins

            _plugins.on_batch_created(self)

    @property
    def scenario_name(self) -> str:
        names = []
        for asset in self._requested:
            name = os.path.basename(os.fspath(asset))
            for suffix in ('.cjb', '.py'):
                name = name.removesuffix(suffix)
            if name not in names:
                names.append(name)
        return ', '.join(names)

    @property
    def observers(self) -> tuple[Any, ...]:
        return self._observers

    def add_observer(self, observer: Any) -> Any:
        self._handle()
        if self._closing:
            raise ClosedError('the batch is closing')
        if not callable(observer):
            raise ArgumentError(f'expected a callable observer, found {observer!r}')
        if any(added is observer for added in self._observers):
            raise ArgumentError(f'expected an observer that was not added yet, found {observer!r}')
        self._observers = (*self._observers, observer)
        return observer

    def remove_observer(self, observer: Any) -> None:
        if not self._discard_observer(observer) and not self._closing:
            raise ArgumentError(f'expected an observer of the batch, found {observer!r}')

    def _discard_observer(self, observer):
        kept = tuple(added for added in self._observers if added is not observer)
        found = len(kept) != len(self._observers)
        self._observers = kept
        return found

    def _capturing(self):
        if self.device == 'cpu':
            return False
        with self._on_device():
            return torch.cuda.is_current_stream_capturing()

    def _notify(self, done, information):
        if self._capturing():
            return
        for observer in self._observers:
            if not any(added is observer for added in self._observers):
                continue
            try:
                observer(self, done, information)
            except Exception as error:
                self._discard_observer(observer)
                _observer_failed(observer, error)
        if self._closing:
            raise ClosedError('an observer closed the batch during the call')

    def _output_ranges(self):
        count, agents = self.environment_count, self.agents
        sizes = {'reward': count * agents * 4, 'done': count, 'score_reward': count * 4}
        if self._observation_shape is not None:
            elements = 1
            for size in self._observation_shape:
                elements *= size
            observation = elements * self._observation_type.itemsize
            sizes[self._observation_name] = sizes['final_' + self._observation_name] = observation
        if self._raw_fields is not None:
            raw = 0
            for _, dtype, shape, offset in self._raw_fields:
                elements = 1
                for size in shape:
                    elements *= size
                raw = max(raw, offset + elements * dtype.itemsize)
            sizes['raw'] = sizes['final_raw'] = raw
        return [
            (self._library.cuju_environment_buffer(self._handle(), name.encode()), size)
            for name, size in sizes.items()
            if size > 0
        ]

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        try:
            for observer in self._observers:
                if not any(added is observer for added in self._observers):
                    continue
                try:
                    observer(self, None, {'close': True})
                except Exception as error:
                    _observer_failed(observer, error)
        finally:
            self._observers = ()
            frames = getattr(self, '_frames', None)
            if frames is not None:
                frames.close()
            native = getattr(self, '_native', None)
            if native is not None:
                native.free()

    def __enter__(self) -> Batch:
        self._handle()
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()

    def __reduce__(self):
        raise TypeError(
            'a batch cannot be pickled; create it inside the worker process that uses it'
        )

    @property
    def memory_bytes(self) -> int:
        return int(self._library.cuju_environment_memory_bytes(self._handle()))

    def _out_of_memory(self, message, configuration, free, local):
        needed = ctypes.c_ulonglong()
        status = self._library.cuju_memory_estimate(
            ctypes.byref(configuration), ctypes.byref(needed)
        )
        requested = f'{needed.value + local} bytes' if status == 0 else 'an unknown number of bytes'
        found = 'an unknown number of bytes' if free is None else f'{free} bytes'
        return f'{message}; the batch needed {requested} and the device had {found} free'

    def _handle(self):
        environment = getattr(getattr(self, '_native', None), 'environment', None)
        if not environment:
            raise ClosedError('the batch is closed')
        return environment

    def _check(self, status):
        if status != 0:
            raise _from_library(self._library.cuju_last_error().decode())

    def _set_render(self, mask=None, render_mode=0, referee_animations=False):
        pointer = None
        if mask is not None:
            values = (ctypes.c_ubyte * self.environment_count)(*(1 if flag else 0 for flag in mask))
            pointer = ctypes.cast(values, ctypes.c_void_p)
        self._check(
            self._library.cuju_environment_set_render(
                self._handle(), pointer, int(render_mode), 1 if referee_animations else 0
            )
        )

    def _render_states(self, indices):
        count = len(indices)
        size = self._library.cuju_engine_render_state_size()
        environments = (ctypes.c_int * max(count, 1))(*indices)
        states = ctypes.create_string_buffer(max(count * size, 1))
        processed = (ctypes.c_uint * max(count, 1))()
        self._check(
            self._library.cuju_environment_render_states(
                self._handle(),
                ctypes.cast(environments, ctypes.c_void_p),
                count,
                ctypes.cast(states, ctypes.c_void_p),
                count * size,
                ctypes.cast(processed, ctypes.c_void_p),
            )
        )
        return states.raw[: count * size], list(processed)[:count]

    def _describe_raw(self):
        fields = []
        element_type, offset = ctypes.c_int(), ctypes.c_ulonglong()
        dimensions = (ctypes.c_int * 2)()
        for field in range(64):
            name = self._library.cuju_environment_raw_field(
                self._handle(),
                field,
                ctypes.byref(element_type),
                dimensions,
                ctypes.byref(offset),
            )
            if name is None:
                break
            shape = (
                self.environment_count,
                self.agents,
                *(dimension for dimension in dimensions if dimension),
            )
            fields.append((name.decode(), _RAW_TYPES[element_type.value], shape, offset.value))
        return fields

    def _raw(self, name):
        pointer = self._library.cuju_environment_buffer(self._handle(), name.encode())
        return {
            field: _view(pointer + offset, shape, dtype, self._torch_device, self._native)
            for field, dtype, shape, offset in self._raw_fields
        }

    def _buffer(self, name, shape, dtype):
        pointer = self._library.cuju_environment_buffer(self._handle(), name.encode())
        return _view(pointer, shape, dtype, self._torch_device, self._native)

    def _outputs(self):
        self._handle()
        if self._cached_outputs is None:
            reward = self._buffer('reward', (self.environment_count, self.agents), torch.float32)
            done = self._buffer('done', (self.environment_count,), torch.uint8)
            if self._representation == 'raw':
                observation, final = self._raw('raw'), self._raw('final_raw')
            else:
                observation = self._buffer(
                    self._observation_name, self._observation_shape, self._observation_type
                )
                final = self._buffer(
                    'final_' + self._observation_name,
                    self._observation_shape,
                    self._observation_type,
                )
            if self.squeeze:
                # GRF: the raw observation of one agent keeps its agent axis.
                if self._representation != 'raw':
                    observation, final = observation.squeeze(1), final.squeeze(1)
                reward = reward.squeeze(1)
            self._cached_outputs = observation, reward, done, final
            self._prepare_operators()
        if not self._operator_static and 'torch._dynamo' in sys.modules:
            for tensor in self._operator_outputs:
                torch._dynamo.mark_static_address(tensor)
            self._operator_static = True
        return self._cached_outputs

    @property
    def torch_device(self) -> torch.device:
        return self._torch_device

    def _prepare_operators(self):
        from . import operators as _operators

        if self._score_reward is None:
            self._score_reward = self._buffer(
                'score_reward', (self.environment_count,), torch.int32
            )
        self._operator_outputs = [
            tensor
            for output in (*self._cached_outputs, self._score_reward)
            for tensor in (output.values() if isinstance(output, dict) else (output,))
        ]
        self._operator_key = _operators.register(self)

    def _operator_step(self, actions, outputs):
        if self._checks_device_actions:
            self._check_device_actions(actions)
        with self._on_device():
            status = self._library.cuju_environment_step(
                self._handle(), actions.data_ptr(), 1, self._stream(), 0
            )
        if status == -2:
            raise ArgumentError(self._library.cuju_last_error().decode())
        self._check(status)
        self._wait_in_debug()
        self._copy_operator_outputs(outputs)

    def _operator_reset_done(self, outputs):
        with self._on_device():
            self._check(self._library.cuju_environment_reset_done(self._handle(), self._stream()))
        self._wait_in_debug()
        self._copy_operator_outputs(outputs)

    def _copy_operator_outputs(self, outputs):
        for given, own in zip(outputs, self._operator_outputs, strict=True):
            if given.data_ptr() != own.data_ptr():
                given.copy_(own)

    def _compiled_step(self, actions):
        shape = (self.environment_count, self.agents)
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)
        if actions.is_floating_point() or actions.is_complex():
            raise TypeError(f'expected integer action indices, found {actions.dtype}')
        if tuple(actions.shape) != shape and not (
            self.agents == 1 and tuple(actions.shape) == (self.environment_count,)
        ):
            raise ArgumentError(
                f'expected actions of the shape {shape}, [environment_count, agents], found '
                f'{tuple(actions.shape)}'
            )
        actions = actions.to(self._torch_device, torch.int32).reshape(shape).contiguous()
        if self._operator_outputs is None:
            self._outputs()
        torch.ops.cuju.step(self._operator_key, actions, self._operator_outputs)
        observation, reward, done, final = self._cached_outputs
        return (
            observation,
            reward,
            done,
            {'final_observation': final, 'score_reward': self._score_reward},
        )

    def _compiled_reset_done(self):
        if self._operator_outputs is None:
            self._outputs()
        torch.ops.cuju.reset_done(self._operator_key, self._operator_outputs)

    def _stream(self):
        if self.device == 'cpu':
            return None
        return torch.cuda.current_stream(self._torch_device).cuda_stream

    def _on_device(self):
        if self.device == 'cpu':
            return contextlib.nullcontext()
        return torch.cuda.device(self._torch_device)

    def _per_environment(self, values, dtype, name):
        return _per_environment(values, dtype, name, self.environment_count, ArgumentError)

    def _staged(self, array):
        if array is not None and self.device == 'cuda' and array.is_pinned():
            return array.clone()
        return array

    def reset(self, seeds: Any = None, reverse: Any = None, mask: Any = None) -> Any:
        seed_array = None
        if seeds is not None:
            seed_array = _integers(
                seeds, torch.int64, 'seeds', self.environment_count, ArgumentError
            )
            # GRF: the scenario holds its seed as a 32-bit `unsigned int`.
            if bool(((seed_array < -1) | (seed_array > 4294967295)).any()):
                raise ArgumentError(
                    f'expected seeds from 0 to 4294967295, or -1 to draw, found '
                    f'{seed_array.tolist()}'
                )
        reverse_array = None
        if reverse is not None:
            reverse_array = _integers(
                reverse, torch.int64, 'reverse', self.environment_count, ArgumentError
            )
            if bool(((reverse_array != 0) & (reverse_array != 1)).any()):
                raise ArgumentError(
                    f'expected reverse flags of 0 or 1, found {reverse_array.tolist()}'
                )
            reverse_array = reverse_array.to(torch.int32)
        if mask is not None:
            return self._reset_masked(mask, seed_array, reverse_array)
        if seed_array is not None and bool((seed_array == -1).any()):
            return self._reset_masked(None, seed_array, reverse_array, synchronize=True)
        if seed_array is not None:
            seed_array = seed_array.to(torch.int32)
        with self._on_device():
            self._check(
                self._library.cuju_environment_reset(
                    self._handle(),
                    None if seed_array is None else seed_array.data_ptr(),
                    None if reverse_array is None else reverse_array.data_ptr(),
                    self._stream(),
                )
            )
        self._draw()
        observation = self._observation_of(self._outputs()[0])
        if self._observers:
            self._notify(None, {'reset': None})
        return observation

    def _reset_masked(self, mask, seed_array, reverse_array, synchronize=False):
        mask_array, on_device = None, 0
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask)
            if mask.is_floating_point() or mask.is_complex():
                raise TypeError(f'expected a boolean or integer mask, found {mask.dtype}')
            if mask.device == self._torch_device and mask.dim() == 1:
                if tuple(mask.shape) != (self.environment_count,):
                    raise ArgumentError(
                        f'expected a mask of the shape ({self.environment_count},), found '
                        f'{tuple(mask.shape)}'
                    )
                mask_array, on_device = mask.to(torch.uint8).contiguous(), 1
            else:
                mask_array = self._staged(self._per_environment(mask, torch.uint8, 'mask'))
                on_device = int(self.device == 'cpu')
        seed_array, reverse_array = self._staged(seed_array), self._staged(reverse_array)
        with self._on_device():
            self._check(
                self._library.cuju_environment_reset_masked(
                    self._handle(),
                    None if mask_array is None else mask_array.data_ptr(),
                    on_device,
                    None if seed_array is None else seed_array.data_ptr(),
                    None if reverse_array is None else reverse_array.data_ptr(),
                    self._stream(),
                    int(synchronize),
                )
            )
        self._wait_in_debug()
        self._draw()
        observation = self._observation_of(self._outputs()[0])
        if self._observers and not self._capturing():
            self._notify(None, {'reset': None if mask_array is None else mask_array.bool()})
        return observation

    def _apply_difficulty(self, chosen):
        values = torch.nan_to_num(chosen, nan=-1.0).contiguous()
        with self._on_device():
            self._check(
                self._library.cuju_environment_set_difficulty(self._handle(), values.data_ptr())
            )

    def set_difficulty(self, values: Any, mask: Any = None) -> None:
        count = self.environment_count
        if _is_pair(values):
            table = torch.tensor(
                [float('nan') if value is None else float(value) for value in values]
            ).expand(count, 2)
        else:
            table = torch.as_tensor(values, dtype=torch.float32)
            if tuple(table.shape) != (count, 2):
                raise ArgumentError(
                    f'expected a (left, right) pair or values of the shape ({count}, 2), found '
                    f'{tuple(table.shape)}'
                )
        if bool((~torch.isnan(table) & ((table < 0) | (table > 1))).any()):
            raise ConfigurationError(
                f"expected difficulties from 0 to 1, or None for the scenario's, found {values!r}"
            )
        selected = (
            torch.ones(count, dtype=torch.bool)
            if mask is None
            else self._per_environment(mask, torch.uint8, 'mask').bool()
        )
        chosen = self._chosen_difficulty.clone()
        chosen[selected] = table.float()[selected]
        self._apply_difficulty(chosen)
        self._chosen_difficulty = chosen
        self._library.cuju_environment_difficulty(self._handle(), self.difficulty.data_ptr())

    def reset_done(self) -> None:
        if _CUSTOM_OPERATORS and torch.compiler.is_compiling():
            return self._compiled_reset_done()
        ended = self._outputs()[2].bool() if self._observers and not self._capturing() else None
        with self._on_device():
            self._check(self._library.cuju_environment_reset_done(self._handle(), self._stream()))
        self._wait_in_debug()
        self._draw()
        if self._observers:
            self._notify(None, {'reset': ended})

    def observation(self) -> Any:
        return self._observation_of(self._outputs()[0])

    def frame(self) -> torch.Tensor:
        if self._frames is None:
            raise ConfigurationError(
                "expected a batch that renders, with representation='pixels' or render_frames=True"
            )
        return self._frames.frame.to(self._torch_device)

    def _observation_of(self, native, final=False):
        frames = self._frames
        if frames is None or frames.kind is None:
            return native
        observation = frames.final_observation if final else frames.observation
        observation = observation.to(self._torch_device)
        return observation.squeeze(1) if self.squeeze else observation

    def _draw(self):
        if self._frames is not None:
            self._frames.draw(range(self.environment_count))

    @property
    def single_observation_space(self) -> Any:
        if self._representation == 'raw':
            return None
        spaces, _ = _spaces()
        import numpy as np

        shape = self._observation_shape[2 if self.squeeze else 1 :]
        if self._representation in _PIXEL_REPRESENTATIONS:
            shape = tuple(self._frames.observation.shape[2 if self.squeeze else 1 :])
            return spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        if self._representation == 'extracted':
            return spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        return spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)

    @property
    def single_action_space(self) -> Any:
        spaces, _ = _spaces()
        if self.squeeze:
            return spaces.Discrete(self.action_count)
        return spaces.MultiDiscrete([self.action_count] * self.agents)

    @property
    def observation_space(self) -> Any:
        single = self.single_observation_space
        return None if single is None else _spaces()[1](single, self.environment_count)

    @property
    def action_space(self) -> Any:
        return _spaces()[1](self.single_action_space, self.environment_count)

    def _raw_of(self, name):
        if self._raw_fields is None:
            raise ConfigurationError('expected a batch with the raw representation or `with_raw`')
        return self._raw(name)

    def raw(self) -> dict[str, torch.Tensor]:
        return self._raw_of('raw')

    def final_raw(self) -> dict[str, torch.Tensor]:
        return self._raw_of('final_raw')

    def step(self, actions: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
        if _CUSTOM_OPERATORS and torch.compiler.is_compiling():
            if self._frames is not None:
                raise ConfigurationError('a batch that renders steps outside torch.compile')
            return self._compiled_step(actions)
        shape = (self.environment_count, self.agents)
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)
            if actions.numel() == 0:
                actions = actions.to(torch.int32)
        # GRF: an action must be an integer.
        if actions.is_floating_point() or actions.is_complex():
            raise TypeError(f'expected integer action indices, found {actions.dtype}')
        if tuple(actions.shape) != shape and not (
            self.agents == 1 and tuple(actions.shape) == (self.environment_count,)
        ):
            raise ArgumentError(
                f'expected actions of the shape {shape}, [environment_count, agents], found '
                f'{tuple(actions.shape)}'
            )
        if actions.device == self._torch_device:
            contiguous_actions = actions.to(torch.int32).reshape(shape).contiguous()
            on_device = 1
            if self._checks_device_actions:
                self._check_device_actions(contiguous_actions)
        else:
            contiguous_actions = self._staged(_host(actions, torch.int32).reshape(shape))
            on_device = int(self.device == 'cpu')
        self._check_unsigned(contiguous_actions)
        # GRF: an action indexes a Python list, so one outside the action set raises.
        with self._on_device():
            status = self._library.cuju_environment_step(
                self._handle(), contiguous_actions.data_ptr(), on_device, self._stream(), 0
            )
        if status == -2:
            raise ArgumentError(self._library.cuju_last_error().decode())
        self._check(status)
        self._wait_in_debug()
        self._draw()
        observation, reward, done, final = self._outputs()
        observation, final = self._observation_of(observation), self._observation_of(final, True)
        if self._score_reward is None:
            self._score_reward = self._buffer(
                'score_reward', (self.environment_count,), torch.int32
            )
        information = {'final_observation': final, 'score_reward': self._score_reward}
        if self._render_frames:
            information['frame'] = self._frames.frame.to(self._torch_device)
            information['final_frame'] = self._frames.final_frame.to(self._torch_device)
        if self._observers:
            actions = self._on_batch_device(contiguous_actions)
            self._notify(done, {**information, 'reward': reward, 'actions': actions})
        return observation, reward, done, information

    def _on_batch_device(self, actions, copy=False):
        if actions.device != self._torch_device:
            return actions.to(self._torch_device)
        return actions.clone() if copy else actions

    def _check_unsigned(self, actions):
        if self.signed_actions or actions.device.type != 'cpu':
            return
        count = self.action_count
        outside = ((actions < 0) | (actions >= count)) & self.agent_mask
        if not bool(outside.any()):
            return
        environment, agent = (int(index) for index in outside.nonzero()[0])
        raise ArgumentError(
            f'expected action indices from 0 to {count - 1}, found '
            f'{int(actions[environment, agent])} for agent {agent} of environment {environment}'
        )

    def _check_device_actions(self, actions):
        with self._on_device():
            if torch.cuda.is_current_stream_capturing():
                return
            if self._agent_mask_on_device is None:
                self._agent_mask_on_device = self.agent_mask.to(self._torch_device)
            count = self.action_count
            lowest = -count if self.signed_actions else 0
            outside = ((actions < lowest) | (actions >= count)) & self._agent_mask_on_device
            if not bool(outside.any()):
                return
            environment, agent = (int(index) for index in outside.nonzero()[0])
            raise ArgumentError(
                f'expected action indices from {lowest} to {count - 1}, found '
                f'{int(actions[environment, agent])} on the device for agent {agent} of '
                f'environment {environment}'
            )

    def _wait_in_debug(self):
        if not self.debug or self.device == 'cpu':
            return
        with self._on_device():
            if torch.cuda.is_current_stream_capturing():
                return
            self._check(self._library.cuju_environment_synchronize(self._handle(), self._stream()))

    def _environment_index(self, index):
        index = operator.index(index)
        if not 0 <= index < self.environment_count:
            raise ArgumentError(
                f'expected an environment from 0 to {self.environment_count - 1}, found {index}'
            )
        return index

    def _check_stack(self, stacked):
        if self.stacked and not stacked:
            raise StateError(
                'expected stacked=True for the state of a batch that stacks its observations, '
                "whose state holds GRF's FrameStack"
            )
        if stacked and not self.stacked:
            raise StateError('expected a batch with stacked=True for a stacked state')

    def _frame_shape(self):
        shape = self._observation_shape[2:]
        dtype = 'uint8' if self._observation_type == torch.uint8 else 'float32'
        return (*shape[:-1], shape[-1] // _frame_stack.FRAMES), dtype

    def step_async(self, actions: Any) -> None:
        if self._step_in_flight:
            raise RuntimeError('a step is in flight; call step_wait before the next step_async')
        if self._frames is not None:
            raise ConfigurationError('a batch that renders steps with step, not step_async')
        shape = (self.environment_count, self.agents)
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)
        if actions.is_floating_point() or actions.is_complex():
            raise TypeError(f'expected integer action indices, found {actions.dtype}')
        if tuple(actions.shape) != shape and not (
            self.agents == 1 and tuple(actions.shape) == (self.environment_count,)
        ):
            raise ArgumentError(
                f'expected actions of the shape {shape}, [environment_count, agents], found '
                f'{tuple(actions.shape)}'
            )
        if actions.device == self._torch_device:
            contiguous_actions = actions.to(torch.int32).reshape(shape).contiguous()
            on_device = 1
            if self._checks_device_actions:
                self._check_device_actions(contiguous_actions)
        else:
            contiguous_actions = self._staged(_host(actions, torch.int32).reshape(shape))
            on_device = int(self.device == 'cpu')
        self._check_unsigned(contiguous_actions)
        with self._on_device():
            status = self._library.cuju_environment_step_async(
                self._handle(), contiguous_actions.data_ptr(), on_device, self._stream()
            )
        if status == -2:
            raise ArgumentError(self._library.cuju_last_error().decode())
        self._check(status)
        self._wait_in_debug()
        self._async_actions = (
            self._on_batch_device(contiguous_actions, copy=True)
            if self._observers
            else contiguous_actions
        )
        self._step_in_flight = True

    def step_wait(self) -> tuple[Any, Any, Any, dict[str, Any]]:
        if not self._step_in_flight:
            raise RuntimeError('no step is in flight; call step_async first')
        self._step_in_flight = False
        actions, self._async_actions = self._async_actions, None
        self._check(self._library.cuju_environment_step_wait(self._handle()))
        observation, reward, done, final = self._outputs()
        information = {'final_observation': final, 'score_reward': self._score_reward}
        if self._observers:
            actions = self._on_batch_device(actions)
            self._notify(done, {**information, 'reward': reward, 'actions': actions})
        return observation, reward, done, information

    def _wait_for_device(self):
        if self.device == 'cuda':
            torch.cuda.synchronize(self._torch_device)

    def get_state(self, index: int, stacked: bool = False) -> bytes:
        index = self._environment_index(index)
        self._check_stack(stacked)
        state = self._state(index, stacked)
        if self._frames is not None:
            self._frames.state_processed(index)
        return state

    def _state(self, index, stacked=False):
        episode = (ctypes.c_int * (3 + _MAXIMUM_AGENTS))()
        size = ctypes.c_ulonglong()
        self._wait_for_device()
        with self._on_device():
            pointer = self._library.cuju_environment_get_state(
                self._handle(), index, ctypes.byref(size), episode
            )
        if not pointer:
            raise _from_library(self._library.cuju_last_error().decode())
        match = ctypes.string_at(pointer, size.value)
        collected = list(episode[3 : 3 + self.agents]) if self._checkpoints else None
        frames = None
        if stacked:
            agents = int(self.left_agents[index] + self.right_agents[index])
            stack = self._buffer(
                self._observation_name, self._observation_shape, self._observation_type
            )[index, :agents].cpu()
            frames = _frame_stack.pickled_frames(_frame_stack.frames_of(stack, agents))
        data = _pickle_episode(list(episode[:3]), collected, frames)
        return struct.pack('<i', len(data)) + data + match

    def set_state(self, index: int, state: bytes, stacked: bool = False) -> None:
        index = self._environment_index(index)
        self._check_stack(stacked)
        if len(state) < 4 or not 0 < struct.unpack_from('<i', state)[0] <= len(state) - 4:
            raise StateError(
                f'expected the bytes of a state: a pickle length, the pickle and the match, '
                f'found {len(state)} bytes that hold none'
            )
        (length,) = struct.unpack_from('<i', state)
        try:
            unpickler = _StateUnpickler(io.BytesIO(state[4 : 4 + length]))
            unpickler.stacked = stacked
            unpickled = unpickler.load()
            logic = unpickled['FootballEnvCore'].__dict__
        except (pickle.UnpicklingError, KeyError, AttributeError, EOFError) as error:
            raise StateError(
                f'expected the pickle of the episode logic in the state, found {error}'
            ) from error
        episode = (ctypes.c_int * (3 + _MAXIMUM_AGENTS))(
            logic['previous_score_diff'], logic['previous_game_mode'], logic['prev_ball_owned_team']
        )
        # GRF: the checkpoint reward reads its counts from the state.
        if self._checkpoints and 'CheckpointRewardWrapper' not in unpickled:
            raise StateError('expected the checkpoints of the checkpoint reward in the state')
        for agent, count in unpickled.get('CheckpointRewardWrapper', {}).items():
            episode[3 + agent] = count
        stack = None
        if stacked and 0 <= index < self.environment_count:
            # GRF: `FrameStack.set_state` takes the deque of the state as its observations.
            if 'FrameStack' not in unpickled:
                raise StateError("expected GRF's FrameStack in the stacked state")
            shape, dtype = self._frame_shape()
            agents = int(self.left_agents[index] + self.right_agents[index])
            try:
                stack = _frame_stack.stack_bytes(unpickled['FrameStack'], agents, shape, dtype)
            except ValueError as error:
                raise StateError(str(error)) from error
        match = state[4 + length :]
        self._wait_for_device()
        with self._on_device():
            status = self._library.cuju_environment_set_state(
                self._handle(), index, match, len(match), episode
            )
        if status != 0:
            message = self._library.cuju_last_error().decode()
            if 'found another' in message:
                # GRF: a state loads only where the constant part of the scenario is the same.
                message += (
                    '; a state loads only into an environment of the same scenario, agent counts, '
                    'episode end flags, duration, ball position, formation and difficulties'
                )
            error = _from_library(message)
            raise StateError(message) if type(error) is LibraryError else error
        if self._frames is not None:
            self._frames.state_processed(index)
        if stack is not None:
            with self._on_device():
                self._check(
                    self._library.cuju_environment_set_stack(
                        self._handle(), index, self._observation_name.encode(), stack, len(stack)
                    )
                )

    def roles(self, index: int) -> dict[str, torch.Tensor]:
        index = self._environment_index(index)
        left = int(self.left_players[index])
        roles = torch.empty(left + int(self.right_players[index]), dtype=torch.int32)
        self._wait_for_device()
        with self._on_device():
            count = self._library.cuju_environment_roles(
                self._handle(), index, roles.data_ptr(), roles.numel()
            )
        if count < 0:
            raise _from_library(self._library.cuju_last_error().decode())
        return {'left_team_roles': roles[:left], 'right_team_roles': roles[left:]}
