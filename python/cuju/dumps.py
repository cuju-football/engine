# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Episodes of a batch as GRF dumps, which GRF's `replay` and `dump_to_video` read."""

import collections
import ctypes
import datetime
import io
import math
import os
import pickle
import re
import struct
import timeit
from collections.abc import Iterator
from typing import Any

import torch

from ._batch import unwrap
from .errors import ArgumentError, LibraryError

__all__ = ['DumpWriter', 'iterate_dump', 'read_dump']

_VALUES_PER_BYTE = 4
_VALUES_ALLOWANCE = 1 << 20
_MAX_DEPTH = 32
# GRF's `e_BackendAction`, by value.
_BACKEND_ACTIONS = (
    'idle',
    'left',
    'top_left',
    'top',
    'top_right',
    'right',
    'bottom_right',
    'bottom',
    'bottom_left',
    'long_pass',
    'high_pass',
    'short_pass',
    'shot',
    'keeper_rush',
    'sliding',
    'pressure',
    'team_pressure',
    'switch',
    'sprint',
    'dribble',
    'release_direction',
    'release_long_pass',
    'release_high_pass',
    'release_short_pass',
    'release_shot',
    'release_keeper_rush',
    'release_sliding',
    'release_pressure',
    'release_team_pressure',
    'release_switch',
    'release_sprint',
    'release_dribble',
    'builtin_ai',
)
_DIRECTIONS = (
    'left',
    'top_left',
    'top',
    'top_right',
    'right',
    'bottom_right',
    'bottom',
    'bottom_left',
)
# GRF: `CoreAction`'s `sticky` and `directional` flags.
_STICKY = frozenset((*_DIRECTIONS, 'keeper_rush', 'pressure', 'team_pressure', 'sprint', 'dribble'))
_DIRECTIONAL = frozenset((*_DIRECTIONS, 'release_direction'))
# GRF: `observation_rotation.flip_single_action` turns a direction by 180 degrees.
_FLIPPED = {name: _DIRECTIONS[(index + 4) % 8] for index, name in enumerate(_DIRECTIONS)}
# GRF: `football_action_set.action_set_dict`.
_DEFAULT_SET = (
    'idle',
    'left',
    'top_left',
    'top',
    'top_right',
    'right',
    'bottom_right',
    'bottom',
    'bottom_left',
    'long_pass',
    'high_pass',
    'short_pass',
    'shot',
    'sprint',
    'release_direction',
    'release_sprint',
    'sliding',
    'dribble',
    'release_dribble',
)
_ACTION_SETS = {
    'default': _DEFAULT_SET,
    'v2': (*_DEFAULT_SET, 'builtin_ai'),
    'full': (
        *_DEFAULT_SET,
        'builtin_ai',
        'keeper_rush',
        'pressure',
        'team_pressure',
        'switch',
        'release_long_pass',
        'release_high_pass',
        'release_short_pass',
        'release_shot',
        'release_keeper_rush',
        'release_sliding',
        'release_pressure',
        'release_team_pressure',
        'release_switch',
    ),
}
_ACTION_SET_NAMES = {len(actions): name for name, actions in _ACTION_SETS.items()}
# GRF: `observation_processor.PAST_STEPS_TRACE_SIZE`, the steps before a goal in its dump.
_STEPS_BEFORE_GOAL = 100
_EPISODE_VALUES = 17


class _BackendAction(int):
    __slots__ = ()


class _CoreAction:
    def __init__(self, backend_action, name, sticky=False, directional=False):
        self._backend_action = backend_action
        self._name = name
        self._sticky = sticky
        self._directional = directional

    def is_in_actionset(self, config):
        return self._name in _ACTION_SETS[config['action_set']]

    # GRF: an action compares by its name for equality and by its backend action for order.
    def __eq__(self, other):
        assert set(other.__dict__) == set(self.__dict__)
        return self._name == other._name

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        assert set(other.__dict__) == set(self.__dict__)
        return self._backend_action < other._backend_action

    def __le__(self, other):
        assert set(other.__dict__) == set(self.__dict__)
        return self._backend_action <= other._backend_action

    def __gt__(self, other):
        assert set(other.__dict__) == set(self.__dict__)
        return self._backend_action > other._backend_action

    def __ge__(self, other):
        assert set(other.__dict__) == set(self.__dict__)
        return self._backend_action >= other._backend_action

    def __hash__(self):
        return int(self._backend_action)

    def __repr__(self):
        return self._name


_CORE_ACTIONS = {
    name: _CoreAction(_BackendAction(value), name, name in _STICKY, name in _DIRECTIONAL)
    for value, name in enumerate(_BACKEND_ACTIONS)
}


class _Reconstruct:
    """Stands in for numpy's `_reconstruct` in a pickle."""


class _NumpyArray:
    """Stands in for `numpy.ndarray` in a pickle."""


class _NumpyType:
    def __init__(self, code, order):
        self.code, self.order = code, order

    def __reduce__(self):
        return _NumpyType, (self.code, False, True), (3, self.order, None, None, None, -1, -1, 0)


_FLOAT64 = (_NumpyType('f8', '<'), 'd')
_INT64 = (_NumpyType('i8', '<'), 'q')
_BOOL = (_NumpyType('b1', '|'), '?')
_UINT8 = (_NumpyType('u1', '|'), 'B')


class _Array:
    def __init__(self, kind, shape, values):
        self.kind, self.shape, self.values = kind, tuple(shape), list(values)

    def __reduce__(self):
        numpy_type, format_code = self.kind
        data = struct.pack(f'<{len(self.values)}{format_code}', *self.values)
        return _Reconstruct, (_NumpyArray, (0,), b'b'), (1, self.shape, numpy_type, False, data)


_GLOBALS = {
    _BackendAction: ('_gameplayfootball', 'e_BackendAction'),
    _CoreAction: ('gfootball.env.football_action_set', 'CoreAction'),
    _Reconstruct: ('numpy.core.multiarray', '_reconstruct'),
    _NumpyArray: ('numpy', 'ndarray'),
    _NumpyType: ('numpy', 'dtype'),
}


class _Pickler(pickle._Pickler):
    def save_global(self, obj, name=None):
        target = _GLOBALS.get(obj)
        if target is None:
            return super().save_global(obj, name)
        module, qualified = target
        self.save(module)
        self.save(qualified)
        self.write(pickle.STACK_GLOBAL)
        self.memoize(obj)
        return None


def _pickle(entry):
    buffer = io.BytesIO()
    _Pickler(buffer, protocol=4).dump(entry)
    return buffer.getvalue()


def _write_file(path, pickles):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + '.writing'
    with open(temporary, 'wb') as handle:
        for data in pickles:
            handle.write(data)
    os.replace(temporary, path)


def _difficulty_text(value):
    return 'none' if value is None else repr(float(value)).replace('.', 'p').replace('-', 'm')


_SCENARIO_FILE = """# A GRF scenario: {base} with the team difficulties of a cuju batch.
import importlib

_BASE = importlib.import_module('gfootball.scenarios.{base}')


def build_scenario(builder):
    _BASE.build_scenario(builder)
{lines}"""


def _flip_sticky(states, sticky_names):
    index = {name: position for position, name in enumerate(sticky_names)}
    return [states[index[_FLIPPED.get(name, name)]] for name in sticky_names]


class _Episode:
    def __init__(self, environment, values, config, full, goals):
        self.environment = environment
        self.episode = values[2]
        self.left, self.right = values[9], values[10]
        self.sizes = (values[11], values[12])
        self.config = config
        self.full = full
        self.goals = goals
        self.pickles = []
        self.recent = collections.deque(maxlen=_STEPS_BEFORE_GOAL)
        self.cumulative = 0
        self.frame = 0
        self.stamp = None


def _stamp():
    return datetime.datetime.now().strftime('%Y%m%d-%H%M%S%f')


class DumpWriter:
    """Writes the episodes of chosen environments of a batch as GRF dumps."""

    def __init__(
        self,
        batch: Any,
        logdir: str,
        environments: Any = None,
        write_full_episode_dumps: bool = True,
        write_goal_dumps: bool = False,
        dump_frequency: int = 1,
    ) -> None:
        """Follows `environments` of `batch` from the episodes that its next steps start."""
        batch = unwrap(batch)
        batch._handle()
        library = batch._library
        try:
            library.cuju_environment_episode.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            library.cuju_environment_scenario.restype = ctypes.c_char_p
            library.cuju_environment_scenario.argtypes = [ctypes.c_void_p, ctypes.c_int]
        except AttributeError as error:
            raise LibraryError(
                'the library of the batch lacks cuju_environment_episode; rebuild it'
            ) from error
        count = batch.environment_count
        if environments is None:
            environments = range(count)
        self.environments = list(dict.fromkeys(int(environment) for environment in environments))
        outside = [index for index in self.environments if not 0 <= index < count]
        if outside:
            raise ArgumentError(f'expected environments from 0 to {count - 1}, found {outside}')
        self.batch = batch
        self.logdir = os.fspath(logdir)
        self.write_full_episode_dumps = bool(write_full_episode_dumps)
        self.write_goal_dumps = bool(write_goal_dumps)
        self.dump_frequency = int(dump_frequency)
        self.paths: list[str] = []
        self._grf_names = False
        self._action_set = _ACTION_SETS[_ACTION_SET_NAMES[batch.action_count]]
        self._sticky = [name for name in self._action_set if name in _STICKY]
        self._index = torch.tensor(self.environments, dtype=torch.long)
        self._episodes: dict[int, _Episode | None] = {}
        self._scenario_files: set[str] = set()
        batch._wait_for_device()
        for environment in self.environments:
            values, _ = self._episode_values(environment)
            self._episodes[environment] = (
                self._open(environment) if values[2] >= 1 and values[5] == -1 else None
            )

    def _episode_values(self, environment, difficulty=False):
        return _episode_values(self.batch, environment, difficulty)

    def _level(self, bundle, difficulties):
        batch = self.batch
        name = batch._library.cuju_environment_scenario(batch._handle(), bundle).decode()
        played, own = difficulties[:2], difficulties[2:]
        pair = [None if value == base else value for value, base in zip(played, own, strict=True)]
        if pair == [None, None]:
            return name
        level = f'{name}_difficulty_{_difficulty_text(pair[0])}_{_difficulty_text(pair[1])}'
        if level not in self._scenario_files:
            lines = ''.join(
                f'    builder.config().{side}_team_difficulty = {value!r}\n'
                for side, value in zip(('left', 'right'), pair, strict=True)
                if value is not None
            )
            path = os.path.join(self.logdir, level + '.py')
            os.makedirs(self.logdir, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as handle:
                handle.write(_SCENARIO_FILE.format(base=name, lines=lines))
            self._scenario_files.add(level)
        return level

    def _config(self, values, difficulties):
        seed, reverse, episode, offset, bundle, _, fixed, deterministic, ticks, left, right = (
            values[:11]
        )
        config = {
            'action_set': _ACTION_SET_NAMES[self.batch.action_count],
            'custom_display_stats': None,
            'display_game_stats': True,
            'dump_full_episodes': self.write_full_episode_dumps,
            'dump_scores': self.write_goal_dumps,
            # GRF: a team without agents plays itself, and no agent gives no player.
            'players': [f'agent:left_players={left},right_players={right}'] if left + right else [],
            'level': self._level(bundle, difficulties),
            'physics_steps_per_frame': ticks,
            'render_resolution_x': 1280,
            'real_time': False,
            'tracesdir': self.logdir,
            'video_format': 'avi',
            'video_quality_level': 0,
            'write_video': False,
            'render_resolution_y': 720,
        }
        # GRF: a seed given as an option precedes the episode number.
        if fixed >= 0:
            config['game_engine_random_seed'] = seed
        # GRF: the creation builds the first scenario number, so the first episode is the second.
        config['episode_number'] = episode + 1 + offset
        if not deterministic:
            config['reverse_team_processing'] = bool(reverse)
        if fixed < 0:
            config['game_engine_random_seed'] = seed
        return config

    def _open(self, environment):
        values, difficulties = self._episode_values(environment, difficulty=True)
        frequency = self.dump_frequency
        # GRF: `PeriodicDumpWriter` counts the resets, the first included.
        selected = frequency <= 1 or (values[2] - 1) % frequency == 0
        return _Episode(
            environment,
            values,
            self._config(values, difficulties),
            selected and self.write_full_episode_dumps,
            selected and self.write_goal_dumps,
        )

    def _observation(self, rows, episode, designated):
        return _match_observation(rows, episode, designated, self._sticky)

    def _core_actions(self, episode, row):
        count = len(self._action_set)
        actions = []
        for agent, index in enumerate(row[: episode.left + episode.right]):
            # GRF: an action indexes a Python list.
            if not -count <= index < count:
                raise ArgumentError(
                    f'expected action indices from {-count} to {count - 1}, found {index} for '
                    f'environment {episode.environment}'
                )
            name = self._action_set[index]
            turned = agent >= episode.left
            actions.append(_CORE_ACTIONS[_FLIPPED.get(name, name) if turned else name])
        return actions

    def _path(self, kind, episode, frame=None, stamp=None):
        if self._grf_names:
            path = f'{self.logdir}{os.sep}{kind}_{stamp or _stamp()}.dump'
        else:
            name = f'{kind}_environment_{episode.environment}_episode_{episode.episode}'
            if frame is not None:
                name += f'_step_{frame}'
            path = os.path.join(self.logdir, name + '.dump')
        self.paths.append(path)
        return path

    def _append(self, episode, entry):
        written = []
        reward = entry['reward']
        # GRF: the first step of a dump takes the configuration into the step itself.
        if episode.goals and reward in (1, -1):
            # GRF: the dump of a goal holds the steps before it and the step that scored.
            entries = [*episode.recent, entry]
            entries[0]['debug']['config'] = episode.config
            kind = 'score' if reward == 1 else 'lost_score'
            path = self._path(kind, episode, entry['debug']['frame_cnt'])
            _write_file(path, map(_pickle, entries))
            written.append((kind, path))
        if episode.full:
            if not episode.pickles:
                entry['debug']['config'] = episode.config
                # GRF: the dump of the episode starts, and takes its time, after the first step.
                episode.stamp = _stamp()
            episode.pickles.append(_pickle(entry))
        if episode.goals:
            episode.recent.append(entry)
        return written

    def _finish(self, episode):
        if not episode.full or not episode.pickles:
            return []
        path = self._path('episode_done', episode, stamp=episode.stamp)
        _write_file(path, episode.pickles)
        return [('episode_done', path)]

    def _rows(self, actions):
        batch = self.batch
        shape = (batch.environment_count, batch.agents)
        _, _, done, _ = batch._outputs()
        index = self._index.to(done.device)
        action_rows = actions.reshape(shape)[index.to(actions.device)].to('cpu').tolist()
        ended = done[index].bool()
        score = batch._buffer('score_reward', (batch.environment_count,), torch.int32)
        reward_rows = score[index].to('cpu').tolist()
        raw_rows = None
        if batch._raw_fields is not None:
            current, final = batch._raw('raw'), batch._raw('final_raw')
            raw_rows = {
                name: torch.where(
                    ended.reshape(-1, *[1] * (current[name].dim() - 1)),
                    final[name][index],
                    current[name][index],
                )
                .to('cpu')
                .tolist()
                for name in current
            }
        return action_rows, ended.to('cpu').tolist(), reward_rows, raw_rows

    def _entry(self, episode, actions, reward, raw, designated):
        episode.cumulative += reward
        entry = {
            'debug': {
                'action': self._core_actions(episode, actions),
                'frame_cnt': episode.frame,
                'reward': reward,
                'time': timeit.default_timer(),
            }
        }
        observation = None if raw is None else self._observation(raw, episode, designated)
        if observation is not None:
            entry['observation'] = observation
        entry['reward'] = reward
        entry['cumulative_reward'] = episode.cumulative
        episode.frame += 1
        return entry

    def record(self, actions: Any) -> list[str]:
        """Reads the step that just ran and writes the dumps it completes."""
        return [path for _, path in self._record(actions)]

    def _record(self, actions):
        batch = self.batch
        batch._handle()
        actions = torch.as_tensor(actions)
        shape = (batch.environment_count, batch.agents)
        if tuple(actions.shape) != shape and not (
            batch.agents == 1 and tuple(actions.shape) == (batch.environment_count,)
        ):
            raise ArgumentError(
                f'expected the actions of the step, {shape}, found {tuple(actions.shape)}'
            )
        action_rows, done_rows, reward_rows, raw_rows = self._rows(actions)
        batch._wait_for_device()
        auto_reset = batch.auto_reset
        written = []
        for row, environment in enumerate(self.environments):
            values, _ = self._episode_values(environment)
            episodes, step = values[2], values[5]
            episode = self._episodes[environment]
            ended = bool(done_rows[row])
            if ended:
                continues = episode is not None and (
                    (episodes == episode.episode + 1 and step == -1)
                    if auto_reset
                    else (episodes == episode.episode and step == episode.frame)
                )
            else:
                if episode is not None and not (
                    episodes == episode.episode and step == episode.frame
                ):
                    episode = None
                if episode is None and step == 0:
                    episode = self._open(environment)
                continues = episode is not None
            if continues:
                raw = None
                if raw_rows is not None:
                    raw = [
                        {name: raw_rows[name][row][agent] for name in raw_rows}
                        for agent in range(batch.agents)
                    ]
                designated = values[15:17] if ended else values[13:15]
                entry = self._entry(
                    episode, action_rows[row], int(reward_rows[row]), raw, designated
                )
                written += self._append(episode, entry)
                if ended:
                    written += self._finish(episode)
            if ended:
                episode = self._open(environment) if auto_reset and step == -1 else None
            self._episodes[environment] = episode
        return written

    def close(self) -> list[str]:
        """Writes each unfinished episode as its dump and stops following the batch."""
        written = []
        for environment, episode in self._episodes.items():
            if episode is not None:
                written += [path for _, path in self._finish(episode)]
            self._episodes[environment] = None
        return written

    def __enter__(self) -> 'DumpWriter':
        return self

    def __exit__(self, *exception: Any) -> None:
        self.close()


def _episode_values(batch, environment, difficulty=False):
    library = batch._library
    library.cuju_environment_episode.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    values = (ctypes.c_longlong * _EPISODE_VALUES)()
    difficulties = (ctypes.c_float * 4)() if difficulty else None
    with batch._on_device():
        batch._check(
            library.cuju_environment_episode(
                batch._handle(),
                environment,
                ctypes.addressof(values),
                None if difficulties is None else ctypes.addressof(difficulties),
            )
        )
    return list(values), None if difficulties is None else list(difficulties)


def _sticky_names(action_count):
    return [name for name in _ACTION_SETS[_ACTION_SET_NAMES[action_count]] if name in _STICKY]


def _match_observation(rows, episode, designated, sticky_names):
    left, right = episode.left, episode.right
    if left + right == 0:
        return None
    # GRF: a right agent sees the match turned by 180 degrees with the teams swapped.
    base = rows[0]
    turned = not left
    sign = -1.0 if turned else 1.0

    def point(values):
        return _Array(_FLOAT64, (3,), [sign * values[0], sign * values[1], values[2]])

    observation = {
        'ball': point(base['ball']),
        'ball_direction': point(base['ball_direction']),
        'ball_rotation': _Array(_FLOAT64, (3,), base['ball_rotation']),
    }
    for side, chosen, size in zip(('left', 'right'), designated, episode.sizes, strict=True):
        source = side if not turned else ('right' if side == 'left' else 'left')
        positions = [sign * value for pair in base[f'{source}_team'][:size] for value in pair]
        directions = [
            sign * value for pair in base[f'{source}_team_direction'][:size] for value in pair
        ]
        observation[f'{side}_team'] = _Array(_FLOAT64, (size, 2), positions)
        observation[f'{side}_team_direction'] = _Array(_FLOAT64, (size, 2), directions)
        observation[f'{side}_team_tired_factor'] = _Array(
            _FLOAT64, (size,), base[f'{source}_team_tired_factor'][:size]
        )
        observation[f'{side}_team_active'] = _Array(
            _BOOL, (size,), base[f'{source}_team_active'][:size]
        )
        observation[f'{side}_team_yellow_card'] = _Array(
            _BOOL, (size,), base[f'{source}_team_yellow_card'][:size]
        )
        observation[f'{side}_team_roles'] = _Array(
            _INT64, (size,), base[f'{source}_team_roles'][:size]
        )
        # GRF: every team has one, -1 for none, with or without agents.
        observation[f'{side}_team_designated_player'] = chosen
    sticky = len(sticky_names)
    observation['left_agent_sticky_actions'] = [
        _Array(_UINT8, (sticky,), rows[agent]['sticky_actions']) for agent in range(left)
    ]
    observation['left_agent_controlled_player'] = [rows[agent]['active'] for agent in range(left)]
    observation['right_agent_sticky_actions'] = [
        _Array(_UINT8, (sticky,), _flip_sticky(rows[agent]['sticky_actions'], sticky_names))
        for agent in range(left, left + right)
    ]
    observation['right_agent_controlled_player'] = [
        rows[agent]['active'] for agent in range(left, left + right)
    ]
    owned = base['ball_owned_team']
    score = base['score']
    observation['game_mode'] = base['game_mode']
    observation['score'] = [score[1], score[0]] if turned else [score[0], score[1]]
    observation['ball_owned_team'] = (1 - owned if owned > -1 else -1) if turned else owned
    observation['ball_owned_player'] = base['ball_owned_player']
    observation['steps_left'] = base['steps_left']
    return observation


class _ReadAction:
    """GRF's `CoreAction` as `read_dump` loads it."""


class _ReadBackendAction(int):
    """GRF's `e_BackendAction` as `read_dump` loads it."""


class _ReadType:
    def __init__(self, code, align=False, copy=True):
        self.code, self.order = code, '|'

    def __setstate__(self, state):
        self.order = state[1]


class _ReadArray:
    def __setstate__(self, state):
        _, self.shape, self.type, _, self.data = state


def _read_reconstruct(cls, shape, code):
    return _ReadArray()


class _ReadScalar:
    def __init__(self, numpy_type=None, data=None):
        self.type, self.data = numpy_type, data


def _read_encode(text, encoding='utf-8'):
    # Python 3 writes bytes into a pickle of protocol 2 as text that this function turns back.
    if not isinstance(text, str) or encoding not in ('latin1', 'latin-1'):
        raise pickle.UnpicklingError('expected bytes written as Latin-1 text')
    return text.encode('latin-1')


def _read_reconstructor(cls, base, state):
    # Pickles of protocols 0 and 1 build an object through this function of `copyreg`.
    if cls not in (_ReadAction, _ReadArray):
        raise pickle.UnpicklingError('expected a GRF dump, found an object of another class')
    return cls()


def _read_frombuffer(buffer, numpy_type, shape, order):
    array = _ReadArray()
    array.shape, array.type, array.data = shape, numpy_type, bytes(buffer)
    return array


_READ_TYPES = {
    'f8': torch.float64,
    'f4': torch.float32,
    'f2': torch.float16,
    'i8': torch.int64,
    'i4': torch.int32,
    'i2': torch.int16,
    'i1': torch.int8,
    'u1': torch.uint8,
    'b1': torch.bool,
}
_READ_GLOBALS = {
    ('gfootball.env.football_action_set', 'CoreAction'): _ReadAction,
    ('_gameplayfootball', 'e_BackendAction'): _ReadBackendAction,
    ('gfootball_engine', 'e_BackendAction'): _ReadBackendAction,
    ('cuju.gfootball.env.football_action_set', 'CoreAction'): _ReadAction,
    ('cuju.gfootball_engine._gameplayfootball', 'e_BackendAction'): _ReadBackendAction,
    ('cuju.gfootball_engine', 'e_BackendAction'): _ReadBackendAction,
    ('numpy.core.multiarray', '_reconstruct'): _read_reconstruct,
    ('numpy._core.multiarray', '_reconstruct'): _read_reconstruct,
    ('numpy.core.numeric', '_frombuffer'): _read_frombuffer,
    ('numpy._core.numeric', '_frombuffer'): _read_frombuffer,
    ('numpy.core.multiarray', 'scalar'): _ReadScalar,
    ('numpy._core.multiarray', 'scalar'): _ReadScalar,
    ('numpy', 'ndarray'): _ReadArray,
    ('numpy', 'dtype'): _ReadType,
    ('copyreg', '_reconstructor'): _read_reconstructor,
    ('copy_reg', '_reconstructor'): _read_reconstructor,
    ('_codecs', 'encode'): _read_encode,
}


class _DumpUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        target = _READ_GLOBALS.get((module, name))
        if target is None:
            raise pickle.UnpicklingError(f'expected a GRF dump, found `{module}.{name}`')
        return target


def _spend(budget, count):
    budget[0] -= count
    if budget[0] < 0:
        raise pickle.UnpicklingError('found more values than the size of the file can hold')


def _read_type(value):
    if not isinstance(value, _ReadType) or not isinstance(value.code, str):
        raise pickle.UnpicklingError('expected an array with a numpy type')
    dtype = _READ_TYPES.get(re.sub(r'^[<|=]', '', value.code))
    if dtype is None or value.order == '>':
        raise pickle.UnpicklingError(f'expected a little-endian array, found {value.code}')
    return dtype


def _read_bytes(data, size):
    if isinstance(data, str):
        # Python 2 wrote the bytes of an array as text of Latin-1.
        data = data.encode('latin-1')
    if not isinstance(data, (bytes, bytearray)) or len(data) != size:
        raise pickle.UnpicklingError('expected array bytes that fit the shape')
    return bytearray(data)


def _converted(value, budget=None, depth=0):
    if budget is None:
        budget = [1 << 62]
    if depth > _MAX_DEPTH:
        raise pickle.UnpicklingError(f'found values nested more than {_MAX_DEPTH} deep')
    _spend(budget, 1)
    if isinstance(value, _ReadArray):
        dtype = _read_type(value.type)
        shape = value.shape
        if not isinstance(shape, tuple) or not all(
            isinstance(size, int) and not isinstance(size, bool) and size >= 0 for size in shape
        ):
            raise pickle.UnpicklingError('expected an array with a shape')
        count = math.prod(shape)
        _spend(budget, count)
        data = _read_bytes(value.data, count * dtype.itemsize) if value.data else bytearray()
        if not data:
            if count:
                raise pickle.UnpicklingError('expected array bytes that fit the shape')
            return torch.empty(shape, dtype=dtype)
        return torch.frombuffer(data, dtype=dtype).reshape(shape)
    if isinstance(value, _ReadScalar):
        dtype = _read_type(value.type)
        data = _read_bytes(value.data, dtype.itemsize)
        return torch.frombuffer(data, dtype=dtype)[0].item()
    if isinstance(value, _ReadAction):
        return value.__dict__['_name']
    if isinstance(value, dict):
        return {key: _converted(item, budget, depth + 1) for key, item in value.items()}
    if isinstance(value, list):
        return [_converted(item, budget, depth + 1) for item in value]
    if isinstance(value, tuple):
        return tuple(_converted(item, budget, depth + 1) for item in value)
    if isinstance(value, (bytes, bytearray)):
        _spend(budget, len(value) // 8)
    return value


def read_dump(path: str) -> list[dict[str, Any]]:
    """Reads a GRF dump, of this module or of GRF, without GRF or numpy."""
    return list(_iterate(path, strict=False))


def iterate_dump(path: str) -> Iterator[dict[str, Any]]:
    """Yields the steps of a GRF dump one at a time, as `read_dump` reads them."""
    return _iterate(path, strict=False)


def _iterate(path, *, strict):
    budget = [_VALUES_PER_BYTE * os.path.getsize(path) + _VALUES_ALLOWANCE]
    count = 0
    with open(path, 'rb') as handle:
        while not strict or handle.peek(1):
            try:
                step = _converted(_DumpUnpickler(handle).load(), budget)
            except EOFError:
                if strict:
                    raise pickle.UnpicklingError('the final step is incomplete') from None
                return
            except RecursionError:
                raise pickle.UnpicklingError(
                    f'found values nested more than {_MAX_DEPTH} deep'
                ) from None
            except (TypeError, ValueError, AttributeError, KeyError, IndexError) as error:
                raise pickle.UnpicklingError(
                    f'found a pickle that does not build, {type(error).__name__}'
                ) from None
            if strict and not isinstance(step, dict):
                raise pickle.UnpicklingError('expected each step to be a dictionary')
            count += 1
            yield step
    if strict and not count:
        raise pickle.UnpicklingError('the dump is empty')


def _read_dump(path: str, *, strict: bool) -> list[dict[str, Any]]:
    return list(_iterate(path, strict=strict))


def _read_cli_dump(path: str) -> list[dict[str, Any]]:
    try:
        return _read_dump(path, strict=True)
    except (
        AttributeError,
        IndexError,
        KeyError,
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
        pickle.UnpicklingError,
    ) as error:
        raise ValueError(str(error)) from error
