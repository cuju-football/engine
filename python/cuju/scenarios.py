# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Builds the asset bundle of a GRF scenario file, without GRF."""

from __future__ import annotations

import ctypes
import enum
import hashlib
import importlib.util
import os
import random
import sys
import tempfile
import types
import warnings
from collections.abc import Callable
from typing import Any

from . import _checksums
from ._batch import log as _log
from ._library_path import _automatic, _backend, _check_processor, library_path
from .errors import ConfigurationError, _from_library

__all__ = [
    'Builder',
    'PlayerRole',
    'ScenarioConfig',
    'ScenarioError',
    'Team',
    'build_bundle',
    'cache_directory',
    'cached_bundle',
    'describe',
    'load_scenario',
]

_ANIMATIONS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'animations.bin')
# GRF: a team holds at most 11 players, for 11 formation slots and 11 persons.
_MAXIMUM_TEAM_PLAYERS = 11
# GRF: creating an environment builds episode 1 twice, and the first episode played is 2.
_EPISODES = tuple(range(1, 9)) + tuple(
    10**power + parity for power in range(2, 10) for parity in (0, 1)
)


class PlayerRole(enum.IntEnum):
    """The roles of `e_PlayerRole` of GRF."""

    e_PlayerRole_GK = 0
    e_PlayerRole_CB = 1
    e_PlayerRole_LB = 2
    e_PlayerRole_RB = 3
    e_PlayerRole_DM = 4
    e_PlayerRole_CM = 5
    e_PlayerRole_LM = 6
    e_PlayerRole_RM = 7
    e_PlayerRole_AM = 8
    e_PlayerRole_CF = 9


class Team(enum.IntEnum):
    """The teams of `e_Team` of GRF."""

    e_Left = 0
    e_Right = 1


class ScenarioError(ConfigurationError):
    """A scenario that GRF refuses or that a bundle cannot hold."""


# GRF: boost.python converts the values of a scenario to C++ as these helpers check.
def _to_float(name, value):
    if not isinstance(value, (int, float)):
        raise ScenarioError(f'expected a number for {name}, found {value!r}')
    return float(value)


def _to_integer(name, value, low=-(2**31), high=2**31 - 1):
    if not isinstance(value, int):
        raise ScenarioError(f'expected an integer for {name}, found {value!r}')
    if not low <= value <= high:
        raise ScenarioError(f'expected {name} from {low} to {high}, found {value}')
    return int(value)


def _to_unsigned(name, value):
    return _to_integer(name, value, 0, 2**32 - 1)


def _to_bool(name, value):
    if value is not None and not isinstance(value, int):
        raise ScenarioError(f'expected True or False for {name}, found {value!r}')
    return bool(value)


_FIELDS = {
    'use_magnet': (True, _to_bool),
    'offsides': (True, _to_bool),
    'real_time': (False, _to_bool),
    'left_team_difficulty': (1.0, _to_float),
    'right_team_difficulty': (0.6, _to_float),
    'deterministic': (False, _to_bool),
    'end_episode_on_score': (False, _to_bool),
    'end_episode_on_possession_change': (False, _to_bool),
    'end_episode_on_out_of_play': (False, _to_bool),
    'game_duration': (3000, _to_integer),
    'second_half': (999999999, _to_integer),
    'control_all_players': (False, _to_bool),
    'game_engine_random_seed': (42, _to_unsigned),
    'reverse_team_processing': (False, _to_bool),
    'left_agents': (1, _to_integer),
    'right_agents': (0, _to_integer),
}
# GRF: after `build_scenario`, the configuration replaces these, and the seed and team order.
_REPLACED = ('left_agents', 'right_agents', 'real_time')
_READ_ONLY = ('dynamic_player_selection', 'controllable_left_players', 'controllable_right_players')


class FormationEntry:
    """One player of a team, as `libgame.FormationEntry` holds it."""

    def __init__(self, x, y, role, lazy, controllable):
        self.x = _to_float('x', x)
        self.y = _to_float('y', y)
        engine = sys.modules.get('cuju.gfootball_engine._gameplayfootball')
        if engine is not None and isinstance(role, engine.e_PlayerRole) and 0 <= role <= 9:
            role = PlayerRole(int(role))
        if not isinstance(role, PlayerRole):
            raise ScenarioError(f'expected a role such as e_PlayerRole_GK, found {role!r}')
        self.role = role
        self.lazy = _to_bool('lazy', lazy)
        self.controllable = _to_bool('controllable', controllable)

    def key(self):
        return (self.x, self.y, int(self.role), self.lazy, self.controllable)


class ScenarioConfig:
    """The configuration that `builder.config()` returns, with the fields of GRF."""

    def __init__(self):
        object.__setattr__(self, '_values', {name: value for name, (value, _) in _FIELDS.items()})
        object.__setattr__(self, 'left_team', [])
        object.__setattr__(self, 'right_team', [])
        object.__setattr__(self, 'ball_position', [0.0, 0.0, 0.0])
        object.__setattr__(self, '_cache_read', False)

    def __getattr__(self, name):
        values = object.__getattribute__(self, '_values')
        if name in values:
            return values[name]
        if name in _READ_ONLY:
            return self._read_only(name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in _READ_ONLY:
            raise AttributeError(f"can't set attribute {name}")
        if name in _FIELDS:
            self._values[name] = _FIELDS[name][1](name, value)
        elif name in ('left_team', 'right_team', 'ball_position'):
            raise ScenarioError(f'set {name} through AddPlayer or SetBallPosition')
        else:
            # GRF keeps and ignores it, so a misspelt option would change nothing silently.
            warnings.warn(
                f'the scenario sets {name!r}, which GRF does not know and ignores', stacklevel=2
            )
            object.__setattr__(self, name, value)

    def _read_only(self, name):
        # GRF: the three values are computed once, at the first read, and kept.
        object.__setattr__(self, '_cache_read', True)
        left = sum(1 for player in self.left_team if player.controllable)
        right = sum(1 for player in self.right_team if player.controllable)
        if name == 'controllable_left_players':
            return left
        if name == 'controllable_right_players':
            return right
        agents_left, agents_right = self._values['left_agents'], self._values['right_agents']
        return not (
            (left == agents_left or agents_left == 0)
            and (right == agents_right or agents_right == 0)
        )


class Builder:
    """The `builder` argument of `build_scenario`, as `gfootball.env.scenario_builder.Scenario`."""

    def __init__(self, episode_number):
        self._config = ScenarioConfig()
        self._active_team = Team.e_Left
        self._episode_number = episode_number

    def config(self):
        return self._config

    def ScenarioConfig(self):
        return self._config

    def SetTeam(self, team):
        self._active_team = team

    def AddPlayer(self, x, y, role, lazy=False, controllable=True):
        player = FormationEntry(x, y, role, lazy, controllable)
        if self._active_team == Team.e_Left:
            self._config.left_team.append(player)
        else:
            self._config.right_team.append(player)

    def SetBallPosition(self, ball_x, ball_y):
        self._config.ball_position[0] = _to_float('ball_x', ball_x)
        self._config.ball_position[1] = _to_float('ball_y', ball_y)

    def EpisodeNumber(self):
        return self._episode_number


_PACKAGE = 'cuju._scenario_files'


def _scenario_package():
    package = sys.modules.get(_PACKAGE)
    if package is None:
        package = types.ModuleType(_PACKAGE)
        package.__path__ = []
        for role in PlayerRole:
            setattr(package, role.name, role)
        package.Team = Team
        package.__all__ = [role.name for role in PlayerRole] + ['Team']
        sys.modules[_PACKAGE] = package
    return package


def load_scenario(path):
    """Imports a scenario file and returns its `build_scenario` function."""
    _scenario_package()
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(f'{_PACKAGE}.{name}', path)
    if spec is None or spec.loader is None:
        raise ScenarioError(f'cannot load the scenario file {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    build = getattr(module, 'build_scenario', None)
    if not callable(build):
        raise ScenarioError(f'expected a build_scenario function in {path}')
    return build


def _numpy_state():
    numpy = sys.modules.get('numpy')
    if numpy is None:
        return None
    state = numpy.random.get_state()
    return (state[0], state[1].tobytes(), state[2], state[3], state[4])


def _build(build_scenario: Callable[[Builder], Any], episode_number):
    python_state = random.getstate()
    numpy_state = _numpy_state()
    builder = Builder(episode_number)
    build_scenario(builder)
    if random.getstate() != python_state:
        raise ScenarioError(
            'the scenario draws from the random module of Python; GRF draws the seed '
            'of every episode from that generator right after the scenario, so a bundle '
            'cannot hold it'
        )
    if _numpy_state() != numpy_state:
        raise ScenarioError('the scenario draws from the global random generator of NumPy')
    config = builder.config()
    if config._cache_read:
        raise ScenarioError(
            'the scenario reads controllable_left_players, controllable_right_players or '
            'dynamic_player_selection, which GRF computes once at that read, before it '
            'sets the agents, and keeps for the match, so a bundle cannot hold it'
        )
    # GRF: `Scenario._FakePlayersForEmptyTeam` gives an empty team a lazy goalkeeper.
    for side, team in (('left', config.left_team), ('right', config.right_team)):
        if not team:
            team.append(FormationEntry(-1.0, 0.42, PlayerRole.e_PlayerRole_GK, True, True))
        if len(team) > _MAXIMUM_TEAM_PLAYERS:
            raise ScenarioError(f'a team has {len(team)} players, more than the 11 that GRF holds')
        # GRF: `CalculateDynamicRoles` loops forever in open play without a goalkeeper first.
        if team[0].role != PlayerRole.e_PlayerRole_GK:
            raise ScenarioError(
                f'the first player of the {side} team is not a goalkeeper, which hangs GRF once '
                'the match is in open play'
            )
    values = {name: value for name, value in config._values.items() if name not in _REPLACED}
    if values['deterministic']:
        if values['reverse_team_processing']:
            raise ScenarioError('a deterministic scenario that turns reverse_team_processing on')
    else:
        # GRF: the seed of the episode replaces the seed and the order of the teams.
        values['game_engine_random_seed'] = 42
    values.pop('reverse_team_processing')
    return {
        'values': values,
        'ball': (config.ball_position[0], config.ball_position[1]),
        'left': [player.key() for player in config.left_team],
        'right': [player.key() for player in config.right_team],
    }


def describe(scenario, episodes=_EPISODES):
    """Runs a scenario for several episode numbers and returns its even and odd descriptions."""
    build = load_scenario(scenario) if isinstance(scenario, (str, os.PathLike)) else scenario
    by_parity = {}
    for episode in episodes:
        description = _build(build, episode)
        first, first_episode = by_parity.setdefault(episode % 2, (description, episode))
        if first != description:
            raise ScenarioError(
                f'the scenario differs between episodes {first_episode} and {episode}, which have '
                'the same parity; a bundle holds one template for each parity only'
            )
    if set(by_parity) != {0, 1}:
        raise ScenarioError('expected episode numbers of both parities')
    even, odd = by_parity[0][0], by_parity[1][0]
    if len(even['left']) != len(odd['left']) or len(even['right']) != len(odd['right']):
        raise ScenarioError('the team sizes change with the parity of the episode')
    return even, odd


class _Player(ctypes.Structure):
    _fields_ = [
        ('x', ctypes.c_double),
        ('y', ctypes.c_double),
        ('role', ctypes.c_int),
        ('lazy', ctypes.c_int),
        ('controllable', ctypes.c_int),
    ]


class _Description(ctypes.Structure):
    _fields_ = [
        ('version', ctypes.c_int),
        ('ball_x', ctypes.c_double),
        ('ball_y', ctypes.c_double),
        ('left_team', ctypes.POINTER(_Player)),
        ('left_size', ctypes.c_int),
        ('right_team', ctypes.POINTER(_Player)),
        ('right_size', ctypes.c_int),
        ('use_magnet', ctypes.c_int),
        ('offsides', ctypes.c_int),
        ('real_time', ctypes.c_int),
        ('left_team_difficulty', ctypes.c_double),
        ('right_team_difficulty', ctypes.c_double),
        ('deterministic', ctypes.c_int),
        ('end_episode_on_score', ctypes.c_int),
        ('end_episode_on_possession_change', ctypes.c_int),
        ('end_episode_on_out_of_play', ctypes.c_int),
        ('game_duration', ctypes.c_int),
        ('second_half', ctypes.c_int),
        ('control_all_players', ctypes.c_int),
        ('game_engine_random_seed', ctypes.c_uint),
    ]


_DESCRIPTION_VERSION = 1


def _c_description(description, keep):
    teams = []
    for side in ('left', 'right'):
        players = (_Player * len(description[side]))(
            *[_Player(*player) for player in description[side]]
        )
        keep.append(players)
        teams.append(players)
    values = description['values']
    return _Description(
        version=_DESCRIPTION_VERSION,
        ball_x=description['ball'][0],
        ball_y=description['ball'][1],
        left_team=teams[0],
        left_size=len(description['left']),
        right_team=teams[1],
        right_size=len(description['right']),
        use_magnet=values['use_magnet'],
        offsides=values['offsides'],
        real_time=False,
        left_team_difficulty=values['left_team_difficulty'],
        right_team_difficulty=values['right_team_difficulty'],
        deterministic=values['deterministic'],
        end_episode_on_score=values['end_episode_on_score'],
        end_episode_on_possession_change=values['end_episode_on_possession_change'],
        end_episode_on_out_of_play=values['end_episode_on_out_of_play'],
        game_duration=values['game_duration'],
        second_half=values['second_half'],
        control_all_players=values['control_all_players'],
        game_engine_random_seed=values['game_engine_random_seed'],
    )


def build_bundle(scenario, out, name=None, device='cpu', library=None, animations=None):
    """Writes the asset bundle without agents of a scenario file."""
    even, odd = describe(scenario)
    _warn_if_too_long([even, odd])
    if name is None:
        if isinstance(scenario, (str, os.PathLike)):
            name = os.path.splitext(os.path.basename(scenario))[0]
        else:
            name = getattr(scenario, '__module__', 'scenario').rsplit('.', 1)[-1]
    device = _backend(_automatic(device))
    path = library or library_path(device)
    if device == 'cpu':
        _check_processor(path)
    handle = ctypes.CDLL(path)
    handle.cuju_last_error.restype = ctypes.c_char_p
    build = handle.cuju_scenario_build_bundle
    build.restype = ctypes.c_int
    build.argtypes = [
        ctypes.POINTER(_Description),
        ctypes.POINTER(_Description),
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
    ]
    keep = []
    descriptions = [_c_description(even, keep), _c_description(odd, keep)]
    status = build(
        ctypes.byref(descriptions[0]),
        ctypes.byref(descriptions[1]),
        name.encode(),
        os.fsencode(animations or _ANIMATIONS),
        os.fsencode(out),
    )
    if status != 0:
        raise _from_library(handle.cuju_last_error().decode())
    return out


_HISTORY_SECONDS = 512
_LONGEST_TICKS = _HISTORY_SECONDS * 100 - 1
# GRF's default `physics_steps_per_frame`.
_DEFAULT_TICKS = 10


def _warn_if_too_long(descriptions):
    duration = max(description['values']['game_duration'] for description in descriptions)
    if duration * _DEFAULT_TICKS <= _LONGEST_TICKS:
        return
    ticks = _LONGEST_TICKS // duration
    advice = f'physics_steps_per_frame={ticks} or fewer plays it' if ticks else 'no batch plays it'
    warnings.warn(
        f'the scenario lasts {duration} steps, and a batch holds at most '
        f'{_LONGEST_TICKS // _DEFAULT_TICKS} steps of {_DEFAULT_TICKS} ticks, the default '
        f'physics_steps_per_frame, since it keeps {_HISTORY_SECONDS} seconds of position history '
        f'for each player; {advice}',
        stacklevel=3,
    )


_CACHE_FORMAT = b'cuju bundle cache 1'
_file_digests = {}


def _file_digest(path):
    status = os.stat(path)
    key = (os.path.realpath(path), status.st_size, status.st_mtime_ns)
    if key not in _file_digests:
        _file_digests[key] = _checksums.file_sha256(path)
    return _file_digests[key]


def cache_directory() -> str:
    """Returns the directory of the bundle cache, in the user's cache folder."""
    if os.name == 'nt':
        base = os.environ.get('LOCALAPPDATA') or os.path.join(
            os.path.expanduser('~'), 'AppData', 'Local'
        )
        return os.path.join(base, 'cuju', 'bundles')
    if sys.platform == 'darwin':
        return os.path.join(os.path.expanduser('~'), 'Library', 'Caches', 'cuju', 'bundles')
    base = os.environ.get('XDG_CACHE_HOME') or os.path.join(os.path.expanduser('~'), '.cache')
    return os.path.join(base, 'cuju', 'bundles')


def cached_bundle(scenario, directory=None, device='cpu', library=None, animations=None):
    """Returns the bundle of a scenario file in a cache directory, which builds it only once."""
    if directory is None:
        directory = cache_directory()
    name = os.path.splitext(os.path.basename(scenario))[0]
    with open(scenario, 'rb') as handle:
        source = handle.read()
    key = hashlib.sha256()
    for part in (
        _CACHE_FORMAT,
        source,
        name.encode(),
        _file_digest(animations or _ANIMATIONS).encode(),
        _file_digest(library or library_path(device)).encode(),
        _file_digest(os.path.abspath(__file__)).encode(),
    ):
        key.update(len(part).to_bytes(8, 'little') + part)
    path = os.path.join(directory, f'{name}-{key.hexdigest()[:40]}.cjb')
    if os.path.isfile(path):
        return path
    os.makedirs(directory, exist_ok=True)
    descriptor, partial = tempfile.mkstemp(prefix=f'.{name}-', suffix='.partial', dir=directory)
    os.close(descriptor)
    try:
        build_bundle(
            scenario, partial, name=name, device=device, library=library, animations=animations
        )
        os.replace(partial, path)
    finally:
        if os.path.exists(partial):
            os.remove(partial)
    return path


def _batch_bundle(scenario, device, library, animations=None):
    directory = cache_directory()
    try:
        path = cached_bundle(
            scenario, directory, device=device, library=library, animations=animations
        )
    except OSError as error:
        warnings.warn(
            f'the bundle cache {directory!r} cannot be used ({error}), so the batch builds the '
            'scenario file without it',
            stacklevel=3,
        )
        return None
    _log.info(f'the bundle of {scenario} is {path}, from the bundle cache')
    return path
