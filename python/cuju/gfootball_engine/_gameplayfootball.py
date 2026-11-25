# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""GRF's module `_gameplayfootball` over cuju's engine, with its names, types and conversions.

Every class, enumeration and method keeps the behaviour of GRF's boost.python module, so GRF's
Python runs on it unchanged. A call that GRF answers with a fatal error raises instead.
"""

from __future__ import annotations

import ctypes
import os
import pickle
import struct
import threading
import time
import weakref

from .. import errors as _errors
from ..scenarios import ScenarioError
from . import _errors as _own_errors
from . import _native, _pickle_names, _render_hooks

__all__ = [
    'ControllerInfo',
    'ControllerInfoVec',
    'FloatVec',
    'FormationEntry',
    'FormationEntryVec',
    'GameConfig',
    'GameEnv',
    'GameState',
    'IntVec',
    'PlayerInfo',
    'PlayerInfoVec',
    'Position',
    'ScenarioConfig',
    'SharedInfo',
    'StringVector',
    'Vector3',
    'e_BackendAction',
    'e_GameMode',
    'e_PlayerRole',
    'e_RenderingMode',
    'e_Team',
]

# The name that GRF's reprs and messages print for this module.
_NAME = '_gameplayfootball'
_FLOAT = struct.Struct('<f')
_PICKLING = (
    'Pickling of "_gameplayfootball.{}" instances is not enabled '
    '(http://www.boost.org/libs/python/doc/v2/pickle.html)'
)
_STRING = 'std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >'
# GRF: `FORMATION_Y_SCALE`, a float.
_Y_SCALE = _FLOAT.unpack(_FLOAT.pack(-2.36))[0]


def _float32(value):
    """Returns `value` rounded to float, as a C cast rounds it, infinite past the range."""
    try:
        return _FLOAT.unpack(_FLOAT.pack(float(value)))[0]
    except OverflowError:
        return float('inf') if value > 0 else float('-inf')


class _MismatchError(Exception):
    """An argument that the C++ signature does not take."""


def _type_name(value):
    kind = type(value)
    if kind.__module__ in ('builtins', __name__, _NAME) or kind.__flags__ & (1 << 9):
        return kind.__name__
    return f'{kind.__module__}.{kind.__name__}'


def _signature_error(owner, method, arguments, signature):
    names = ', '.join(_type_name(value) for value in arguments)
    return _own_errors.SignatureError(
        f'Python argument types in\n    {owner}.{method}({names})\n'
        f'did not match C++ signature:\n    {signature}'
    )


# Each converter returns the C++ value of an argument or raises `_MismatchError`.
def _to_float(value):
    if isinstance(value, (int, float)):
        return _float32(value)
    raise _MismatchError


def _to_int(value):
    if not isinstance(value, int):
        raise _MismatchError
    if value > 2**31 - 1:
        raise OverflowError('bad numeric conversion: positive overflow')
    if value < -(2**31):
        raise OverflowError('bad numeric conversion: negative overflow')
    return int(value)


def _to_unsigned(value):
    if not isinstance(value, int):
        raise _MismatchError
    if value < 0:
        raise OverflowError("can't convert negative value to unsigned int")
    if value > 2**32 - 1:
        raise OverflowError('bad numeric conversion: positive overflow')
    return int(value)


def _to_long(value):
    if not isinstance(value, int):
        raise _MismatchError
    if not -(2**63) <= value < 2**63:
        raise OverflowError('Python int too large to convert to C long')
    return int(value)


def _to_bool(value):
    if value is None:
        return False
    if isinstance(value, int):
        return bool(value)
    raise _MismatchError


def _to_string(value):
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode('utf-8')
    raise _MismatchError


def _instance(kind):
    def convert(value):
        if not isinstance(value, kind):
            raise _MismatchError
        return value

    return convert


def _arguments(owner, method, signature, arguments, keywords, converters):
    """Converts the positional `arguments` of a call as boost.python does, or raises its error."""
    if keywords or len(arguments) != len(converters):
        raise _signature_error(owner, method, arguments, signature)
    try:
        return [convert(value) for convert, value in zip(converters, arguments, strict=True)]
    except _MismatchError:
        raise _signature_error(owner, method, arguments, signature) from None


def _setter(owner_type, cpp_type, convert):
    """Returns the conversion of a property's value, raising the error of boost.python's setter."""

    def apply(instance, value):
        try:
            return convert(value)
        except _MismatchError:
            raise _signature_error(
                'None',
                'None',
                (instance, value),
                f'None({owner_type} {{lvalue}}, {cpp_type})',
            ) from None

    return apply


def _no_setter(owner):
    def refuse(instance, value):
        raise AttributeError(f"property of '{owner}' object has no setter")

    return refuse


def _no_deleter(owner):
    def refuse(instance):
        raise AttributeError(f"property of '{owner}' object has no deleter")

    return refuse


class _Instance:
    """The base of every class, which prints as boost.python's and refuses pickling."""

    def __repr__(self):
        return f'<{_NAME}.{type(self).__name__} object at {hex(id(self))}>'

    __str__ = __repr__

    def __reduce_ex__(self, protocol):
        raise RuntimeError(_PICKLING.format(type(self).__name__))

    def __reduce__(self):
        raise RuntimeError(_PICKLING.format(type(self).__name__))


# Enumerations.


class enum(int):
    """The base of every enumeration, as boost.python's `enum`."""

    names: dict
    values: dict

    def __new__(cls, value=0):
        return super().__new__(cls, int(value))

    def __repr__(self):
        name = self.__dict__.get('name')
        kind = type(self).__name__
        return f'{_NAME}.{kind}.{name}' if name is not None else f'{_NAME}.{kind}({int(self)})'

    def __str__(self):
        name = self.__dict__.get('name')
        return name if name is not None else repr(self)

    def __format__(self, spec):
        return str(self) if not spec else int.__format__(int(self), spec)

    def __reduce_ex__(self, protocol):
        if protocol < 2:
            raise pickle.PicklingError(
                "Can't pickle <class 'Boost.Python.enum'>: import of module 'Boost.Python' failed"
            )
        return (type(self), (int(self),))

    __hash__ = int.__hash__


enum.__module__ = 'Boost.Python'


def _make_enum(name, pairs):
    kind = type(name, (enum,), {'__module__': __name__, '__qualname__': name})
    kind.names = {}
    kind.values = {}
    for label, value in pairs:
        member = kind(value)
        member.name = label
        kind.names[label] = member
        kind.values[value] = member
        setattr(kind, label, member)
    return kind


def _enum_value(kind, value):
    """The named value of `kind` for `value`, or an unnamed one, as boost.python converts it."""
    member = kind.values.get(value)
    return member if member is not None and int(member) == value else kind(value)


GameState = _make_enum(
    'GameState',
    [('game_created', 0), ('game_initiated', 1), ('game_running', 2), ('game_done', 3)],
)
e_PlayerRole = _make_enum(
    'e_PlayerRole',
    [
        (f'e_PlayerRole_{role}', index)
        for index, role in enumerate(('GK', 'CB', 'LB', 'RB', 'DM', 'CM', 'LM', 'RM', 'AM', 'CF'))
    ],
)
e_RenderingMode = _make_enum(
    'e_RenderingMode', [('e_Disabled', 0), ('e_Onscreen', 1), ('e_Offscreen', 2)]
)
e_GameMode = _make_enum(
    'e_GameMode',
    [
        (f'e_GameMode_{mode}', index)
        for index, mode in enumerate(
            ('Normal', 'KickOff', 'GoalKick', 'FreeKick', 'Corner', 'ThrowIn', 'Penalty')
        )
    ],
)
_ACTIONS = (
    'idle left top_left top top_right right bottom_right bottom bottom_left long_pass high_pass '
    'short_pass '
    'shot keeper_rush sliding pressure team_pressure switch sprint dribble release_direction '
    'release_long_pass release_high_pass release_short_pass release_shot release_keeper_rush '
    'release_sliding release_pressure release_team_pressure release_switch release_sprint '
    'release_dribble builtin_ai'
).split()
e_BackendAction = _make_enum(
    'e_BackendAction', [(name, index) for index, name in enumerate(_ACTIONS)]
)
e_Team = _make_enum('e_Team', [('e_Left', 0), ('e_Right', 1)])


# Records, which hold the values that several wrappers may share. A record of a `ScenarioConfig`
# counts its changes in `version`, and every record inside it points at it through `root`.


class _Root:
    __slots__ = ('version',)

    def __init__(self):
        self.version = 0


def _touch(record):
    if record.root is not None:
        record.root.version += 1


class _VectorRecord:
    __slots__ = ('root', 'values')

    def __init__(self, values=(0.0, 0.0, 0.0), root=None):
        self.values = list(values)
        self.root = root


class _EntryRecord:
    __slots__ = ('controllable', 'lazy', 'position', 'role', 'root', 'start_position')

    def copy(self, root=None):
        record = _EntryRecord()
        record.position = list(self.position)
        record.start_position = list(self.start_position)
        record.role = self.role
        record.lazy = self.lazy
        record.controllable = self.controllable
        record.root = root
        return record

    def same(self, other):
        # GRF: `FormationEntry::operator==` leaves out the start position.
        return (
            self.role == other.role
            and self.lazy == other.lazy
            and self.position == other.position
            and self.controllable == other.controllable
        )


class Vector3(_Instance):
    """GRF's `blunted::Vector3`, three floats that `ScenarioConfig.ball_position` holds."""

    def __init__(self, *arguments, **keywords):
        x, y, z = _arguments(
            'Vector3',
            '__init__',
            '__init__(_object*, float, float, float)',
            (self, *arguments),
            keywords,
            [_instance(Vector3), _to_float, _to_float, _to_float],
        )[1:]
        self._record = _VectorRecord((x, y, z))

    @classmethod
    def _over(cls, record):
        vector = cls.__new__(cls)
        vector._record = record
        return vector

    def __getitem__(self, *arguments):
        (index,) = _arguments(
            'Vector3',
            '__getitem__',
            '__getitem__(blunted::Vector3 {lvalue}, int)',
            (self, *arguments),
            {},
            [_instance(Vector3), _to_int],
        )[1:]
        if not 0 <= index <= 2:
            # GRF logs a fatal error, which ends the process.
            raise IndexError(f'expected a Vector3 index from 0 to 2, found {index}')
        return self._record.values[index]

    def __setitem__(self, *arguments):
        index, value = _arguments(
            'Vector3',
            '__setitem__',
            '__setitem__(blunted::Vector3 {lvalue}, int, float)',
            (self, *arguments),
            {},
            [_instance(Vector3), _to_int, _to_float],
        )[1:]
        if not 0 <= index <= 2:
            raise IndexError(f'expected a Vector3 index from 0 to 2, found {index}')
        self._record.values[index] = value
        _touch(self._record)


class FormationEntry(_Instance):
    """GRF's `FormationEntry(x, y, role, lazy, controllable)`, in environment coordinates."""

    def __init__(self, *arguments, **keywords):
        x, y, role, lazy, controllable = _arguments(
            'FormationEntry',
            '__init__',
            '__init__(_object*, float, float, e_PlayerRole, bool, bool)',
            (self, *arguments),
            keywords,
            _ENTRY_ARGUMENTS,
        )[1:]
        record = _EntryRecord()
        # GRF: `position(x, y * FORMATION_Y_SCALE, 0)`, a float product.
        record.position = [x, _float32(y * _Y_SCALE), 0.0]
        record.start_position = list(record.position)
        record.role = int(role)
        record.lazy = lazy
        record.controllable = controllable
        record.root = None
        self._record = record

    @classmethod
    def _over(cls, record):
        entry = cls.__new__(cls)
        entry._record = record
        return entry

    @property
    def role(self):
        return _enum_value(e_PlayerRole, self._record.role)

    role = role.setter(_no_setter('FormationEntry')).deleter(_no_deleter('FormationEntry'))

    @property
    def position(self):
        # GRF: `position_env()`, a new vector whose writes change nothing.
        x, y, z = self._record.position
        return Vector3._over(_VectorRecord((x, _float32(y / _Y_SCALE), z)))

    position = position.setter(_no_setter('FormationEntry')).deleter(_no_deleter('FormationEntry'))

    def _get_lazy(self):
        return self._record.lazy

    def _set_lazy(self, value):
        self._record.lazy = _setter('FormationEntry', 'bool', _to_bool)(self, value)
        _touch(self._record)

    lazy = property(_get_lazy, _set_lazy, _no_deleter('FormationEntry'))

    def _get_controllable(self):
        return self._record.controllable

    def _set_controllable(self, value):
        self._record.controllable = _setter('FormationEntry', 'bool', _to_bool)(self, value)
        _touch(self._record)

    controllable = property(_get_controllable, _set_controllable, _no_deleter('FormationEntry'))


_ENTRY_ARGUMENTS = [
    _instance(FormationEntry),
    _to_float,
    _to_float,
    _instance(e_PlayerRole),
    _to_bool,
    _to_bool,
]


# Vectors, as boost.python's `vector_indexing_suite` gives them.


class _ListRecord:
    __slots__ = ('items', 'root')

    def __init__(self, items=None, root=None):
        self.items = items if items is not None else []
        self.root = root


class _Vector(_Instance):
    """A `std::vector` of GRF, with the semantics of boost.python's `vector_indexing_suite`.

    `_store` turns a Python value into a stored item or raises `_MismatchError`, `_load` turns a
    stored item into the value that indexing returns, and `_same` is the C++ equality of two items.
    """

    def __init__(self, *arguments, **keywords):
        name = type(self).__name__
        _arguments(
            name,
            '__init__',
            '__init__(_object*)',
            (self, *arguments),
            keywords,
            [_instance(_Vector)],
        )
        self._record = _ListRecord()

    @classmethod
    def _over(cls, record):
        vector = cls.__new__(cls)
        vector._record = record
        return vector

    def _store(self, value):
        raise NotImplementedError

    def _load(self, item):
        return item

    def _same(self, first, second):
        return first == second

    def _owned(self, item):
        return item

    def _index(self, index):
        items = self._record.items
        if isinstance(index, slice):
            if index.step is not None:
                raise IndexError('slice step size not supported.')
            return index
        if not isinstance(index, int):
            raise TypeError('Invalid index type')
        if index < 0:
            index += len(items)
        if not 0 <= index < len(items):
            raise IndexError('Index out of range')
        return index

    def __len__(self):
        return len(self._record.items)

    def __getitem__(self, index):
        position = self._index(index)
        items = self._record.items
        if isinstance(position, slice):
            return type(self)._over(_ListRecord([self._copy(item) for item in items[position]]))
        return self._load(items[position])

    def _copy(self, item):
        return item.copy() if hasattr(item, 'copy') else item

    def __setitem__(self, index, value):
        position = self._index(index)
        items = self._record.items
        try:
            if isinstance(position, slice):
                stored = [self._adopt(self._store(element)) for element in value]
            else:
                stored = self._adopt(self._store(value))
        except _MismatchError:
            raise TypeError('Invalid assignment') from None
        items[position] = stored
        _touch(self._record)

    def __delitem__(self, index):
        position = self._index(index)
        del self._record.items[position]
        _touch(self._record)

    def __iter__(self):
        for item in list(self._record.items):
            yield self._load(item)

    def __contains__(self, value):
        try:
            stored = self._store(value)
        except _MismatchError:
            return False
        return any(self._same(item, stored) for item in self._record.items)

    def _adopt(self, item):
        if hasattr(item, 'root'):
            item.root = self._record.root
        return item

    def append(self, *arguments):
        if len(arguments) != 1:
            raise _signature_error(type(self).__name__, 'append', (self, *arguments), 'append')
        try:
            stored = self._adopt(self._store(arguments[0]))
        except _MismatchError:
            raise TypeError('Attempting to append an invalid type') from None
        self._record.items.append(stored)
        _touch(self._record)

    def extend(self, *arguments):
        if len(arguments) != 1:
            raise _signature_error(type(self).__name__, 'extend', (self, *arguments), 'extend')
        try:
            stored = [self._adopt(self._store(element)) for element in arguments[0]]
        except _MismatchError:
            raise TypeError('Incompatible Data Type') from None
        self._record.items.extend(stored)
        _touch(self._record)


class FloatVec(_Vector):
    """GRF's `std::vector<float>`."""

    __slots__ = ()

    def _store(self, value):
        return _to_float(value)


class IntVec(_Vector):
    """GRF's `std::vector<int>`."""

    __slots__ = ()

    def _store(self, value):
        return _to_int(value)


class StringVector(_Vector):
    """GRF's `std::vector<std::string>`."""

    __slots__ = ()

    def _store(self, value):
        return _to_string(value).decode('utf-8', 'surrogateescape')


class FormationEntryVec(_Vector):
    """GRF's `std::vector<FormationEntry>`, whose items stay shared with the vector."""

    __slots__ = ()

    def _store(self, value):
        if not isinstance(value, FormationEntry):
            raise _MismatchError
        return value._record.copy()

    def _load(self, item):
        return FormationEntry._over(item)

    def _same(self, first, second):
        return first.same(second)


class Position(_Instance):
    """GRF's `Position`, a raw vector of the engine that indexes in environment coordinates."""

    def __init__(self, *arguments, **keywords):
        _arguments(
            'Position',
            '__init__',
            '__init__(_object*)',
            (self, *arguments),
            keywords,
            [_instance(Position)],
        )
        self._values = (0.0, 0.0, 0.0)

    @classmethod
    def _of(cls, values):
        position = cls.__new__(cls)
        position._values = tuple(values)
        return position

    def __getitem__(self, *arguments):
        (index,) = _arguments(
            'Position',
            '__getitem__',
            '__getitem__(Position {lvalue}, int)',
            (self, *arguments),
            {},
            [_instance(Position), _to_int],
        )[1:]
        # GRF: `X_FIELD_SCALE` and `Y_FIELD_SCALE` are doubles, and z divides by the integer 1.
        if index == 0:
            return _float32(self._values[0] / 54.4)
        if index == 1:
            return _float32(self._values[1] / -83.6)
        if index == 2:
            return self._values[2]
        raise IndexError(f'expected a Position index from 0 to 2, found {index}')

    def __str__(self):
        x, y, z = self._values
        return f'{x:f},{y:f},{z:f}'


class PlayerInfo(_Instance):
    """GRF's `PlayerInfo`, one player of `SharedInfo`."""

    def __init__(self, *arguments, **keywords):
        _arguments(
            'PlayerInfo',
            '__init__',
            '__init__(_object*)',
            (self, *arguments),
            keywords,
            [_instance(PlayerInfo)],
        )
        self._values = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0, False, True, False, 0)

    @classmethod
    def _of(cls, values):
        info = cls.__new__(cls)
        info._values = values
        return info

    def _field(index, convert=None):
        def get(self):
            value = self._values[index]
            return convert(value) if convert else value

        return property(get, _no_setter('PlayerInfo'), _no_deleter('PlayerInfo'))

    position = _field(0, Position._of)
    direction = _field(1, Position._of)
    tired_factor = _field(2)
    has_card = _field(3)
    is_active = _field(4)
    designated_player = _field(5)
    role = _field(6, lambda value: _enum_value(e_PlayerRole, value))
    del _field


class PlayerInfoVec(_Vector):
    """GRF's `std::vector<PlayerInfo>`."""

    __slots__ = ()

    def _store(self, value):
        if not isinstance(value, PlayerInfo):
            raise _MismatchError
        return value._values

    def _load(self, item):
        return PlayerInfo._of(item)

    def _same(self, first, second):
        return first == second


class ControllerInfo(_Instance):
    """GRF's `ControllerInfo(controlled_player)`."""

    def __init__(self, *arguments, **keywords):
        (player,) = _arguments(
            'ControllerInfo',
            '__init__',
            '__init__(_object*, int)',
            (self, *arguments),
            keywords,
            [_instance(ControllerInfo), _to_int],
        )[1:]
        self._player = player

    @classmethod
    def _of(cls, player):
        info = cls.__new__(cls)
        info._player = player
        return info

    controlled_player = property(
        lambda self: self._player, _no_setter('ControllerInfo'), _no_deleter('ControllerInfo')
    )


class ControllerInfoVec(_Vector):
    """GRF's `std::vector<ControllerInfo>`."""

    __slots__ = ()

    def _store(self, value):
        if not isinstance(value, ControllerInfo):
            raise _MismatchError
        return value._player

    def _load(self, item):
        return ControllerInfo._of(item)

    def _same(self, first, second):
        return False


class SharedInfo(_Instance):
    """GRF's `SharedInfo`, what `GameEnv.get_info` returns."""

    def __init__(self, *arguments, **keywords):
        _arguments(
            'SharedInfo',
            '__init__',
            '__init__(_object*)',
            (self, *arguments),
            keywords,
            [_instance(SharedInfo)],
        )
        self._info = None

    @classmethod
    def _of(cls, info):
        shared = cls.__new__(cls)
        shared._info = info
        return shared

    def _read(name, convert, default):
        def get(self):
            if self._info is None:
                return default()
            return convert(self, name)

        return property(get, _no_setter('SharedInfo'), _no_deleter('SharedInfo'))

    def _team(self, name):
        cache = self.__dict__.setdefault('_teams', {})
        if name in cache:
            return PlayerInfoVec._over(cache[name])
        info = self._info
        players = getattr(info, name)
        count = info.left_size if name == 'left_team' else info.right_size
        items = []
        for index in range(count):
            player = players[index]
            items.append(
                (
                    tuple(player.position),
                    tuple(player.direction),
                    player.tired_factor,
                    bool(player.has_card),
                    bool(player.is_active),
                    bool(player.designated_player),
                    player.role,
                )
            )
        cache[name] = _ListRecord(items)
        return PlayerInfoVec._over(cache[name])

    def _controllers(self, name):
        info = self._info
        # GRF: `Match::GetState` sizes the controllers to the team of the scenario.
        count = info.left_size if name == 'left_controllers' else info.right_size
        values = getattr(info, name)
        return ControllerInfoVec._over(_ListRecord([values[index] for index in range(count)]))

    def _plain(self, name):
        return getattr(self._info, name)

    def _flag(self, name):
        return bool(getattr(self._info, name))

    ball_position = _read(
        'ball_position',
        lambda i, n: Position._of(i._info.ball_position),
        lambda: Position._of((0,) * 3),
    )
    ball_rotation = _read(
        'ball_rotation',
        lambda i, n: Position._of(i._info.ball_rotation),
        lambda: Position._of((0,) * 3),
    )
    ball_direction = _read(
        'ball_direction',
        lambda i, n: Position._of(i._info.ball_direction),
        lambda: Position._of((0,) * 3),
    )
    left_team = _read('left_team', _team, lambda: PlayerInfoVec._over(_ListRecord()))
    right_team = _read('right_team', _team, lambda: PlayerInfoVec._over(_ListRecord()))
    left_goals = _read('left_goals', _plain, lambda: 0)
    right_goals = _read('right_goals', _plain, lambda: 0)
    is_in_play = _read('is_in_play', _flag, lambda: False)
    ball_owned_team = _read('ball_owned_team', _plain, lambda: 0)
    ball_owned_player = _read('ball_owned_player', _plain, lambda: 0)
    left_controllers = _read(
        'left_controllers', _controllers, lambda: ControllerInfoVec._over(_ListRecord())
    )
    right_controllers = _read(
        'right_controllers', _controllers, lambda: ControllerInfoVec._over(_ListRecord())
    )
    game_mode = _read(
        'game_mode',
        lambda i, n: _enum_value(e_GameMode, i._info.game_mode),
        lambda: e_GameMode.e_GameMode_Normal,
    )
    step = _read('step', _plain, lambda: 0)
    del _read, _team, _controllers, _plain, _flag


# The configurations.


class _ScenarioRecord(_Root):
    __slots__ = (
        'ball',
        'cache',
        'control_all_players',
        'deterministic',
        'end_episode_on_out_of_play',
        'end_episode_on_possession_change',
        'end_episode_on_score',
        'game_duration',
        'game_engine_random_seed',
        'left_agents',
        'left_team',
        'left_team_difficulty',
        'offsides',
        'real_time',
        'reverse_team_processing',
        'right_agents',
        'right_team',
        'right_team_difficulty',
        'second_half',
        'use_magnet',
    )

    def __init__(self):
        super().__init__()
        self.ball = _VectorRecord(root=self)
        self.left_team = _ListRecord(root=self)
        self.right_team = _ListRecord(root=self)
        self.left_agents = 1
        self.right_agents = 0
        self.use_magnet = True
        self.offsides = True
        self.real_time = False
        self.game_engine_random_seed = 42
        self.reverse_team_processing = False
        self.left_team_difficulty = 1.0
        self.right_team_difficulty = _float32(0.6)
        self.deterministic = False
        self.end_episode_on_score = False
        self.end_episode_on_possession_change = False
        self.end_episode_on_out_of_play = False
        self.game_duration = 3000
        self.control_all_players = False
        self.second_half = 999999999
        # GRF's cache: computed, the dynamic player selection and the controllable counts.
        self.cache = [False, False, -1, -1]

    def assign(self, other):
        """Copies `other` in, as GRF's `ScenarioConfig` assignment does, the cache included."""
        for name in _SCENARIO_FIELDS:
            setattr(self, name, getattr(other, name))
        self.ball.values = list(other.ball.values)
        self.left_team.items = [item.copy(self) for item in other.left_team.items]
        self.right_team.items = [item.copy(self) for item in other.right_team.items]
        self.cache = list(other.cache)
        self.version += 1

    def compute_cache(self):
        """GRF: `ScenarioConfig::ComputeCache`."""
        if self.cache[0]:
            return
        left = sum(1 for item in self.left_team.items if item.controllable)
        right = sum(1 for item in self.right_team.items if item.controllable)
        dynamic = not (
            (left == self.left_agents or self.left_agents == 0)
            and (right == self.right_agents or self.right_agents == 0)
        )
        self.cache = [True, dynamic, left, right]
        self.version += 1

    def to_native(self):
        """Returns `cuju_engine_scenario` of the record and the entry arrays it points at."""
        teams = []
        for team in (self.left_team.items, self.right_team.items):
            if len(team) > _native.TEAM_PLAYERS:
                raise ScenarioError(
                    f'a team holds {len(team)} formation entries, more than the 11 players that '
                    'GRF plays, where GRF crashes'
                )
            entries = (_native.Entry * max(1, len(team)))()
            for index, item in enumerate(team):
                entry = entries[index]
                entry.position[:] = item.position
                entry.start_position[:] = item.start_position
                entry.role = item.role
                entry.lazy = item.lazy
                entry.controllable = item.controllable
            teams.append(entries)
        scenario = _native.Scenario()
        scenario.version = _native.VERSION
        scenario.ball_position[:] = self.ball.values
        scenario.left_team = ctypes.cast(teams[0], ctypes.POINTER(_native.Entry))
        scenario.left_size = len(self.left_team.items)
        scenario.right_team = ctypes.cast(teams[1], ctypes.POINTER(_native.Entry))
        scenario.right_size = len(self.right_team.items)
        for name in _SCENARIO_FIELDS:
            setattr(scenario, name, getattr(self, name))
        (
            scenario.cache_computed,
            scenario.cache_dynamic_player_selection,
            scenario.cache_controllable_left,
            scenario.cache_controllable_right,
        ) = self.cache
        return scenario, teams

    def from_native(self, scenario):
        """Copies `cuju_engine_scenario` in, its entries included."""
        for name in _SCENARIO_FIELDS:
            value = getattr(scenario, name)
            if isinstance(getattr(self, name), bool):
                value = bool(value)
            setattr(self, name, value)
        self.ball.values = list(scenario.ball_position)
        for team, entries, size in (
            (self.left_team, scenario.left_team, scenario.left_size),
            (self.right_team, scenario.right_team, scenario.right_size),
        ):
            items = []
            for index in range(size):
                entry = entries[index]
                item = _EntryRecord()
                item.position = list(entry.position)
                item.start_position = list(entry.start_position)
                item.role = entry.role
                item.lazy = bool(entry.lazy)
                item.controllable = bool(entry.controllable)
                item.root = self
                items.append(item)
            team.items = items
        self.cache = [
            bool(scenario.cache_computed),
            bool(scenario.cache_dynamic_player_selection),
            scenario.cache_controllable_left,
            scenario.cache_controllable_right,
        ]
        self.version += 1


# Each field of `ScenarioConfig` with its C++ type and its conversion.
_SCENARIO_TYPES = {
    'left_agents': ('int', _to_int),
    'right_agents': ('int', _to_int),
    'use_magnet': ('bool', _to_bool),
    'offsides': ('bool', _to_bool),
    'real_time': ('bool', _to_bool),
    'game_engine_random_seed': ('unsigned int', _to_unsigned),
    'reverse_team_processing': ('bool', _to_bool),
    'left_team_difficulty': ('float', _to_float),
    'right_team_difficulty': ('float', _to_float),
    'deterministic': ('bool', _to_bool),
    'end_episode_on_score': ('bool', _to_bool),
    'end_episode_on_possession_change': ('bool', _to_bool),
    'end_episode_on_out_of_play': ('bool', _to_bool),
    'game_duration': ('int', _to_int),
    'second_half': ('int', _to_int),
    'control_all_players': ('bool', _to_bool),
}
_SCENARIO_FIELDS = tuple(_SCENARIO_TYPES)


def _scenario_field(name, cpp_type, convert):
    check = _setter('ScenarioConfig', cpp_type, convert)

    def get(self):
        return getattr(self._record, name)

    def set_(self, value):
        setattr(self._record, name, check(self, value))
        self._record.version += 1

    return property(get, set_, _no_deleter('ScenarioConfig'))


def _cached(index):
    def get(self):
        self._record.compute_cache()
        return self._record.cache[index]

    return property(get, _no_setter('ScenarioConfig'), _no_deleter('ScenarioConfig'))


class ScenarioConfig(_Instance):
    """GRF's `ScenarioConfig`, which `ScenarioConfig.make()` creates."""

    def __init__(self, *arguments, **keywords):
        raise RuntimeError('This class cannot be instantiated from Python')

    @staticmethod
    def make(*arguments, **keywords):
        _arguments(
            'ScenarioConfig',
            'make',
            'make()',
            arguments,
            keywords,
            [],
        )
        return ScenarioConfig._over(_ScenarioRecord())

    @classmethod
    def _over(cls, record):
        config = cls.__new__(cls)
        config._record = record
        return config

    def _get_ball(self):
        return Vector3._over(self._record.ball)

    def _set_ball(self, value):
        _setter('ScenarioConfig', 'blunted::Vector3', _instance(Vector3))(self, value)
        self._record.ball.values = list(value._record.values)
        self._record.version += 1

    ball_position = property(_get_ball, _set_ball, _no_deleter('ScenarioConfig'))

    def _team_property(name):
        def get(self):
            return FormationEntryVec._over(getattr(self._record, name))

        def set_(self, value):
            _setter(
                'ScenarioConfig',
                'std::vector<FormationEntry, std::allocator<FormationEntry> >',
                _instance(FormationEntryVec),
            )(self, value)
            record = self._record
            getattr(record, name).items = [item.copy(record) for item in value._record.items]
            record.version += 1

        return property(get, set_, _no_deleter('ScenarioConfig'))

    left_team = _team_property('left_team')
    right_team = _team_property('right_team')
    del _team_property

    dynamic_player_selection = _cached(1)
    controllable_left_players = _cached(2)
    controllable_right_players = _cached(3)

    for _name, (_cpp, _convert) in _SCENARIO_TYPES.items():
        locals()[_name] = _scenario_field(_name, _cpp, _convert)
    del _name, _cpp, _convert


class _GameRecord:
    __slots__ = ('physics_steps_per_frame', 'render', 'render_resolution_x', 'render_resolution_y')

    def __init__(self):
        self.render = False
        self.physics_steps_per_frame = 10
        self.render_resolution_x = 1280
        self.render_resolution_y = 720


def _game_field(name, cpp_type, convert):
    check = _setter('GameConfig', cpp_type, convert)

    def get(self):
        return getattr(self._record, name)

    def set_(self, value):
        setattr(self._record, name, check(self, value))

    return property(get, set_, _no_deleter('GameConfig'))


class GameConfig(_Instance):
    """GRF's `GameConfig`, which `GameConfig.make()` creates."""

    def __init__(self, *arguments, **keywords):
        raise RuntimeError('This class cannot be instantiated from Python')

    @staticmethod
    def make(*arguments, **keywords):
        _arguments('GameConfig', 'make', 'make()', arguments, keywords, [])
        return GameConfig._over(_GameRecord())

    @classmethod
    def _over(cls, record):
        config = cls.__new__(cls)
        config._record = record
        return config

    render = _game_field('render', 'bool', _to_bool)
    physics_steps_per_frame = _game_field('physics_steps_per_frame', 'int', _to_int)
    render_resolution_x = _game_field('render_resolution_x', 'int', _to_int)
    render_resolution_y = _game_field('render_resolution_y', 'int', _to_int)


# GRF's tracker, which plays two engines of a process in lockstep and compares their states.


class _Tracker:
    def __init__(self):
        self.condition = threading.Condition()
        self.start = 0
        self.end = 1000000000
        self.waiting = None
        # The failure of the comparison that the waiting engine takes up when it wakes.
        self.failure = None

    def setup(self, env, start, end):
        self.start = start
        self.end = end
        env._tracker_disabled = 0
        env._tracker_position = 0

    def verify(self, env):
        if env._tracker_disabled:
            return
        env._tracker_position += 1
        position = env._tracker_position
        if position < self.start or position > self.end:
            return
        with self.condition:
            if self.waiting is None:
                if position % 10000 == 0:
                    print(f'Validating: {position}', flush=True)
                self.waiting = env
                self.failure = None
                # GRF: a wait without a predicate, which the engine that verifies ends.
                self.condition.wait()
                if self.failure is not None and self.failure[0] is env:
                    failure = self.failure[1]
                    self.failure = None
                    raise failure
                return
            env._tracker_disabled += 1
            waiting = self.waiting
            try:
                failure = self._snapshot(env, waiting, position)
            except BaseException as error:
                failure = error
            finally:
                env._tracker_disabled -= 1
                self.waiting = None
            if failure is not None:
                self.failure = (waiting, failure)
            self.condition.notify()
            if failure is not None:
                raise failure

    def _snapshot(self, env, waiting, position):
        waiting_position = waiting._tracker_position
        detail = ''
        failed = position != waiting_position
        if not failed:
            env._push()
            offset = ctypes.c_longlong(-1)
            status = _native.library().cuju_engine_verify(
                env._engine, waiting._engine, ctypes.byref(offset)
            )
            env._raise(status if status < 0 else 0)
            env._pull_cache()
            failed = status == 1
            detail = f'\nState position: {offset.value}'
        if not failed:
            return None
        return _own_errors.TrackerError(
            f'Validation range: {self.start} - {self.end}\n'
            f'Position: {position} vs {waiting_position}{detail}\n'
            'State comparison failure, where GRF ends the process'
        )


_TRACKER = _Tracker()

# The engines of finalized `GameEnv`s, each with a weak reference to its `GameEnv`, which a
# resurrected one keeps alive.
_RELEASED = []
_RELEASED_LOCK = threading.Lock()


def _free_released():
    """Frees the engines of every finalized `GameEnv` that no object refers to any more."""
    with _RELEASED_LOCK:
        kept = []
        for reference, engine in _RELEASED:
            if reference() is None:
                _native.library().cuju_engine_destroy(engine)
            else:
                kept.append((reference, engine))
        _RELEASED[:] = kept


class GameEnv(_Instance):
    """GRF's `GameEnv`, one engine, on cuju's simulation."""

    def __init__(self, *arguments, **keywords):
        _arguments(
            'GameEnv',
            '__init__',
            '__init__(_object*)',
            (self, *arguments),
            keywords,
            [_instance(GameEnv)],
        )
        from .._batch.assets import _DATA_DIRECTORY

        library = _native.library()
        _free_released()
        path = os.path.join(_DATA_DIRECTORY, 'animations.bin')
        self._engine = library.cuju_engine_create(path.encode())
        if not self._engine:
            raise _errors.LibraryError(_native.last_error())
        self._config = _ScenarioRecord()
        self._game = _GameRecord()
        self._pushed = -1
        self._match = False
        self._started = False
        self._info = _native.Info()
        self._info.version = _native.VERSION
        self._tracker_disabled = 1
        self._tracker_position = 0
        self._last_step_rendered_frames = 1
        self._render = None
        self._render_mode = False

    def __del__(self):
        # A finalizer may resurrect the engine, as GRF's core does when it returns an engine to its
        # pool during a collection, so the engine is freed only once nothing refers to it.
        engine = getattr(self, '_engine', None)
        if engine:
            with _RELEASED_LOCK:
                _RELEASED.append((weakref.ref(self), engine))

    def _raise(self, status):
        if status >= 0:
            return
        message = _native.last_error()
        if status == -2:
            raise _errors.UndefinedPathError(message)
        if status == -3:
            raise _errors.StateError(message)
        if status == -4:
            raise ScenarioError(message)
        if status == -5:
            raise _errors.LibraryError(message)
        raise _errors.ArgumentError(message)

    def _push(self):
        """Writes `config` into the engine if it changed since the engine last saw it."""
        if self._match and self._config.version != self._pushed:
            scenario, _entries = self._config.to_native()
            self._raise(_native.library().cuju_engine_set_config(self._engine, scenario))
            self._pushed = self._config.version

    def _pull_cache(self):
        """Reads the engine's cache of the scenario, which a step or a state may change."""
        cache = (ctypes.c_int * 4)()
        _native.library().cuju_engine_config_cache(self._engine, cache)
        values = [bool(cache[0]), bool(cache[1]), cache[2], cache[3]]
        if values != self._config.cache:
            self._config.cache = values
            if self._pushed == self._config.version:
                self._config.version += 1
                self._pushed = self._config.version
            else:
                self._config.version += 1

    def _ticks(self):
        return self._game.physics_steps_per_frame

    def _sync_render(self):
        """Tells the engine whether it renders, as GRF's graphics read `game_config.render`."""
        mode = self._render is not None and bool(self._game.render)
        if mode != self._render_mode:
            _native.library().cuju_engine_set_render(self._engine, 1 if mode else 0)
            self._render_mode = mode

    def start_game(self, *arguments, **keywords):
        _arguments(
            'GameEnv',
            'start_game',
            'start_game(GameEnv_Python {lvalue})',
            (self, *arguments),
            keywords,
            [_instance(GameEnv)],
        )
        self._raise(_native.library().cuju_engine_start(self._engine))
        self._started = True
        # GRF: the first reset plays `ScenarioConfig::make()`, whose ball `setConfig` scales.
        record = _ScenarioRecord()
        record.ball.values = [0.0, -0.0, 0.0]
        self._config.assign(record)
        if self._game.render:
            self._render = _render_hooks.open_session(self)
        self._sync_render()

    def reset(self, *arguments, **keywords):
        scenario, animations = _arguments(
            'GameEnv',
            'reset',
            'reset(GameEnv_Python {lvalue}, ScenarioConfig {lvalue}, bool)',
            (self, *arguments),
            keywords,
            [_instance(GameEnv), _instance(ScenarioConfig), _to_bool],
        )[1:]
        # GRF: a reset disables the tracker until `tracker_setup`.
        self._tracker_disabled += 1
        self._sync_render()
        record = scenario._record
        native, _entries = record.to_native()
        status = _native.library().cuju_engine_reset(
            self._engine, native, 1 if animations else 0, self._ticks()
        )
        self._raise(status)
        # GRF: `setConfig` scales the caller's ball in place before the copy.
        record.ball.values = list(native.ball_position)
        record.version += 1
        self._config.assign(record)
        self._pushed = self._config.version
        self._match = True
        if self._render is not None:
            self._render.reset(self, animations)

    def get_info(self, *arguments, **keywords):
        _arguments(
            'GameEnv',
            'get_info',
            'get_info(GameEnv_Python {lvalue})',
            (self, *arguments),
            keywords,
            [_instance(GameEnv)],
        )
        info = _native.Info()
        info.version = _native.VERSION
        self._raise(_native.library().cuju_engine_info(self._engine, self._ticks(), info))
        return SharedInfo._of(info)

    def get_frame(self, *arguments, **keywords):
        _arguments(
            'GameEnv',
            'get_frame',
            'get_frame(GameEnv_Python {lvalue})',
            (self, *arguments),
            keywords,
            [_instance(GameEnv)],
        )
        return _render_hooks.require(self._render).frame()

    def perform_action(self, *arguments, **keywords):
        action, left_team, player = _arguments(
            'GameEnv',
            'perform_action',
            'perform_action(GameEnv_Python {lvalue}, int, bool, int)',
            (self, *arguments),
            keywords,
            [_instance(GameEnv), _to_int, _to_bool, _to_int],
        )[1:]
        self._raise(
            _native.library().cuju_engine_action(
                self._engine, action, 1 if left_team else 0, player
            )
        )

    def sticky_action_state(self, *arguments, **keywords):
        action, left_team, player = _arguments(
            'GameEnv',
            'sticky_action_state',
            'sticky_action_state(GameEnv_Python {lvalue}, int, bool, int)',
            (self, *arguments),
            keywords,
            [_instance(GameEnv), _to_int, _to_bool, _to_int],
        )[1:]
        status = _native.library().cuju_engine_sticky(
            self._engine, action, 1 if left_team else 0, player
        )
        self._raise(status)
        return status == 1

    def step(self, *arguments, **keywords):
        _arguments(
            'GameEnv',
            'step',
            'step(GameEnv_Python {lvalue})',
            (self, *arguments),
            keywords,
            [_instance(GameEnv)],
        )
        library = _native.library()
        self._push()
        self._sync_render()
        ticks = self._ticks()
        tracked = not self._tracker_disabled
        if self._config.real_time:
            # GRF: a tick at a time, rendered on a schedule that follows the wall clock.
            started = time.monotonic()
            frames = self._last_step_rendered_frames
            for tick in range(1, ticks + 1):
                status = library.cuju_engine_ticks(self._engine, ticks, 1)
                self._raise(status)
                if status == 1 and tracked:
                    _TRACKER.verify(self)
                if tick * frames // ticks != (tick - 1) * frames // ticks:
                    self._render_now()
            elapsed = int((time.monotonic() - started) * 1000)
            if elapsed > 9 * (ticks + 1) and frames > 1:
                self._last_step_rendered_frames -= 1
            elif elapsed < 9 * (ticks - 1) and frames < ticks:
                self._last_step_rendered_frames += 1
            self._raise(library.cuju_engine_finish_step(self._engine))
        elif tracked or self._game.render:
            status = library.cuju_engine_ticks(self._engine, ticks, ticks)
            self._raise(status)
            if status == 1 and tracked:
                _TRACKER.verify(self)
            if self._game.render:
                self._render_now()
            self._raise(library.cuju_engine_finish_step(self._engine))
        else:
            self._raise(library.cuju_engine_step(self._engine, ticks))
        self._pull_cache()

    def _render_now(self):
        self._prepare_render().render(self, True)

    def _state_processed(self):
        """GRF: `ProcessState` of a save or a load makes the graphics draw the nets afresh."""
        processed = getattr(self._render, 'state_processed', None)
        if processed is not None:
            processed(self)

    def _prepare_render(self):
        """Runs GRF's `PrepareRender` on the match and returns the renderer's session."""
        session = _render_hooks.require(self._render)
        self._sync_render()
        self._raise(_native.library().cuju_engine_prepare_render(self._engine))
        return session

    def render(self, *arguments, **keywords):
        (swap_buffer,) = _arguments(
            'GameEnv',
            'render',
            'render(GameEnv_Python {lvalue}, bool)',
            (self, *arguments),
            keywords,
            [_instance(GameEnv), _to_bool],
        )[1:]
        self._prepare_render().render(self, swap_buffer)

    def get_state(self, *arguments, **keywords):
        (pickled,) = _arguments(
            'GameEnv',
            'get_state',
            f'get_state(GameEnv_Python {{lvalue}}, {_STRING})',
            (self, *arguments),
            keywords,
            [_instance(GameEnv), _to_string],
        )[1:]
        self._push()
        self._sync_render()
        pickled = _pickle_names.to_grf(pickled)
        size = ctypes.c_ulonglong()
        data = ctypes.POINTER(ctypes.c_char)()
        self._raise(
            _native.library().cuju_engine_get_state(
                self._engine, pickled, len(pickled), ctypes.byref(data), ctypes.byref(size)
            )
        )
        state = ctypes.string_at(data, size.value)
        self._pull_cache()
        self._state_processed()
        return state

    def set_state(self, *arguments, **keywords):
        (state,) = _arguments(
            'GameEnv',
            'set_state',
            f'set_state(GameEnv_Python {{lvalue}}, {_STRING})',
            (self, *arguments),
            keywords,
            [_instance(GameEnv), _to_string],
        )[1:]
        self._push()
        self._sync_render()
        offset, size = ctypes.c_ulonglong(), ctypes.c_ulonglong()
        self._raise(
            _native.library().cuju_engine_set_state(
                self._engine, state, len(state), ctypes.byref(offset), ctypes.byref(size)
            )
        )
        self._read_config()
        self._state_processed()
        return _pickle_names.from_grf(state[offset.value : offset.value + size.value])

    def _read_config(self):
        entries = [(_native.Entry * _native.TEAM_PLAYERS)() for _ in range(2)]
        scenario = _native.Scenario()
        scenario.left_team = ctypes.cast(entries[0], ctypes.POINTER(_native.Entry))
        scenario.right_team = ctypes.cast(entries[1], ctypes.POINTER(_native.Entry))
        self._raise(_native.library().cuju_engine_config(self._engine, scenario))
        self._config.from_native(scenario)
        self._pushed = self._config.version

    def tracker_setup(self, *arguments, **keywords):
        start, end = _arguments(
            'GameEnv',
            'tracker_setup',
            'tracker_setup(GameEnv_Python {lvalue}, long, long)',
            (self, *arguments),
            keywords,
            [_instance(GameEnv), _to_long, _to_long],
        )[1:]
        _TRACKER.setup(self, start, end)

    def _get_config(self):
        return ScenarioConfig._over(self._config)

    def _set_config(self, value):
        _setter('GameEnv_Python', 'ScenarioConfig', _instance(ScenarioConfig))(self, value)
        self._config.assign(value._record)

    config = property(_get_config, _set_config, _no_deleter('GameEnv'))

    def _get_game_config(self):
        return GameConfig._over(self._game)

    def _set_game_config(self, value):
        _setter('GameEnv_Python', 'GameConfig', _instance(GameConfig))(self, value)
        for name in _GameRecord.__slots__:
            setattr(self._game, name, getattr(value._record, name))

    game_config = property(_get_game_config, _set_game_config, _no_deleter('GameEnv'))

    def _get_state(self):
        return _enum_value(GameState, _native.library().cuju_engine_game_state(self._engine))

    def _set_state(self, value):
        value = _setter('GameEnv_Python', 'GameState', _instance(GameState))(self, value)
        _native.library().cuju_engine_set_game_state(self._engine, int(value))

    state = property(_get_state, _set_state, _no_deleter('GameEnv'))

    def _get_waiting(self):
        return _native.library().cuju_engine_waiting_count(self._engine)

    def _set_waiting(self, value):
        value = _setter('GameEnv_Python', 'int', _to_int)(self, value)
        _native.library().cuju_engine_set_waiting_count(self._engine, value)

    waiting_for_game_count = property(_get_waiting, _set_waiting, _no_deleter('GameEnv'))
