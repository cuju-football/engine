# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

import functools
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from ..errors import ArgumentError, LibraryError
from ._constants import JOINTS

VERSION = 1
_JOINTS = len(JOINTS)
_OFFICIALS = 3


def _player() -> list[tuple[Any, ...]]:
    return [
        ('team', '<i4'),
        ('slot', '<i4'),
        ('present', '<i4'),
        ('role', '<i4'),
        ('cards', '<i4', (2,)),
        ('controlled', '<i4'),
        ('designated', '<i4'),
        ('animation', '<i4'),
        ('touch', '<i4'),
        ('height', '<f4'),
        ('root_position', '<f4', (3,)),
        ('root_rotation', '<f4', (4,)),
        ('joints', '<f4', (_JOINTS, 3)),
        ('joint_rotations', '<f4', (_JOINTS, 4)),
    ]


@functools.cache
def frame_dtype(team_players: int) -> np.dtype:
    """Returns the type of one environment's scene, which the library writes byte for byte."""
    entries = 2 * team_players + _OFFICIALS
    return np.dtype(
        [
            ('version', '<u4'),
            ('episode', '<i4'),
            ('step', '<i4'),
            ('steps_per_second', '<i4'),
            ('duration', '<i4'),
            ('match_ms', '<i4'),
            ('half', '<i4'),
            ('in_play', '<i4'),
            ('score', '<i4', (2,)),
            ('game_mode', '<i4'),
            ('set_piece', '<i4'),
            ('restart_position', '<f4', (3,)),
            ('possession', '<i4', (2,)),
            ('last_touch', '<i4', (3,)),
            ('offside_player', '<i4'),
            ('offside_called', '<i4'),
            ('foul', '<i4', (3,)),
            ('goal', '<i4', (2,)),
            ('ball_position', '<f4', (3,)),
            ('ball_rotation', '<f4', (4,)),
            ('ball_touches_net', '<i4'),
            ('card_kind', '<i4'),
            ('card_position', '<f4', (3,)),
            ('card_rotation', '<f4', (4,)),
            ('players', _player(), (entries,)),
        ]
    )


def check_library(library: Any) -> int:
    """Returns the room of a team in the library's scene, after checking its version and size."""
    team_players = int(library.cuju_scene_team_players())
    if (
        int(library.cuju_scene_state_version()) != VERSION
        or int(library.cuju_scene_state_size()) != frame_dtype(team_players).itemsize
    ):
        raise LibraryError(
            f'expected a scene of version {VERSION} with {frame_dtype(team_players).itemsize} '
            f'bytes, the library holds version {library.cuju_scene_state_version()} with '
            f'{library.cuju_scene_state_size()}; reinstall cuju'
        )
    return team_players


class Frames:
    """The exact scene of matches, as numpy arrays.

    Every field is an array whose leading axes are the shape of the frames, `[environments]` for
    the scene now and `[steps, environments]` for a followed or replayed match. The fields of
    the players have the axis `[players]` after them, the left team's room, then the right
    team's and then the three officials. A player's index in it is what `possession`,
    `last_touch`, `offside_player`, `foul` and `goal` hold, and -1 holds for none.

    Attributes:
        version: The version of the fields, which a new release only extends.
        data: The structured array that holds the fields.
    """

    def __init__(self, data: np.ndarray) -> None:
        if data.dtype.names is None or 'players' not in data.dtype.names:
            raise TypeError(f'expected the structured array of a scene, found {data.dtype}')
        self.data = data

    @property
    def version(self) -> int:
        return VERSION

    @property
    def shape(self) -> tuple[int, ...]:
        """The leading axes, one for each of steps and environments that the frames span."""
        return self.data.shape

    @property
    def team_players(self) -> int:
        """The room of a team among the players of the frames."""
        return (self.data.dtype['players'].shape[0] - _OFFICIALS) // 2

    @property
    def names(self) -> tuple[str, ...]:
        """The names of every field, those of the match and the ball, then those of the players."""
        players = self.data.dtype['players'].base.names
        return tuple(name for name in self.data.dtype.names if name != 'players') + tuple(players)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: Any) -> 'Frames':
        chosen = self.data[index]
        if not isinstance(chosen, np.ndarray):
            chosen = np.asarray(chosen)
        return Frames(chosen)

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_') or name == 'data':
            raise AttributeError(name)
        names = self.data.dtype.names
        if name in names:
            return self.data[name]
        players = self.data['players']
        if name in players.dtype.names:
            return players[name]
        raise AttributeError(f'the frames hold no field {name!r}')

    def __dir__(self) -> list[str]:
        return sorted({*super().__dir__(), *self.names})

    def __repr__(self) -> str:
        return f'Frames(shape={self.shape}, version={VERSION})'

    @classmethod
    def stack(cls, frames: Sequence['Frames']) -> 'Frames':
        """Joins frames along a new leading axis, such as the scenes of the steps of a match."""
        return cls(np.stack([item.data for item in frames]))

    @classmethod
    def from_bytes(cls, raw: bytes, team_players: int, count: int) -> 'Frames':
        """Reads the frames of `count` environments from what the library wrote."""
        return cls(np.frombuffer(raw, frame_dtype(team_players), count).copy())


def environments_of(batch: Any, environments: Iterable[int] | int | None) -> list[int]:
    count = int(batch.environment_count)
    if environments is None:
        return list(range(count))
    if isinstance(environments, (int, np.integer)):
        environments = [environments]
    chosen = [int(index) for index in environments]
    outside = [index for index in chosen if not 0 <= index < count]
    if outside:
        raise ArgumentError(f'expected environments from 0 to {count - 1}, found {outside}')
    return chosen
