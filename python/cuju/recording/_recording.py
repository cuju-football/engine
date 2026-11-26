# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .._version import __version__
from ..types import ActionSet, Device, Representation
from . import _files
from ._errors import RecordingError

ACTION_SETS = {'default': 19, 'v2': 20, 'full': 33}
_NUMERIC = 'biufc'
_REQUIRED = ('actions', 'rewards', 'done', 'end')


def engine_version() -> str:
    return __version__


def _name(kind: str, name: Any) -> str:
    if not isinstance(name, str) or not name or '/' in name:
        raise RecordingError(f'expected a {kind} name without a slash, found {name!r}')
    return name


def _table(kind: str, values: Mapping[str, Any] | None, rows: int) -> dict[str, np.ndarray]:
    table = {}
    for name, value in (values or {}).items():
        array = np.asarray(value)
        if array.dtype.kind not in _NUMERIC:
            raise RecordingError(f'expected numbers in {kind} {name!r}, found {array.dtype}')
        if array.ndim < 1 or array.shape[0] != rows:
            raise RecordingError(
                f'expected {rows} rows in {kind} {name!r}, found the shape {array.shape}'
            )
        table[_name(kind, name)] = array
    return table


def _pair(kind: str, value: Any, convert: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise RecordingError(f'expected two {kind}, found {value!r}')
    return [None if item is None else convert(item) for item in value]


class Recording:
    """One match episode of one environment, with its actions, rewards and optional states."""

    def __init__(
        self,
        scenario: str,
        left_agents: int,
        actions: Any,
        rewards: Any = None,
        done: Any = None,
        *,
        right_agents: int = 0,
        action_set: ActionSet = 'default',
        difficulty: Sequence[float | None] | None = None,
        players: Sequence[int] | None = None,
        end: int | None = None,
        state: Any = None,
        step: Mapping[str, Any] | None = None,
        array: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
        teams: Sequence[str] | None = None,
        reward_terms: str = 'scoring',
        engine: str | None = None,
    ) -> None:
        if not isinstance(scenario, str) or not scenario:
            raise RecordingError(f'expected a scenario name, found {scenario!r}')
        for name, count in (('left_agents', left_agents), ('right_agents', right_agents)):
            if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
                raise RecordingError(f'expected {name} to be a count, found {count!r}')
        if action_set not in ACTION_SETS:
            raise RecordingError(
                f'expected an action set from {sorted(ACTION_SETS)}, found {action_set!r}'
            )
        self.scenario = scenario
        self.left_agents = int(left_agents)
        self.right_agents = int(right_agents)
        self.action_set = action_set
        self.reward_terms = reward_terms
        self.engine = engine
        agents = self.left_agents + self.right_agents

        actions = np.asarray(actions)
        if actions.ndim != 2 or actions.shape[1] != agents:
            raise RecordingError(
                f'expected actions of the shape [steps, {agents}], found {actions.shape}'
            )
        if actions.dtype.kind not in 'iub':
            raise RecordingError(f'expected whole number actions, found {actions.dtype}')
        if actions.size and (actions.min() < -1 or actions.max() > np.iinfo(np.int16).max):
            raise RecordingError('expected actions from -1 to 32767')
        steps = actions.shape[0]
        self.actions = actions.astype(np.int16)

        if rewards is None:
            rewards = np.zeros((steps, agents), np.float32)
        rewards = np.asarray(rewards)
        if rewards.shape != (steps, agents) or rewards.dtype.kind not in 'iuf':
            raise RecordingError(
                f'expected numeric rewards of the shape {(steps, agents)}, found {rewards.shape}'
            )
        self.rewards = rewards.astype(np.float32)

        if end is not None and end not in (0, 1, 2):
            raise RecordingError(f'expected the end 0, 1 or 2, found {end!r}')
        if done is None:
            done = np.zeros(steps, bool)
            if steps:
                done[-1] = bool(end)
        done = np.asarray(done)
        if done.shape != (steps,) or done.dtype.kind not in 'bui':
            raise RecordingError(f'expected done flags of the shape {(steps,)}, found {done.shape}')
        done = done.astype(bool)
        if done[:-1].any():
            raise RecordingError('expected done at the last step alone, found it earlier')
        last = bool(done[-1]) if steps else False
        if end is None:
            end = 1 if last else 0
        elif bool(end) != last:
            raise RecordingError(f'expected done at the last step to match the end {end}')
        self.done = done
        self.end = int(end)

        if state is not None and not isinstance(state, bytes):
            state = np.asarray(state, np.uint8).tobytes()
        if state is not None and not state:
            raise RecordingError('expected a state with bytes, found none')
        self.state = state
        self.step = _table('step field', step, steps + 1)
        self.array = _table('array', array, steps + 1)

        self.difficulty = _pair('difficulties', difficulty, float)
        self.players = _pair('player counts', players, int)
        self.teams = _pair('team names', teams, str)
        self.provenance = dict(provenance or {})
        try:
            _files.dumps(self.provenance)
        except (TypeError, ValueError) as error:
            raise RecordingError(f'expected provenance that JSON holds, found {error}') from error

    @property
    def steps(self) -> int:
        """The number of steps."""
        return len(self.actions)

    @property
    def header(self) -> dict[str, Any]:
        """The header as the file holds it."""
        header: dict[str, Any] = {
            'format': _files.FORMAT,
            'version': _files.VERSION,
            'scenario': self.scenario,
            'left_agents': self.left_agents,
            'right_agents': self.right_agents,
            'action_set': self.action_set,
            'reward_terms': self.reward_terms,
            'difficulty': self.difficulty,
            'steps': self.steps,
            'players': None
            if self.players is None
            else {'left': self.players[0], 'right': self.players[1]},
            'engine': self.engine,
            'provenance': self.provenance,
        }
        if self.teams is not None:
            header['teams'] = self.teams
        return header

    def __repr__(self) -> str:
        return (
            f'Recording({self.scenario!r}, {self.steps} steps, {self.left_agents} left and '
            f'{self.right_agents} right agents, end {self.end})'
        )

    def write(self, target: Any = None) -> bytes | None:
        """Writes the file to a path or an open binary file, or returns its bytes without one."""
        members: dict[str, np.ndarray] = {}
        if self.state is not None:
            members['state'] = np.frombuffer(self.state, np.uint8)
        members['actions'] = self.actions
        members['rewards'] = self.rewards
        members['done'] = self.done
        members['end'] = np.asarray(self.end, np.uint8)
        for name, array in self.step.items():
            members[f'step/{name}'] = array
        for name, array in self.array.items():
            members[f'array/{name}'] = array
        return _files.write_archive(self.header, members, target)

    def replay(
        self,
        representation: Representation = 'simple115v2',
        device: Device = 'cpu',
        scenario: Any = None,
    ) -> Any:
        """Plays the state and actions in the engine and checks the stored steps."""
        from . import _replay

        return _replay.replay(self, representation, device, scenario)

    def with_steps(self, device: Device = 'cpu', scenario: Any = None) -> 'Recording':
        """Returns a copy whose step fields come from a replay of the state and actions."""
        from . import _replay

        return _replay.with_steps(self, device, scenario)

    def _copy(self, **changes: Any) -> 'Recording':
        values: dict[str, Any] = {
            'scenario': self.scenario,
            'left_agents': self.left_agents,
            'actions': self.actions,
            'rewards': self.rewards,
            'done': self.done,
            'right_agents': self.right_agents,
            'action_set': self.action_set,
            'difficulty': self.difficulty,
            'players': self.players,
            'end': self.end,
            'state': self.state,
            'step': self.step,
            'array': self.array,
            'provenance': self.provenance,
            'teams': self.teams,
            'reward_terms': self.reward_terms,
            'engine': self.engine,
        }
        values.update(changes)
        return Recording(**values)


def load(source: Any) -> tuple[Recording, list[str]]:
    header, members = _files.read_archive(source)
    for name in _REQUIRED:
        if name not in members:
            raise RecordingError(f'expected the member {name}.npy, found none')
    try:
        players = header.get('players')
        values: dict[str, Any] = {
            'scenario': header['scenario'],
            'left_agents': header['left_agents'],
            'actions': members['actions'],
            'rewards': members['rewards'],
            'done': members['done'],
            'right_agents': header.get('right_agents', 0),
            'action_set': header.get('action_set', 'default'),
            'difficulty': header.get('difficulty'),
            'players': None if players is None else [players['left'], players['right']],
            'end': int(members['end'].reshape(())),
            'state': members['state'].tobytes() if 'state' in members else None,
            'step': {n[5:]: a for n, a in members.items() if n.startswith('step/')},
            'array': {n[6:]: a for n, a in members.items() if n.startswith('array/')},
            'provenance': header.get('provenance') or {},
            'teams': header.get('teams'),
            'reward_terms': header.get('reward_terms', 'scoring'),
            'engine': header.get('engine'),
        }
    except (KeyError, TypeError) as error:
        raise RecordingError(f'expected a complete header, found {error} missing') from error
    except ValueError as error:
        raise RecordingError(f'expected end.npy to hold one number, found {error}') from error
    match = Recording(**values)
    if header.get('steps') != match.steps:
        raise RecordingError(
            f'expected {header.get("steps")} steps, found {match.steps} rows of actions'
        )
    known = {'state', *_REQUIRED}
    unknown = sorted(
        name for name in members if name not in known and not name.startswith(('step/', 'array/'))
    )
    return match, unknown


def read(source: Any) -> 'Recording':
    """Reads a recording from a path, an open binary file or bytes."""
    return load(source)[0]
