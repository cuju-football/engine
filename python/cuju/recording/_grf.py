# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from collections.abc import Mapping
from typing import Any

import numpy as np

from .. import dumps
from ..types import Device
from . import _files, _views
from ._errors import RecordingError
from ._recording import Recording, engine_version

# GRF: `football_action_set.action_set_dict`, the actions of each set by index.
_DEFAULT = (
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
ACTION_NAMES = {
    'default': _DEFAULT,
    'v2': (*_DEFAULT, 'builtin_ai'),
    'full': (
        *_DEFAULT,
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
# GRF: a dump stores the actions of the right agents turned to the frame of the pitch, so the
# eight directions, which come first in the sticky actions, turn by half a turn.
_DIRECTIONS = _DEFAULT[1:9]
_TURNED = {name: _DIRECTIONS[(index + 4) % 8] for index, name in enumerate(_DIRECTIONS)}
_FLOATS = ('ball', 'ball_direction', 'ball_rotation')
_FLOAT_TEAM = ('team', 'team_direction', 'team_tired_factor')
_INT_TEAM = ('team_yellow_card', 'team_active', 'team_roles')
_NUMBERS = ('ball_owned_team', 'ball_owned_player', 'game_mode', 'steps_left', 'score')


def _agents(players: Any) -> tuple[int, int]:
    if not isinstance(players, list) or not players:
        raise RecordingError(f'expected the players of agents, found {players!r}')
    left = right = 0
    for definition in players:
        name, _, text = str(definition).partition(':')
        counts = {'left_players': 0, 'right_players': 0}
        for part in filter(None, text.split(',')):
            key, _, value = part.partition('=')
            if key not in counts or not value.isdigit():
                raise RecordingError(f'expected the players of agents, found {definition!r}')
            counts[key] = int(value)
        if name != 'agent':
            raise RecordingError(f'expected only agents among the players, found {definition!r}')
        if not counts['left_players'] and not counts['right_players']:
            counts['left_players'] = 1
        left += counts['left_players']
        right += counts['right_players']
    return left, right


def _scenario(level: str) -> tuple[str, list[tuple[float | None, float | None]] | None]:
    # cuju's `DumpWriter` names a scenario that plays with difficulties of its own as
    # `<scenario>_difficulty_<left>_<right>`, each `none` or a number such as `0p5`.
    base, marker, rest = level.rpartition('_difficulty_')
    parts = rest.split('_')
    if not marker or len(parts) != 2:
        return level, None

    def number(text: str) -> float | None:
        return None if text == 'none' else float(text.replace('p', '.').replace('m', '-'))

    try:
        return base, [(number(parts[0]), number(parts[1]))]
    except ValueError:
        return level, None


def _turned_sticky(states: np.ndarray) -> np.ndarray:
    values = np.asarray(states).copy()
    values[:8] = np.asarray(states)[[(index + 4) % 8 for index in range(8)]]
    return values


def _bits(value: Any) -> bytes:
    return np.ascontiguousarray(np.asarray(value, np.float64)).tobytes()


def _same(stored: Any, replayed: Any, floating: bool) -> bool:
    if floating:
        return _bits(stored) == _bits(replayed)
    return np.array_equal(np.asarray(stored, np.int64), np.asarray(replayed, np.int64))


def _difference(
    observation: Mapping[str, Any], row: Mapping[str, np.ndarray], left: int, right: int
) -> str | None:
    pairs = [(name, True) for name in _FLOATS]
    for side in ('left', 'right'):
        pairs += [(f'{side}_{rest}', True) for rest in _FLOAT_TEAM]
        pairs += [(f'{side}_{rest}', False) for rest in _INT_TEAM]
    pairs += [(name, False) for name in _NUMBERS]
    for name, floating in pairs:
        if name in observation and name in row:
            if not _same(observation[name], row[name], floating):
                return f'{name}, the dump holds {np.asarray(observation[name]).tolist()}'
    sticky, active = row['sticky_actions'], row['active']
    for side, agents, first in (('left', left, 0), ('right', right, left)):
        stored = observation.get(f'{side}_agent_sticky_actions')
        if stored is not None:
            for index in range(agents):
                mine = (
                    sticky[first + index]
                    if side == 'left'
                    else _turned_sticky(sticky[first + index])
                )
                if len(stored) <= index or not _same(stored[index], mine, False):
                    return f'{side}_agent_sticky_actions'
        controlled = observation.get(f'{side}_agent_controlled_player')
        if controlled is not None and not _same(controlled, active[first : first + agents], False):
            return f'{side}_agent_controlled_player'
        designated = observation.get(f'{side}_team_designated_player')
        if designated is not None and agents and int(designated) != int(row['designated'][first]):
            return f'{side}_team_designated_player'
    return None


def from_grf_dump(source: Any, device: Device = 'cpu') -> 'Recording':
    """Converts a GRF dump by playing it again in the engine, and refuses one that differs."""
    path = os.fspath(source) if isinstance(source, (str, os.PathLike)) else None
    name = None if path is None else os.path.basename(path)
    try:
        entries = _steps(path, source)
    except ValueError as error:
        raise RecordingError(f'expected a GRF dump, found {error}') from error
    try:
        match = _convert(entries, device)
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise RecordingError(f'expected a GRF dump, found a step without {error}') from error
    if name:
        match.provenance['file'] = name
    return match


def _steps(path: str | None, source: Any) -> list[dict[str, Any]]:
    # The reader of the engine takes a path, so bytes and open files go through a file.
    if path is not None:
        return dumps._read_cli_dump(path)
    with tempfile.TemporaryDirectory() as folder:
        copy = os.path.join(folder, 'input.dump')
        with open(copy, 'wb') as handle:
            handle.write(_files.read_bytes(source))
        return dumps._read_cli_dump(copy)


def _convert(entries: list[dict[str, Any]], device: Any) -> 'Recording':
    import torch

    from .. import _state
    from .._batch import Batch
    from ..errors import ConfigurationError, MissingFileError

    config = entries[0]['debug']['config']
    if entries[0]['debug'].get('frame_cnt', 0) != 0:
        raise RecordingError('expected a dump that starts at the kick off, found one that does not')
    left, right = _agents(config.get('players'))
    action_set = config.get('action_set', 'default')
    if action_set not in ACTION_NAMES:
        raise RecordingError(f'expected an action set of GRF, found {action_set!r}')
    if int(config.get('physics_steps_per_frame', 10)) != 10:
        raise RecordingError('expected 10 physics steps for each frame, the engine default')
    names = ACTION_NAMES[action_set]
    actions = []
    for index, entry in enumerate(entries):
        row = (entry.get('debug') or {}).get('action')
        if not isinstance(row, list) or len(row) != left + right:
            raise RecordingError(f'expected the actions of {left + right} agents at step {index}')
        turned = [_TURNED.get(n, n) if agent >= left else n for agent, n in enumerate(row)]
        unknown = [n for n in turned if n not in names]
        if unknown:
            raise RecordingError(f'expected actions of the set {action_set}, found {unknown[0]!r}')
        actions.append([names.index(n) for n in turned])
    scenario, difficulty = _scenario(str(config['level']))
    try:
        batch = Batch(
            scenario,
            environment_count=1,
            representation='raw',
            action_set=action_set,
            auto_reset=False,
            device=device,
            threads=1,
            physics_steps_per_frame=10,
            game_engine_random_seed=int(config['game_engine_random_seed']),
            agent_counts=[(left, right)],
            difficulty=difficulty,
            # GRF: the first reset of an environment plays the episode after its number, and the
            # parity of that number picks the form of a scenario.
            episode_number=(int(config.get('episode_number', 2)) - 2) % 2,
        )
    except (ConfigurationError, MissingFileError) as error:
        raise RecordingError(f'the engine refused the dump, {error}') from error
    try:
        reverse = config.get('reverse_team_processing')
        batch.reset(reverse=None if reverse is None else int(bool(reverse)))
        state = batch.get_state(0)
        start, duration = _state.read(
            state, ['environment.step', 'environment.scenario.gameDuration']
        )
        players = (int(batch.left_players[0]), int(batch.right_players[0]))

        def view() -> dict[str, np.ndarray]:
            raw = {n: tensor[0].cpu().numpy().copy() for n, tensor in batch.raw().items()}
            return _views.left_view(raw, left, *players)

        rows = [view()]
        rewards = []
        ended = False
        for index, (entry, row) in enumerate(zip(entries, actions, strict=True)):
            if ended:
                raise RecordingError(
                    f'the dump differs from its replay, which ends the episode at step {index}'
                )
            _, reward, done, information = batch.step(torch.tensor([row], dtype=torch.int32))
            ended = bool(done[0])
            rewards.append(reward[0].cpu().numpy().copy())
            rows.append(view())
            found = _difference(entry.get('observation') or {}, rows[-1], left, right)
            if (
                found is None
                and 'reward' in entry
                and int(entry['reward']) != int(information['score_reward'][0])
            ):
                found = 'reward'
            if found is not None:
                raise RecordingError(f'the dump differs from its replay at step {index} in {found}')
        steps = len(entries)
        end = 0 if not ended else 2 if int(duration) - int(start) - steps <= 0 else 1
        return Recording(
            batch.scenario_name,
            left,
            np.asarray(actions),
            np.stack(rewards),
            right_agents=right,
            action_set=action_set,
            difficulty=batch.difficulty[0].tolist(),
            players=players,
            end=end,
            state=state,
            step={n: np.stack([r[n] for r in rows]) for n in rows[0]},
            provenance={'source': 'grf-dump'},
            engine=engine_version(),
        )
    finally:
        batch.close()
