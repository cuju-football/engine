# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime
import io
import os
import re
import tempfile
from typing import Any, NamedTuple

import torch

from . import dumps
from ._batch import Batch

# GRF: `observation_processor.PAST_STEPS_TRACE_SIZE`, the steps that a dump on demand holds.
_TRACE_SIZE = 100
# GRF: the dumps that the options of an environment write, which `write_dump` does not start.
_OPTION_DUMPS = frozenset({'episode_done', 'score', 'lost_score'})
_DIFFICULTY_LEVEL = re.compile(r'(.+)_difficulty_([0-9a-z]+)_([0-9a-z]+)')
# GRF: the values that its replay sets over `config_update`.
_REPLAY_VALUES = frozenset({'physics_steps_per_frame', 'real_time', 'tracesdir', 'write_video'})
_UNCOMPARED = frozenset({'time', 'config', 'frame'})


def _difficulty(text):
    return None if text == 'none' else float(text.replace('p', '.').replace('m', '-'))


def agent_counts(players: list[str]) -> tuple[int, int]:
    from ._grf_config import count_left_players, count_right_players

    return (
        sum(count_left_players(player) for player in players),
        sum(count_right_players(player) for player in players),
    )


def batch_for(
    config: dict[str, Any], level: str | None = None, device: Any = 'cpu', threads: int = 1
) -> Batch:
    level = config['level'] if level is None else level
    difficulty = None
    match = _DIFFICULTY_LEVEL.fullmatch(level)
    if match is not None and not os.path.isfile(level):
        try:
            difficulty = [(_difficulty(match.group(2)), _difficulty(match.group(3)))]
            level = match.group(1)
        except ValueError:
            difficulty = None
    left, right = agent_counts(config['players'])
    return Batch(
        level,
        environment_count=1,
        representation='raw',
        action_set=config['action_set'],
        auto_reset=False,
        device=device,
        threads=threads,
        physics_steps_per_frame=config['physics_steps_per_frame'],
        game_engine_random_seed=config['game_engine_random_seed'],
        agent_counts=[(left, right)],
        difficulty=difficulty,
        episode_number=(config['episode_number'] - 2) % 2,
    )


def start(batch: Batch, config: dict[str, Any]) -> None:
    reverse = config.get('reverse_team_processing')
    batch.reset(reverse=None if reverse is None else int(bool(reverse)))


class Player:
    def __init__(self, batch: Batch) -> None:
        self.batch = batch
        self.writer = dumps.DumpWriter(
            batch, tempfile.gettempdir(), environments=[0], write_full_episode_dumps=False
        )
        self.episode = None

    def begin(self, frame: int | None = None, cumulative: int = 0) -> None:
        values, _ = self.writer._episode_values(0)
        self.episode = dumps._Episode(0, values, None, False, False)
        self.episode.frame = values[5] + 1 if frame is None else frame
        self.episode.cumulative = cumulative

    def step(self, row: Any, entry: bool = True) -> tuple[dict[str, Any] | None, int, bool]:
        batch = self.batch
        actions = torch.as_tensor(row, dtype=torch.int32).reshape(1, batch.agents).cpu()
        _, _, done, information = batch.step(actions.to(batch.torch_device))
        if not entry:
            reward = int(information['score_reward'][0])
            self.episode.cumulative += reward
            self.episode.frame += 1
            return None, reward, bool(done[0])
        action_rows, done_rows, reward_rows, raw_rows = self.writer._rows(actions)
        values, _ = self.writer._episode_values(0)
        ended = bool(done_rows[0])
        raw = [
            {name: raw_rows[name][0][agent] for name in raw_rows} for agent in range(batch.agents)
        ]
        designated = values[15:17] if ended else values[13:15]
        reward = int(reward_rows[0])
        entry = self.writer._entry(self.episode, action_rows[0], reward, raw, designated)
        return entry, reward, ended


class EpisodeSteps:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> list[Any]:
        self.segments: list[tuple[bytes | None, list[Any]]] = [(None, [])]
        # GRF: the trace of the dumps starts again with each episode.
        self.first = 0
        self.names: set[str] = set()
        self.configs: dict[int, dict[str, Any]] = {}
        return self.segments[-1][1]

    def set_state(self, state: bytes, ended: bool) -> list[Any]:
        self.segments.append((state, []))
        if ended:
            # GRF: an ended episode has no trace, and loading a state starts a new one.
            self.first = len(self.segments) - 1
            self.names = set()
            self.configs = {}
        return self.segments[-1][1]

    def _play(self, environment, config):
        level = environment._settings['level']
        batch = batch_for(config, level, environment.batch.torch_device, 1)
        try:
            player = Player(batch)
            cumulative = 0
            trace_length = sum(len(rows) for _, rows in self.segments[self.first :])
            kept = trace_length - min(trace_length, _TRACE_SIZE)
            played, entries, goals = 0, [], []
            for index, (state, rows) in enumerate(self.segments):
                if state is None:
                    start(batch, config)
                    player.begin(0, cumulative)
                else:
                    batch.set_state(0, state)
                    player.begin(None, cumulative)
                traced = index >= self.first
                for row in rows:
                    wanted = traced and played >= kept
                    entry, reward, _ = player.step(row, wanted)
                    if wanted:
                        entries.append((played, entry))
                    if traced:
                        if reward in (1, -1):
                            goals.append(played)
                        played += 1
                cumulative = player.episode.cumulative
        finally:
            batch.close()
        return entries, goals

    def write_dump(self, environment: Any, name: str) -> str | None:
        if environment._done:
            raise RuntimeError(
                "the episode ended, and GRF's environment keeps no steps to dump until reset()"
            )
        # GRF: the options start these dumps, and a name dumps once an episode.
        if name in _OPTION_DUMPS or name in self.names:
            return None
        self.names.add(name)
        # GRF: the configuration of a dump is the environment's, with the episode's values.
        config = environment._config.get_dictionary()
        batch = environment.batch
        writer = dumps.DumpWriter(
            batch, os.curdir, environments=[0], write_full_episode_dumps=False
        )
        values, _ = writer._episode_values(0)
        played = dict(config)
        played.update(
            {
                'action_set': dumps._ACTION_SET_NAMES[batch.action_count],
                'game_engine_random_seed': values[0],
                'reverse_team_processing': bool(values[1]),
                'episode_number': values[2] + 1 + values[3],
                'physics_steps_per_frame': values[8],
            }
        )
        entries, goals = self._play(environment, played)
        configs = self.configs
        if entries:
            configs[entries[0][0]] = config
        # GRF: the first step of each dump of the options takes the configuration as well.
        own = environment._dumps
        current = own._writer._episodes.get(0) if own is not None else None
        if current is not None and len(self.segments) == 1:
            if current.full:
                configs.setdefault(0, config)
            if current.goals:
                for position in goals:
                    configs.setdefault(max(0, position - _TRACE_SIZE), config)
        for position, entry in entries:
            if position in configs:
                entry['debug']['config'] = configs[position]
        # GRF writes into `tracesdir`, the `logdir` of `create_environment`.
        directory = config['tracesdir'] or os.curdir
        os.makedirs(directory, exist_ok=True)
        stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S%f')
        path = f'{directory}{os.sep}{name}_{stamp}'
        dumps._write_file(path + '.dump', (dumps._pickle(entry) for _, entry in entries))
        return path


class ReplayResult(NamedTuple):
    steps: int
    ended: bool
    difference: str | None


def _bits(tensor):
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def _shown(value):
    if isinstance(value, torch.Tensor):
        if value.numel() <= 6:
            return f'{value.tolist()}'
        return f'a {value.dtype} tensor of the shape {tuple(value.shape)}'
    return repr(value)


def difference(replayed: Any, stored: Any, where: str) -> str | None:
    if isinstance(stored, torch.Tensor):
        if (
            not isinstance(replayed, torch.Tensor)
            or replayed.dtype != stored.dtype
            or replayed.shape != stored.shape
            or not torch.equal(_bits(replayed), _bits(stored))
        ):
            return f'{where}: the replay gives {_shown(replayed)}, the dump holds {_shown(stored)}'
        return None
    if isinstance(stored, dict):
        if not isinstance(replayed, dict):
            return f'{where}: the replay gives {_shown(replayed)}, the dump holds a dictionary'
        for key, value in stored.items():
            if key in _UNCOMPARED:
                continue
            if key not in replayed:
                return f'{where}.{key}: the replay lacks it'
            found = difference(replayed[key], value, f'{where}.{key}')
            if found is not None:
                return found
        return None
    if isinstance(stored, (list, tuple)):
        if type(replayed) is not type(stored) or len(replayed) != len(stored):
            return f'{where}: the replay gives {_shown(replayed)}, the dump holds {_shown(stored)}'
        for index, (first, second) in enumerate(zip(replayed, stored, strict=True)):
            found = difference(first, second, f'{where}[{index}]')
            if found is not None:
                return found
        return None
    if type(replayed) is not type(stored) or replayed != stored:
        return f'{where}: the replay gives {_shown(replayed)}, the dump holds {_shown(stored)}'
    return None


def loaded(entry: dict[str, Any]) -> dict[str, Any]:
    return dumps._converted(dumps._DumpUnpickler(io.BytesIO(dumps._pickle(entry))).load())


def replay(
    path: str,
    device: Any = 'cpu',
    threads: int = 1,
    overrides: dict[str, Any] | None = None,
) -> ReplayResult:
    return _replay_steps(dumps.read_dump(path), path, device, threads, overrides)


def _replay_steps(steps, path, device='cpu', threads=1, overrides=None) -> ReplayResult:
    if not steps or 'config' not in steps[0].get('debug', {}):
        raise ValueError(f'expected a dump whose first step holds the configuration in {path}')
    if steps[0]['debug']['frame_cnt'] != 0:
        # GRF: `ScriptHelpers.replay` asserts the same.
        raise ValueError(
            'the dump does not start from the beginning of its episode, so it cannot be replayed'
        )
    config = dict(steps[0]['debug']['config'])
    config.update(
        (key, value) for key, value in (overrides or {}).items() if key not in _REPLAY_VALUES
    )
    names = dumps._ACTION_SETS[config['action_set']]
    left, _ = agent_counts(config['players'])
    batch = batch_for(config, device=device, threads=threads)
    try:
        start(batch, config)
        player = Player(batch)
        player.begin(0)
        ended = False
        for index, step in enumerate(steps):
            if ended:
                return ReplayResult(
                    index,
                    True,
                    f'step {index}: the replay ended its episode at step {index - 1}, and the dump '
                    f'holds {len(steps)} steps',
                )
            row = []
            for agent, name in enumerate(step['debug']['action']):
                turned = dumps._FLIPPED.get(name, name) if agent >= left else name
                if turned not in names:
                    raise ValueError(
                        f'step {index} holds the action {name!r}, outside the action set '
                        f'{config["action_set"]!r}'
                    )
                row.append(names.index(turned))
            row += [0] * (batch.agents - len(row))
            entry, _, ended = player.step(row)
            found = difference(loaded(entry), step, f'step {index}')
            if found is not None:
                return ReplayResult(index, ended, found)
        return ReplayResult(len(steps), ended, None)
    finally:
        batch.close()
