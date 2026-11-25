# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import importlib
import re
from types import ModuleType
from typing import Any

import numpy as np
import torch

from . import dumps
from ._batch import unwrap
from ._raw import grf_observations

_PACKAGE = 'cuju.gfootball.env.players'
_MODULE_PATH = re.compile(r'\w+(\.\w+)*')


def _import(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name is not None and (name == error.name or name.startswith(error.name + '.')):
            return None
        raise


def player_module(name: str) -> ModuleType | None:
    if not _MODULE_PATH.fullmatch(name):
        return None
    candidates = ('cuju._replay_player',) if name == 'replay' else (f'{_PACKAGE}.{name}', name)
    for candidate in candidates:
        module = _import(candidate)
        if module is not None:
            return module
    return None


def _match_view(observation, sticky_names):
    view = {}
    for key in ('ball', 'ball_direction'):
        point = observation[key]
        view[key] = np.array([-point[0], -point[1], point[2]])
    view['ball_rotation'] = observation['ball_rotation']
    owned = observation['ball_owned_team']
    view['ball_owned_team'] = 1 - owned if owned > -1 else -1
    view['ball_owned_player'] = observation['ball_owned_player']
    view['score'] = [observation['score'][1], observation['score'][0]]
    view['game_mode'] = observation['game_mode']
    view['steps_left'] = observation['steps_left']
    for source, target in (('left', 'right'), ('right', 'left')):
        view[f'{target}_team'] = -observation[f'{source}_team']
        view[f'{target}_team_direction'] = -observation[f'{source}_team_direction']
        for suffix in ('tired_factor', 'active', 'yellow_card', 'roles'):
            view[f'{target}_team_{suffix}'] = observation[f'{source}_team_{suffix}']
    view['designated'] = observation['designated']
    view['active'] = observation['active']
    view['sticky_actions'] = np.array(
        dumps._flip_sticky(list(observation['sticky_actions']), sticky_names), dtype=np.uint8
    )
    return view


class PythonPlayer:
    def __init__(self, player: Any, slots: list[int], left: int, action_set: str) -> None:
        self.player = player
        self._slots = slots
        self._left = left
        self._names = dumps._ACTION_SETS[action_set]
        self._sticky = [name for name in self._names if name in dumps._STICKY]
        self._batch = None

    def bind(self, batch: Any) -> PythonPlayer:
        self._batch = unwrap(batch)
        return self

    def _index(self, action, turned):
        names = self._names
        if hasattr(action, '_name') and hasattr(action, '_backend_action'):
            name = action._name
            if name not in names:
                raise ValueError(
                    f'expected an action of the set of the environment, found {name!r}, so create '
                    'the environment with a set that holds it'
                )
        elif isinstance(action, (bool, int, np.int32, np.int64)):
            index = int(action)
            if not index < len(names):
                raise ValueError(f'expected an action inside the action set, found {index}')
            name = names[index]
        else:
            raise TypeError(f'expected an action index or a CoreAction, found {action!r}')
        if turned:
            name = dumps._FLIPPED.get(name, name)
        return names.index(name)

    def act(self) -> torch.Tensor:
        observations = grf_observations(self._batch, 0, self._slots) if self._slots else []
        # GRF: a player that plays the right side as it is sees the match as it is.
        right_as_is = bool(self.player.can_play_right())
        if right_as_is:
            for index in range(self._left, len(observations)):
                observations[index] = _match_view(observations[index], self._sticky)
        actions = self.player.take_action(observations)
        # GRF: `FootballEnv._action_to_list`.
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()
        if not isinstance(actions, list):
            actions = [actions]
        if len(actions) != len(observations):
            raise ValueError(
                f'Player provided {len(actions)} actions instead of {len(observations)}.'
            )
        indices = [
            self._index(action, right_as_is and position >= self._left)
            for position, action in enumerate(actions)
        ]
        return torch.tensor(indices, dtype=torch.int64)

    def reset(self) -> None:
        self.player.reset()


class PythonPlayers:
    def __init__(
        self,
        definitions: list[str],
        parsed: list[tuple[Any, ...]],
        scripted: list[tuple[str, list[int]]],
        config: Any,
        action_set: str,
    ) -> None:
        from ._grf_config import parse_player_definition

        self.config = config
        others = {name for name, *_ in parsed if name not in ('agent', 'bot', 'lazy')}
        built = []
        shared: dict[str, Any] = {'index': 0}
        for definition, (name, _, left, _, _, _) in zip(definitions, parsed, strict=True):
            _, parameters = parse_player_definition(definition)
            key = f'player_{name}'
            shared[key] = shared[key] + 1 if key in shared else 0
            if name in ('keyboard', 'gamepad'):
                raise ValueError(
                    f'expected a player that acts on observations, found {definition!r}, one of '
                    "GRF's keyboard and gamepad players, which take a person's input in GRF's "
                    'rendered game'
                )
            if name in others:
                player_config = copy.deepcopy(shared)
                player_config.update(parameters)
                module = player_module(name)
                if module is None:
                    raise ValueError(
                        f'expected the players agent, bot, lazy, or a module that defines the '
                        f'player, in {definition!r}, and found no module {_PACKAGE}.{name} or '
                        f'{name}'
                    )
                player_class = getattr(module, 'Player', None)
                if player_class is None:
                    raise ValueError(f'expected a class Player in the module {module.__name__}')
                built.append((player_class(player_config, self.config), left))
            shared['index'] += 1
        players = iter(built)
        self.scripted = []
        self.players = []
        for name, slots in scripted:
            if name in others:
                player, left = next(players)
                name = PythonPlayer(player, slots, left, action_set)
                self.players.append(name)
            self.scripted.append((name, slots))

    def reset(self) -> None:
        for player in self.players:
            player.reset()
