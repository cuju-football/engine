# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""The standard controllers."""

import contextlib
from typing import Any

import numpy as np
import torch

from ..errors import ConfigurationError
from ._contract import ACTION_COUNTS, Actions, Controller, Observation, Spec, as_actions
from .client import Kaggle as Kaggle
from .client import resolve

__all__ = [
    'Bot',
    'Builtin',
    'Function',
    'Grf',
    'Kaggle',
    'Lazy',
    'PerEnvironment',
    'Pool',
    'Random',
]


def _callable(value: Any) -> Any:
    if isinstance(value, str):
        try:
            value = resolve(value, {})
        except ValueError as error:
            raise ConfigurationError(str(error)) from None
    if not callable(value):
        raise ConfigurationError(f'expected a callable, found {value!r}')
    return value


def _array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class Function(Controller):
    """Wraps a callable of an observation, batched and in the process, such as a network."""

    def __init__(
        self,
        function: Any,
        representation: str = 'raw',
        inference: bool = True,
        control: dict[str, Any] | None = None,
    ) -> None:
        """Wraps `function`, a callable or the text `module:object`, with the representation."""
        self.function = _callable(function)
        self.representation = representation
        self.inference = inference
        self.control = control

    def act(self, observation: Observation) -> Actions:
        """Calls the function, under `torch.inference_mode` unless `inference` is false."""
        guard = torch.inference_mode() if self.inference else contextlib.nullcontext()
        with guard:
            return as_actions(self.function(observation))


def _one(observation: Observation, row: int) -> Observation:
    def pick(value: Any) -> Any:
        if isinstance(value, dict):
            return {name: pick(item) for name, item in value.items()}
        return None if value is None else value[row]

    environments = observation.environments
    return Observation(
        data=pick(observation.data),
        first=bool(observation.first[row]),
        active=pick(observation.active),
        left_players=int(observation.left_players[row]),
        right_players=int(observation.right_players[row]),
        previous=pick(observation.previous),
        step=int(observation.step[row]),
        side=int(observation.side[row]),
        meta=observation.meta,
        environments=None if environments is None else int(environments[row]),
    )


class PerEnvironment(Controller):
    """Adapts a callable of one environment's observation, with arrays `[agents, ...]`."""

    def __init__(
        self,
        function: Any,
        representation: str = 'raw',
        fresh: bool = False,
        control: dict[str, Any] | None = None,
    ) -> None:
        """Wraps `function`, or with `fresh` a factory that makes one callable per episode."""
        self.function = _callable(function)
        self.representation = representation
        self.fresh = fresh
        self.control = control
        self._made: dict[int, Any] = {}

    def start(self, spec: Spec) -> None:
        """Forgets the callables of every environment."""
        self._made = {}

    def act(self, observation: Observation) -> Actions:
        """Calls the function once for each environment that has an active agent."""
        count = observation.count
        active = _array(observation.active)
        first = _array(observation.first)
        names = None if observation.environments is None else _array(observation.environments)
        out = np.zeros(active.shape, dtype=np.int64)
        for row in range(count):
            if not active[row].any():
                continue
            key = row if names is None else int(names[row])
            function = self.function
            if self.fresh:
                if first[row] or key not in self._made:
                    self._made[key] = self.function()
                function = self._made[key]
            answer = as_actions(function(_one(observation, row)))
            out[row] = _array(answer.actions).reshape(out.shape[1])
        return Actions(out)


class Random(Controller):
    """Draws uniform actions of the controller's action set, from its own seed."""

    def __init__(self, control: dict[str, Any] | None = None) -> None:
        """Takes the action set from the control."""
        self.control = control

    def start(self, spec: Spec) -> None:
        """Seeds the draws."""
        self._count = ACTION_COUNTS[spec.action_set]
        self._rng = np.random.default_rng(spec.seed)

    def act(self, observation: Observation) -> Actions:
        """Returns one action for each agent of each environment."""
        shape = tuple(_array(observation.active).shape)
        return Actions(self._rng.integers(0, self._count, size=shape))


class Builtin(Controller):
    """Hands a side to the built-in AI, which a match does by giving the side no agents."""

    def __init__(self, difficulty: float | None = None) -> None:
        """Holds the difficulty from 0 to 1, or `None` for the scenario's own."""
        if difficulty is not None and not 0 <= difficulty <= 1:
            raise ConfigurationError(f'expected a difficulty from 0 to 1, found {difficulty!r}')
        self.difficulty = difficulty

    def act(self, observation: Observation) -> Actions:
        """Never called, since a side without agents has nothing to answer."""
        raise ConfigurationError('expected builtin at the top of a side, found it called')


def _bot_core(own: Any, other: Any, device: Any, count: int) -> Any:
    from ._scripted import _IDLE, Bot as Core

    bot = Core.__new__(Core)
    bot._own, bot._other = own, other
    bot._last_action = torch.full((count,), _IDLE, device=device)
    bot._pressure = torch.zeros(count, dtype=torch.bool, device=device)
    return bot


class Bot(Controller):
    """GRF's `bot`, one scripted player for each agent, which needs the full action set."""

    def __init__(self) -> None:
        """Builds nothing until the session starts."""
        self.control = {'action_set': 'full'}
        self._bots: list[Any] = []

    def start(self, spec: Spec) -> None:
        """Checks the action set and forgets the state of every environment."""
        if spec.action_set != 'full':
            raise ConfigurationError(
                "expected the control action_set 'full', which GRF's bot needs, "
                f'found {spec.action_set!r}'
            )
        self._bots = []
        self._environments = spec.environments

    def act(self, observation: Observation) -> Actions:
        """Returns GRF's bot actions for each agent of each environment."""
        from ._scripted import _IDLE

        raw = observation.data
        device = raw['ball'].device
        first = torch.as_tensor(_array(observation.first), device=device)
        own = torch.as_tensor(_array(observation.left_players), device=device).long()
        other = torch.as_tensor(_array(observation.right_players), device=device).long()
        ids = (
            None
            if observation.environments is None
            else torch.as_tensor(_array(observation.environments), device=device).long()
        )
        agents = raw['ball'].shape[1]
        while len(self._bots) < agents:
            self._bots.append(_bot_core(own, other, device, self._environments))
        columns = []
        for agent in range(agents):
            bot = self._bots[agent]
            last, pressure = bot._last_action, bot._pressure
            subset_last = last if ids is None else last[ids]
            subset_pressure = pressure if ids is None else pressure[ids]
            subset_last = torch.where(first, torch.full_like(subset_last, _IDLE), subset_last)
            subset_pressure = subset_pressure & ~first
            bot._own, bot._other = own, other
            bot._last_action, bot._pressure = subset_last, subset_pressure
            bot.agent = agent
            action, state = bot._decide(raw)
            if ids is None:
                bot._last_action, bot._pressure = action, state
            else:
                last[ids], pressure[ids] = action, state
                bot._last_action, bot._pressure = last, pressure
            columns.append(action.to(torch.int32))
        return Actions(torch.stack(columns, 1))


class Lazy(Controller):
    """GRF's `lazy`, which stands still with `idle`."""

    def __init__(self, control: dict[str, Any] | None = None) -> None:
        """Takes the control, whose action set may be any."""
        self.control = control

    def act(self, observation: Observation) -> Actions:
        """Returns `idle` for every agent."""
        return Actions(np.zeros(tuple(_array(observation.active).shape), dtype=np.int64))


class Grf(Controller):
    """A player of GRF's definition string, such as `ppo2_cnn:right_players=1,checkpoint=...`."""

    mirrored = False

    def __init__(self, definition: str) -> None:
        """Takes the player's name and parameters from the string, as GRF does."""
        from .._grf_config import parse_player_definition

        self.definition = definition
        self.name, self.parameters = parse_player_definition(definition)
        if self.name == 'agent':
            raise ConfigurationError(
                f"expected a player other than 'agent' in {definition!r}, which a controller "
                'replaces'
            )
        self._inner: Controller | None = None
        self._players: list[Any] = []
        if self.name == 'bot':
            self._inner = Bot()
            self.control = self._inner.control
        elif self.name == 'lazy':
            self._inner = Lazy()

    def start(self, spec: Spec) -> None:
        """Builds one player for each environment, as GRF builds its players once."""
        if self._inner is not None:
            self._inner.start(spec)
            return
        import copy

        from .._grf_config import _without_draw
        from ._grf_players import PythonPlayer, player_module

        module = player_module(self.name)
        if module is None:
            raise ConfigurationError(f'expected a module for the player {self.name!r}, found none')
        player_class = getattr(module, 'Player', None)
        if player_class is None:
            raise ConfigurationError(f'expected a class Player in the module {module.__name__}')
        config = _without_draw(
            {'level': spec.scenario, 'action_set': spec.action_set, 'players': [self.definition]}
        )
        shared = {'index': 0, 'player_' + self.name: 0}
        self._players = []
        for _ in range(spec.environments):
            player_config = copy.deepcopy(shared)
            player_config.update(self.parameters)
            self._players.append(player_class(player_config, config))
        self._flip = spec.side == 1 and bool(self._players[0].can_play_right())
        self._helper = PythonPlayer(None, [], 0, spec.action_set)
        self._sticky = self._helper._sticky

    def act(self, observation: Observation) -> Actions:
        """Calls each environment's player with GRF's observations, and returns its actions."""
        if self._inner is not None:
            return self._inner.act(observation)
        from .._raw import grf_observation
        from ._grf_players import _match_view

        raw = {name: torch.as_tensor(_array(value)) for name, value in observation.data.items()}
        active = _array(observation.active)
        first = _array(observation.first)
        names = None if observation.environments is None else _array(observation.environments)
        out = np.zeros(active.shape, dtype=np.int64)
        agents = active.shape[1]
        for row in range(observation.count):
            key = row if names is None else int(names[row])
            player = self._players[key]
            if first[row]:
                player.reset()
            if not active[row].any():
                continue
            own, other = int(observation.left_players[row]), int(observation.right_players[row])
            views = [grf_observation(raw, row, agent, own, other) for agent in range(agents)]
            if self._flip:
                views = [_match_view(view, self._sticky) for view in views]
            answer = player.take_action(views)
            if isinstance(answer, np.ndarray):
                answer = answer.tolist()
            if not isinstance(answer, list):
                answer = [answer]
            if len(answer) != agents:
                raise ValueError(f'Player provided {len(answer)} actions instead of {agents}.')
            out[row] = [self._helper._index(value, self._flip) for value in answer]
        return Actions(out)

    def close(self) -> None:
        """Drops the players."""
        self._players = []


class Pool(Controller):
    """Draws one inner controller for each environment at each `first`, from its own seed."""

    swappable = False

    def __init__(
        self,
        inner: list[Any],
        weights: list[float] | None = None,
        load: Any = None,
        control: dict[str, Any] | None = None,
    ) -> None:
        """Builds the inner controllers, which may be nodes, entries or controllers."""
        self._load = load
        self.control = control
        self._members: list[Any] = []
        self.owned: set[int] = set()
        self._groups: list[tuple[list[int], np.ndarray]] = []
        self._latest = 0
        self._add_group(inner, weights)
        self.representation = self._members[0].representation
        if self.control is None:
            self.control = getattr(self._members[0], 'control', None)
        self._spec: Spec | None = None
        self._assignment = np.zeros(0, dtype=np.int64)
        self._group = np.zeros(0, dtype=np.int64)

    def _build(self, item: Any, index: int) -> tuple[Any, bool]:
        if isinstance(item, Controller) or (hasattr(item, 'act') and not isinstance(item, type)):
            return item, False
        if isinstance(item, dict) and self._load is not None:
            return self._load(item, f'inner[{index}]'), True
        from ._load import load

        return load(item), True

    def _add_group(
        self, inner: list[Any], weights: list[float] | None, owned: list[bool] | None = None
    ) -> int:
        if not isinstance(inner, (list, tuple)) or not inner:
            raise ConfigurationError(f'expected a list of inner controllers, found {inner!r}')
        weights = [1.0] * len(inner) if weights is None else [float(w) for w in weights]
        if len(weights) != len(inner) or min(weights) < 0 or sum(weights) <= 0:
            raise ConfigurationError(
                f'expected one weight for each of the {len(inner)} inner controllers, found '
                f'{weights!r}'
            )
        members = []
        for index, item in enumerate(inner):
            controller, made = self._build(item, index)
            if owned is not None:
                made = owned[index]
            wanted = getattr(controller, 'representation', 'raw')
            if self._members and wanted != self.representation_of_first():
                raise ConfigurationError(
                    'expected inner controllers of one representation, found '
                    f'{self.representation_of_first()!r} and {wanted!r}'
                )
            self._members.append(controller)
            if made:
                self.owned.add(len(self._members) - 1)
            members.append(len(self._members) - 1)
        self._groups.append((members, np.asarray(weights) / sum(weights)))
        self._latest = len(self._groups) - 1
        return self._latest

    def representation_of_first(self) -> str:
        """Returns the representation that every inner controller shares."""
        for member in self._members:
            if member is not None:
                return getattr(member, 'representation', 'raw')
        return 'raw'

    def _reset_state(self, spec: Spec) -> None:
        self._spec = spec
        self._assignment = np.full(spec.environments, -1, dtype=np.int64)
        self._group = np.full(spec.environments, self._latest, dtype=np.int64)
        self._rng = np.random.default_rng(spec.seed)

    def start(self, spec: Spec) -> None:
        """Starts every inner controller and clears the draws."""
        self._reset_state(spec)
        for member in self._members:
            if member is not None:
                member.start(spec)

    def adopt(self, spec: Spec) -> None:
        """Takes the place of a controller that has started, which is the first member."""
        self._reset_state(spec)
        self._group[:] = 0
        self._assignment[:] = 0

    def set_pool(
        self,
        inner: list[Any],
        weights: list[float] | None = None,
        mask: Any = None,
        owned: list[bool] | None = None,
    ) -> None:
        """Swaps the inner controllers of the environments of `mask`, at their next reset."""
        start = len(self._members)
        group = self._add_group(inner, weights, owned)
        if self._spec is None:
            return
        for member in self._members[start:]:
            member.start(self._spec)
        selected = (
            np.ones(self._spec.environments, dtype=bool)
            if mask is None
            else _array(mask).astype(bool)
        )
        self._group[selected] = group

    def act(self, observation: Observation) -> Actions:
        """Calls each inner controller once, on the environments it holds."""
        count = observation.count
        ids = (
            np.arange(count)
            if observation.environments is None
            else _array(observation.environments)
        )
        first = _array(observation.first)
        for row in np.nonzero(first | (self._assignment[ids] < 0))[0]:
            members, weights = self._groups[self._group[ids[row]]]
            self._assignment[ids[row]] = members[int(self._rng.choice(len(members), p=weights))]
        chosen = self._assignment[ids]
        shape = tuple(_array(observation.active).shape)
        out = np.zeros(shape, dtype=np.int64)
        extras: dict[str, np.ndarray] = {}
        for index in np.unique(chosen):
            rows = np.nonzero(chosen == index)[0]
            sub = observation if len(rows) == count else observation.select(rows)
            answer = as_actions(self._members[int(index)].act(sub))
            out[rows] = _array(answer.actions).reshape(len(rows), shape[1])
            for name, value in answer.extras.items():
                value = _array(value)
                extras.setdefault(name, np.zeros((count, *value.shape[1:]), dtype=value.dtype))[
                    rows
                ] = value
        self._release()
        return Actions(out, extras)

    def _release(self) -> None:
        live = set(self._assignment[self._assignment >= 0].tolist())
        for group in set(self._group.tolist()):
            live.update(self._groups[group][0])
        for index, member in enumerate(self._members):
            if member is not None and index not in live:
                if index in self.owned:
                    member.close()
                self._members[index] = None

    def close(self) -> None:
        """Closes the inner controllers that the pool built."""
        for index, member in enumerate(self._members):
            if member is not None and index in self.owned:
                member.close()


NAMES = {
    'function': Function,
    'per_environment': PerEnvironment,
    'kaggle': Kaggle,
    'grf': Grf,
    'bot': Bot,
    'lazy': Lazy,
    'random': Random,
    'pool': Pool,
    'builtin': Builtin,
}
