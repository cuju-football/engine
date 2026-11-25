# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""Kaggle's GRF competition protocol, for a batch and for Kaggle agents."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import torch

from ._batch import Batch
from .raw import grf_observation, team_sizes

__all__ = [
    'AgentRunner',
    'KaggleAgent',
    'accepts_configuration',
    'action_index',
    'action_indices',
    'evaluate',
    'observation',
    'players_raw',
]

_OVERAGE_TIME = 60


class _Struct(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def _lists(entry):
    return {
        name: value.tolist() if isinstance(value, np.ndarray) else value
        for name, value in entry.items()
    }


def players_raw(
    batch: Any, environment: int, agent: int, final: bool = False
) -> list[dict[str, Any]]:
    """Returns `obs['players_raw']` of a Kaggle agent that plays an agent of a batch."""
    raw = batch.final_raw() if final else batch.raw()
    return [
        _lists(grf_observation(raw, environment, agent, *team_sizes(batch, environment, agent)))
    ]


def observation(
    batch: Any,
    environment: int,
    agent: int | list[int],
    step: int = 0,
    final: bool = False,
    raw: dict[str, Any] | None = None,
) -> _Struct:
    """Returns a Kaggle agent's observation, a dictionary whose keys are also attributes."""
    several = isinstance(agent, (list, tuple))
    agents = [int(index) for index in agent] if several else [int(agent)]
    if raw is None:
        raw = batch.final_raw() if final else batch.raw()
    return _Struct(
        players_raw=[
            _lists(grf_observation(raw, environment, index, *team_sizes(batch, environment, index)))
            for index in agents
        ],
        controlled_players=len(agents),
        step=step,
        remainingOverageTime=_OVERAGE_TIME,
    )


def accepts_configuration(function: Any) -> bool:
    """Returns whether an agent function takes a configuration, as Kaggle decides it."""
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return False
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 2 or any(
        parameter.kind == parameter.VAR_POSITIONAL for parameter in parameters
    )


def action_index(value: Any, count: int) -> int:
    """Returns the action index of a Kaggle agent's answer for one controlled player."""
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f'expected one action for the one controlled player, found {value!r}')
        value = value[0]
    return _index(value, count)


def action_indices(value: Any, count: int, players: int) -> list[int]:
    """Returns the action indices of an answer for `players` controlled players, one each."""
    if isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    if not isinstance(value, (list, tuple)) or len(value) != players:
        raise ValueError(
            f'expected one action for each of the {players} controlled players, found {value!r}'
        )
    return [_index(item, count) for item in value]


def _index(value, count):
    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.item()
    if not isinstance(value, (int, np.integer)) and isinstance(getattr(value, 'value', None), int):
        value = value.value
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f'expected an integer action from the agent, found {value!r}')
    # GRF: an action indexes a Python list, so it may count from the end.
    if not -count <= int(value) < count:
        raise IndexError(f'expected an action from {-count} to {count - 1}, found {value}')
    return int(value)


_accepts_configuration = accepts_configuration
_action = action_index
_actions = action_indices


class AgentRunner:
    """Calls a Kaggle agent function as Kaggle does, and checks its answer against an action set."""

    def __init__(self, function: Any, configuration: dict[str, Any] | None = None) -> None:
        """Binds a Kaggle agent function."""
        self.function = function
        self.configuration = _Struct(configuration or {})
        self.accepts_configuration = accepts_configuration(function)

    def __call__(self, obs: Any) -> Any:
        """Calls the function with an observation, and the configuration if it takes one."""
        if self.accepts_configuration:
            return self.function(obs, self.configuration)
        return self.function(obs)

    def act(self, obs: Any, count: int) -> list[int]:
        """Calls the function and returns one action index per controlled player."""
        players = len(obs['players_raw'])
        answer = self(obs)
        if players == 1:
            return [action_index(answer, count)]
        return action_indices(answer, count, players)


class KaggleAgent:
    """Runs a Kaggle agent function as one agent, or several, of every environment of a batch."""

    def __init__(
        self,
        function: Any,
        batch: Any,
        agent: int,
        configuration: dict[str, Any] | None = None,
        agents: list[int] | None = None,
    ) -> None:
        """Binds a Kaggle agent function to an agent of a batch."""
        if getattr(batch, '_raw_fields', None) is None:
            raise ValueError('expected a batch with the raw representation or with_raw=True')
        self.agents = [agent] if agents is None else [int(index) for index in agents]
        if not self.agents or len(set(self.agents)) != len(self.agents):
            raise ValueError(f'expected distinct agents, found {agents!r}')
        for index in self.agents:
            if not 0 <= index < batch.agents or not bool(batch.agent_mask[:, index].all()):
                raise ValueError(f'expected an agent of every environment, found {index}')
        self._several = agents is not None
        self.function = function
        self.batch = batch
        self.agent = self.agents[0]
        self.configuration = _Struct(configuration or {})
        self._two_arguments = accepts_configuration(function)
        self._steps = [0] * batch.environment_count

    def reset(self, mask: Any = None) -> None:
        """Counts Kaggle's `step` from 0 again, in the environments of `mask` or in all."""
        for environment in range(self.batch.environment_count):
            if mask is None or bool(mask[environment]):
                self._steps[environment] = 0

    def act(self, done: Any = None, skip: Any = None) -> torch.Tensor:
        """Returns the agent's actions, as `[environment_count]` `int32` indices."""
        if done is not None:
            self.reset(torch.as_tensor(done).bool().cpu())
        skip = None if skip is None else torch.as_tensor(skip).bool().cpu()
        raw = {name: tensor.cpu() for name, tensor in self.batch.raw().items()}
        actions = []
        players = len(self.agents)
        for environment in range(self.batch.environment_count):
            if skip is not None and bool(skip[environment]):
                actions.append([0] * players if self._several else 0)
                continue
            entries = [
                _lists(
                    grf_observation(
                        raw, environment, agent, *team_sizes(self.batch, environment, agent)
                    )
                )
                for agent in self.agents
            ]
            obs = _Struct(
                players_raw=entries,
                controlled_players=players,
                step=self._steps[environment],
                remainingOverageTime=_OVERAGE_TIME,
            )
            answer = (
                self.function(obs, self.configuration)
                if self._two_arguments
                else self.function(obs)
            )
            if self._several:
                actions.append(action_indices(answer, self.batch.action_count, players))
            else:
                actions.append(action_index(answer, self.batch.action_count))
            self._steps[environment] += 1
        actions = torch.tensor(actions, dtype=torch.int32, device=self.batch._torch_device)
        # GRF: a negative index counts from the end, which the batch plays as the same action.
        return torch.where(actions < 0, actions + self.batch.action_count, actions)


def evaluate(
    left: Any,
    right: Any = None,
    episodes: int = 1,
    scenario: str = '11_vs_11_kaggle',
    seed: int = 0,
    device: str = 'cpu',
    threads: int = 0,
    configuration: dict[str, Any] | None = None,
) -> list[tuple[int, int]]:
    """Plays matches of Kaggle agents, one per environment, and returns their final scores."""
    batch = Batch(
        scenario,
        environment_count=episodes,
        representation='raw',
        seed=seed,
        auto_reset=False,
        device=device,
        threads=threads,
        agent_counts=[(1, 0 if right is None else 1)],
    )
    try:
        players = [KaggleAgent(left, batch, 0, configuration)]
        if right is not None:
            players.append(KaggleAgent(right, batch, 1, configuration))
        batch.reset()
        finished = torch.zeros(episodes, dtype=torch.bool)
        while not bool(finished.all()):
            actions = torch.stack([player.act(skip=finished) for player in players], 1)
            _, _, done, _ = batch.step(actions)
            finished |= done.bool().cpu()
        scores = batch.raw()['score'][:, 0].cpu().tolist()
    finally:
        batch.close()
    return [(int(own), int(other)) for own, other in scores]
