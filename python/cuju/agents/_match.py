# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""Plays a batch against controllers: `Match`, its `Result`, `play` and `play_batch`."""

import dataclasses
import math
from typing import Any

import numpy as np
import torch

from .._batch import unwrap
from ..errors import ConfigurationError
from ..types import Device
from . import _control
from ._contract import (
    ACTION_COUNTS,
    SIDES,
    Controller,
    ControllerError,
    Spec,
    as_actions,
    normalize,
)
from ._load import load
from .client import Observation
from .standard import Builtin, Pool

__all__ = ['Episodes', 'Match', 'Result', 'play', 'play_batch']

_REPRESENTATIONS = ('raw', 'simple115', 'simple115v2', 'extracted')
_OVERAGE = 60


@dataclasses.dataclass
class Episodes:
    """The finished episodes of the caller's side, totals that a read brings from the device."""

    count: int
    total_return: float
    total_length: int

    @property
    def mean_return(self) -> float:
        """The mean episode return, or 0 before any episode ended."""
        return self.total_return / self.count if self.count else 0.0

    @property
    def mean_length(self) -> float:
        """The mean episode length in steps, or 0 before any episode ended."""
        return self.total_length / self.count if self.count else 0.0


@dataclasses.dataclass
class Result:
    """What `Match.run` returns, from the left side's point of view."""

    score: np.ndarray
    steps: np.ndarray
    error: BaseException | None = None
    recordings: list[Any] = dataclasses.field(default_factory=list)

    @property
    def episodes(self) -> int:
        """The finished episodes."""
        return len(self.steps)

    @property
    def wins(self) -> int:
        """The episodes that the left side won."""
        return int((self.score[:, 0] > self.score[:, 1]).sum())

    @property
    def draws(self) -> int:
        """The drawn episodes."""
        return int((self.score[:, 0] == self.score[:, 1]).sum())

    @property
    def losses(self) -> int:
        """The episodes that the left side lost."""
        return int((self.score[:, 0] < self.score[:, 1]).sum())

    @property
    def goal_difference(self) -> int:
        """The goals of the left side minus the goals of the right side, in all episodes."""
        return int((self.score[:, 0] - self.score[:, 1]).sum())

    @property
    def win_rate(self) -> float:
        """The share of episodes that the left side won."""
        return self.wins / self.episodes if self.episodes else 0.0

    @property
    def interval(self) -> tuple[float, float]:
        """The 95% Wilson confidence interval of the win rate."""
        n = self.episodes
        if not n:
            return 0.0, 1.0
        z, p = 1.959963984540054, self.wins / n
        center = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return max(0.0, center - half), min(1.0, center + half)

    def summary(self) -> dict[str, Any]:
        """Returns the counts, the goal difference and the win rate interval as plain data."""
        low, high = self.interval
        return {
            'episodes': self.episodes,
            'wins': self.wins,
            'draws': self.draws,
            'losses': self.losses,
            'goal_difference': self.goal_difference,
            'win_rate': self.win_rate,
            'win_rate_interval': [low, high],
        }

    def raise_error(self) -> None:
        """Raises the error that stopped the run, if there was one."""
        if self.error is not None:
            raise self.error


def _gather(value: Any, columns: torch.Tensor) -> Any:
    if isinstance(value, dict):
        return {name: _gather(item, columns) for name, item in value.items()}
    rows = torch.arange(value.shape[0], device=value.device)[:, None]
    return value[rows, columns.to(value.device)]


def _seed_of(seed: int, side: int) -> int:
    return int(np.random.SeedSequence([int(seed), side]).generate_state(1, dtype=np.uint32)[0])


class _Side:
    def __init__(self, index: int, controller: Controller | None, owned: bool) -> None:
        self.index = index
        self.controller = controller
        self.owned = owned
        self.difficulty: float | None = None
        self.control: dict[str, Any] = {}
        self.first = 0
        self.agents = 0
        self.chosen: list[int] = []
        self.held: list[dict[int, int]] = []
        self.spec: Spec | None = None
        self.timeout = 300.0
        self.convert: str | None = None
        self.builtin = False
        self.representation = 'raw'
        self.formation: list[tuple[int, bool]] = []

    @property
    def columns(self) -> tuple[int, int]:
        return self.first, self.first + self.agents


class Match:
    """Combines a batch with the controllers of its sides, which you name by `left` and `right`.

    A side that you leave out is yours, so you take its observations from `step` and give back
    its actions. A controller that you pass as an object stays yours to close, and one that
    `Match` builds from a file or an entry closes with the match.
    """

    def __init__(
        self,
        batch: Any,
        left: Any = None,
        right: Any = None,
        control: dict[str, dict[str, Any]] | None = None,
        seed: int = 0,
        check: bool = True,
    ) -> None:
        """Checks the controllers against the batch and refuses a mismatch, naming both sides.

        Args:
            batch: The `BatchEnvironment`, with `auto_reset` as you need it.
            left: The controller of the left side, in any form that `load` takes, or `builtin`.
            right: The controller of the right side.
            control: Control tables by side, `{'left': {...}}`, over those of the controllers.
            seed: The seed from which each side's own seed derives.
            check: Whether to check every answer against the controller's action set. The check
                reads the actions, which waits for the device, so a training loop may turn it off.
        """
        self.batch = batch
        self._inner = unwrap(batch)
        inner = self._inner
        self.environments = inner.environment_count
        self.seed = int(seed)
        self.check = check
        self._device = inner._torch_device
        for name in ('left_agents', 'right_agents'):
            values = getattr(inner, name)
            if not bool((values == values[0]).all()):
                raise ConfigurationError(f'expected the same {name} in every environment')
        self._agents = (int(inner.left_agents[0]), int(inner.right_agents[0]))
        self._players = (int(inner.left_players[0]), int(inner.right_players[0]))
        self._sides: list[_Side] = []
        self._owned: list[Controller] = []
        control = control or {}
        for index, given in enumerate((left, right)):
            self._sides.append(self._side(index, given, control.get(('left', 'right')[index])))
        self.yours = [
            side.index for side in self._sides if side.controller is None and not side.builtin
        ]
        self._started = False
        self._in_flight: Any = None
        self._closed = False
        count = self.environments
        device = self._device
        self._live = torch.ones(count, dtype=torch.bool, device=device)
        self._first = torch.ones(count, dtype=torch.bool, device=device)
        self._count_steps = torch.zeros(count, dtype=torch.int64, device=device)
        self._finished_steps = self._count_steps
        self._applied = torch.zeros((count, inner.agents), dtype=torch.int32, device=device)
        self._observation: Any = None
        self._length: int | None = None
        zero = torch.zeros((), dtype=torch.float64, device=device)
        self._episode_return = torch.zeros(count, dtype=torch.float64, device=device)
        self._episode_length = torch.zeros(count, dtype=torch.int64, device=device)
        self._total_return, self._total_length, self._total_count = (
            zero.clone(),
            zero.clone().long(),
            zero.clone().long(),
        )
        self.extras: dict[int, dict[str, Any]] = {}

    # Building the sides.

    def _side(self, index: int, given: Any, table: dict[str, Any] | None) -> _Side:
        inner = self._inner
        if given is None:
            return _Side(index, None, False)
        owned = not (
            isinstance(given, Controller) or (hasattr(given, 'act') and not isinstance(given, type))
        )
        controller = load(given)
        if isinstance(controller, Builtin):
            side = _Side(index, None, False)
            side.builtin = True
            side.difficulty = controller.difficulty
            if self._agents[index]:
                raise ConfigurationError(
                    f'expected {("left", "right")[index]} to have no agents for the built-in AI, '
                    f'found {self._agents[index]}, so create the batch with '
                    f'{("left", "right")[index]}_agents=0'
                )
            if controller.difficulty is not None:
                pair = (
                    (controller.difficulty, None) if index == 0 else (None, controller.difficulty)
                )
                self.batch.set_difficulty(pair)
            return side
        side = _Side(index, controller, owned)
        if owned:
            self._owned.append(controller)
        side.agents = self._agents[index]
        side.first = 0 if index == 0 else self._agents[0]
        name = ('left', 'right')[index]
        side.control = normalize({**(getattr(controller, 'control', None) or {}), **(table or {})})
        side.timeout = float(getattr(controller, 'timeout', 300.0))
        batch_count = inner.action_count
        if (side.control['action_set'] == 'full' and batch_count != 33) or (
            side.control['action_set'] == 'v2' and batch_count < 20
        ):
            raise ConfigurationError(
                f'expected a batch with action_set={side.control["action_set"]!r} for the '
                f'{name} controller, found a batch of {batch_count} actions'
            )
        if side.control['assignment'] == 'fixed' and batch_count < 20:
            raise ConfigurationError(
                f"expected a batch with action_set='v2' or 'full' for the fixed assignment of "
                f'the {name} controller, found a batch of {batch_count} actions'
            )
        if side.agents == 0:
            raise ConfigurationError(
                f'expected agents on the {name} side for its controller, found none, so create '
                f'the batch with {name}_agents'
            )
        path = _control.scenario_path(inner)
        formation = _control.formation(path, index)
        side.formation = formation
        players = side.control['players']
        if side.control['assignment'] == 'dynamic':
            if players is not None and players != side.agents:
                raise ConfigurationError(
                    f'expected {players} agents for the {name} controller, found the batch gives '
                    f'its {name} side {side.agents}'
                )
            side.control['players'] = side.agents
        else:
            able = len(_control.controllable(formation))
            if side.agents != able:
                raise ConfigurationError(
                    f'expected {able} agents for the fixed assignment of the {name} controller, '
                    f'the controllable players of the scenario, found the batch gives its {name} '
                    f'side {side.agents}'
                )
        side.chosen = _control.selected(side.control, formation)
        if side.control['assignment'] == 'dynamic':
            side.chosen = side.chosen[: side.agents]
        side.held = [{} for _ in range(self.environments)]
        self._representation(side)
        return side

    def _representation(self, side: _Side) -> None:
        inner = self._inner
        wanted = getattr(side.controller, 'representation', 'raw')
        name = ('left', 'right')[side.index]
        if wanted not in _REPRESENTATIONS:
            raise ConfigurationError(
                f'expected a representation from {", ".join(_REPRESENTATIONS)}, found {wanted!r}'
            )
        have_raw = inner._raw_fields is not None
        if wanted == 'raw' and not have_raw:
            raise ConfigurationError(
                f"expected a batch with raw=True for the {name} controller, which observes 'raw'"
            )
        if wanted != 'raw' and wanted != inner._representation:
            if not have_raw:
                raise ConfigurationError(
                    f'expected a batch with raw=True for the {name} controller, which observes '
                    f'{wanted!r} while the batch gives {inner._representation!r}'
                )
            if wanted not in ('simple115', 'simple115v2'):
                raise ConfigurationError(
                    f'expected the batch to give {wanted!r} itself for the {name} controller, '
                    "since only 'raw', 'simple115' and 'simple115v2' convert on the host"
                )
            side.convert = wanted
        side.representation = wanted

    # The session.

    def _spec(self, side: _Side) -> Spec:
        control = side.control
        controller = side.controller
        formation = side.formation
        return Spec(
            scenario=self._inner.scenario_name,
            length=self._length,
            environments=self.environments,
            agents=len(side.chosen) if control['assignment'] == 'fixed' else side.agents,
            action_set=control['action_set'],
            representation=side.representation,
            players=control['players'],
            assignment=control['assignment'],
            selection=list(control['selection']),
            goalkeeper=control['goalkeeper'],
            roles=[formation[i][0] for i in side.chosen],
            seed=_seed_of(self.seed, side.index),
            side=side.index,
            mirrored=bool(getattr(controller, 'mirrored', True)),
            options=dict(getattr(controller, 'options', None) or {}),
        )

    def _start(self) -> None:
        if self._started:
            return
        self._started = True
        if self._inner._raw_fields is not None:
            self._length = int(self._inner.raw()['steps_left'][0, 0])
        for side in self._sides:
            if side.controller is None:
                continue
            side.spec = self._spec(side)
            self._guard(side, lambda side=side: side.controller.start(side.spec))

    def _guard(self, side: _Side, call: Any) -> Any:
        try:
            return call()
        except BaseException as error:
            self._attach(error, side)
            raise

    def _attach(self, error: BaseException, side: _Side) -> None:
        step = int(self._count_steps.max()) if self._started else 0
        if isinstance(error, ControllerError):
            if error.side is None:
                error.side = side.index
            if error.step is None:
                error.step = step
            return
        for name, value in (('side', side.index), ('step', step)):
            try:
                setattr(error, name, value)
            except (AttributeError, TypeError):  # pragma: no cover, a slotted exception
                pass

    # Observing.

    def _yours_columns(self) -> list[int]:
        found: list[int] = []
        for index in self.yours:
            first, last = (
                (0, self._agents[0]) if index == 0 else (self._agents[0], self._inner.agents)
            )
            found.extend(range(first, last))
        return found

    def _slice_yours(self, value: Any) -> Any:
        columns = self._yours_columns()
        if not columns:
            return None
        if isinstance(value, dict):
            return {name: self._slice_yours(item) for name, item in value.items()}
        first, last = columns[0], columns[-1] + 1
        return value[:, first:last]

    def _fixed_columns(self, side: _Side, raw: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        active = raw['active'][:, :].cpu().numpy()
        count = len(side.chosen)
        picked = np.zeros((self.environments, count), dtype=np.int64)
        valid = np.ones((self.environments, count), dtype=bool)
        first = self._first.cpu().numpy()
        for environment in range(self.environments):
            if first[environment]:
                side.held[environment].clear()
            found = _control.columns(active[environment], side.chosen, side.held[environment])
            if found is None:
                picked[environment] = np.arange(count)
                valid[environment] = False
                continue
            chosen, missing = found
            picked[environment] = chosen
            valid[environment, [chosen.index(column) for column in missing]] = False
        return torch.as_tensor(picked), torch.as_tensor(valid)

    def _observe(self, side: _Side) -> tuple[Observation, Any, torch.Tensor]:
        inner = self._inner
        low, high = side.columns
        device = self._device
        count = self.environments
        columns = None
        raw = None
        if side.representation == 'raw' or side.convert:
            raw = {name: value[:, low:high] for name, value in inner.raw().items()}
        if side.control['assignment'] == 'fixed':
            columns, valid = self._fixed_columns(
                side, raw if raw is not None else {'active': inner.raw()['active'][:, low:high]}
            )
            active = valid.to(device)
        else:
            active = torch.ones((count, side.agents), dtype=torch.bool, device=device)
        active = active & self._live[:, None]
        if side.representation == 'raw':
            data: Any = raw
        elif side.convert:
            data = self._convert(side, raw)
        else:
            data = self._observation_tensor()[:, low:high]
        if columns is not None:
            data = _gather(data, columns)
        own, other = (0, 1) if side.index == 0 else (1, 0)
        sizes = [
            torch.full((count,), self._players[own], dtype=torch.int32, device=device),
            torch.full((count,), self._players[other], dtype=torch.int32, device=device),
        ]
        previous = self._applied[:, low:high]
        if columns is not None:
            previous = _gather(previous, columns)
        observation = Observation(
            data=data,
            first=self._first,
            active=active,
            left_players=sizes[0],
            right_players=sizes[1],
            previous=previous,
            step=self._count_steps,
            side=torch.full((count,), side.index, dtype=torch.int8, device=device),
            meta={'overage': _OVERAGE},
        )
        return observation, columns, active

    def _observation_tensor(self) -> Any:
        if self._observation is None:
            self._observation = self._inner.observation()
        return self._observation

    def _convert(self, side: _Side, raw: dict[str, Any]) -> torch.Tensor:
        from .._raw import grf_observation
        from ..gfootball.env.wrappers import Simple115StateWrapper

        cpu = {name: value.cpu() for name, value in raw.items()}
        own = int(self._players[side.index])
        other = int(self._players[1 - side.index])
        fixed = side.convert == 'simple115v2'
        rows = []
        for environment in range(self.environments):
            views = [
                grf_observation(cpu, environment, agent, own, other) for agent in range(side.agents)
            ]
            rows.append(Simple115StateWrapper.convert_observation(views, fixed))
        return torch.as_tensor(np.asarray(rows, dtype=np.float32)).to(self._device)

    # Answering.

    def _validate(self, side: _Side, answer: Any, active: torch.Tensor) -> torch.Tensor:
        name = side.index
        expected = (
            self.environments,
            len(side.chosen) if side.control['assignment'] == 'fixed' else side.agents,
        )
        actions = answer.actions
        if not isinstance(actions, torch.Tensor):
            try:
                actions = torch.as_tensor(np.asarray(actions))
            except (TypeError, ValueError, RuntimeError) as error:
                raise ControllerError(
                    f'expected integer actions, found {answer.actions!r}', 'type', name
                ) from error
        if actions.dtype.is_floating_point or actions.dtype.is_complex:
            raise ControllerError(f'expected integer actions, found {actions.dtype}', 'type', name)
        if actions.dim() == 1 and expected[1] == 1:
            actions = actions[:, None]
        if tuple(actions.shape) != expected:
            raise ControllerError(
                f'expected actions of the shape {expected}, [environments, agents], found '
                f'{tuple(actions.shape)}',
                'shape',
                name,
            )
        actions = actions.to(self._device).to(torch.int64)
        count = ACTION_COUNTS[side.control['action_set']]
        if self.check:
            outside = ((actions < -count) | (actions >= count)) & active
            if bool(outside.any()):
                row, column = (int(item) for item in outside.nonzero()[0])
                raise ControllerError(
                    f'expected actions from {-count} to {count - 1} of the '
                    f'{side.control["action_set"]} set, found {int(actions[row, column])}',
                    'range',
                    name,
                    environment=row,
                )
        return torch.where(actions < 0, actions + count, actions)

    def _place(
        self,
        side: _Side,
        actions: torch.Tensor,
        columns: Any,
        active: torch.Tensor,
        full: torch.Tensor,
    ) -> None:
        low, high = side.columns
        # An environment that has ended its quota idles, and the built-in AI plays what no
        # answer covers in a batch that has it.
        idle = 0
        if side.control['assignment'] == 'fixed':
            block = torch.full(
                (self.environments, side.agents),
                _control.BUILTIN_AI,
                dtype=torch.int32,
                device=self._device,
            )
            usable = active & self._live[:, None]
            rows = torch.arange(self.environments, device=self._device)[:, None].expand_as(actions)
            target = columns.to(self._device)
            values = torch.where(
                usable,
                actions.to(torch.int32),
                torch.full_like(actions, _control.BUILTIN_AI, dtype=torch.int32),
            )
            block[rows, target] = values
            block[~self._live] = idle
            full[:, low:high] = block
        else:
            full[:, low:high] = torch.where(
                self._live[:, None],
                actions.to(torch.int32),
                torch.zeros_like(actions, dtype=torch.int32),
            )

    def _yours_actions(self, actions: Any) -> torch.Tensor | None:
        columns = self._yours_columns()
        if not columns:
            if actions is not None:
                raise ConfigurationError(
                    'expected no actions, found some, since every side has a controller'
                )
            return None
        if actions is None:
            raise ConfigurationError(
                f'expected actions for the {len(columns)} agents that are yours, found none'
            )
        values = (
            actions if isinstance(actions, torch.Tensor) else torch.as_tensor(np.asarray(actions))
        )
        if values.dim() == 1 and len(columns) == 1:
            values = values[:, None]
        if tuple(values.shape) != (self.environments, len(columns)):
            raise ConfigurationError(
                f'expected actions of the shape {(self.environments, len(columns))}, found '
                f'{tuple(values.shape)}'
            )
        return values.to(self._device).to(torch.int32)

    # The public steps.

    def reset(self, seeds: Any = None, mask: Any = None) -> Any:
        """Resets the batch, or the environments of `mask`, and returns your side's observation."""
        self._check_open()
        observation = self.batch.reset(seeds=seeds, mask=mask)
        self._observation = observation
        if mask is None:
            self._first = torch.ones(self.environments, dtype=torch.bool, device=self._device)
            self._count_steps = torch.zeros_like(self._count_steps)
            self._episode_return.zero_()
            self._episode_length.zero_()
        else:
            chosen = torch.as_tensor(mask).to(self._device).bool()
            self._first = self._first | chosen
            self._count_steps = torch.where(
                chosen, torch.zeros_like(self._count_steps), self._count_steps
            )
            self._episode_return = torch.where(
                chosen, torch.zeros_like(self._episode_return), self._episode_return
            )
            self._episode_length = torch.where(
                chosen, torch.zeros_like(self._episode_length), self._episode_length
            )
        self._start()
        return self._slice_yours(observation)

    def step(self, actions: Any = None) -> tuple[Any, Any, Any, dict[str, Any]]:
        """Plays one step, with your side's actions, and returns what `BatchEnvironment` returns."""
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions: Any = None) -> None:
        """Sends the observations to the process controllers and returns, so you can work."""
        self._check_open()
        if self._in_flight is not None:
            raise RuntimeError('a step is in flight; call step_wait before the next step_async')
        mine = self._yours_actions(actions)
        if not self._started:
            raise RuntimeError('call reset before the first step')
        prepared = []
        for side in self._sides:
            if side.controller is None:
                continue
            observation, columns, active = self._observe(side)
            if side.controller.asynchronous:
                self._guard(
                    side,
                    lambda side=side, observation=observation: side.controller.send(observation),
                )
            prepared.append((side, observation, columns, active))
        self._in_flight = (mine, prepared)

    def step_wait(self) -> tuple[Any, Any, Any, dict[str, Any]]:
        """Collects the answers, steps the batch and returns your side's outputs."""
        if self._in_flight is None:
            raise RuntimeError('no step is in flight; call step_async first')
        mine, prepared = self._in_flight
        try:
            full = torch.zeros(
                (self.environments, self._inner.agents), dtype=torch.int32, device=self._device
            )
            columns = self._yours_columns()
            if columns:
                full[:, columns[0] : columns[-1] + 1] = mine
            for side, observation, picked, active in prepared:
                controller = side.controller
                if controller.asynchronous:
                    answer = self._guard(
                        side,
                        lambda controller=controller, side=side: controller.receive(side.timeout),
                    )
                    if answer is None:
                        raise ControllerError(
                            f'no answer within {side.timeout:g} s',
                            'timeout',
                            side.index,
                            int(self._count_steps.max()),
                        )
                else:
                    answer = as_actions(
                        self._guard(
                            side,
                            lambda controller=controller, observation=observation: controller.act(
                                observation
                            ),
                        )
                    )
                answer = as_actions(answer)
                try:
                    values = self._validate(side, answer, active)
                except ControllerError as error:
                    self._attach(error, side)
                    raise
                self.extras[side.index] = answer.extras
                self._place(side, values, picked, active, full)
        finally:
            self._in_flight = None
        self._applied = full
        observation, reward, done, information = self.batch.step(full)
        self._observation = observation
        self._account(reward, done)
        ran = self._count_steps + 1
        self._finished_steps = ran
        if self._inner.auto_reset:
            self._first = done
            self._count_steps = torch.where(done, torch.zeros_like(ran), ran)
        else:
            self._first = torch.zeros_like(done)
            self._count_steps = ran
        out_reward = None if not self.yours else self._slice_yours(reward)
        return self._slice_yours(observation), out_reward, done, information

    def _account(self, reward: torch.Tensor, done: torch.Tensor) -> None:
        columns = self._yours_columns() or list(
            range(self._sides[0].columns[0], self._sides[0].columns[1])
        )
        if not columns:
            columns = list(range(self._inner.agents))
        step = reward[:, columns[0] : columns[-1] + 1].sum(1).to(torch.float64)
        self._episode_return = self._episode_return + step
        self._episode_length = self._episode_length + 1
        ended = done.to(torch.bool) & self._live
        self._total_return = self._total_return + (self._episode_return * ended).sum()
        self._total_length = self._total_length + (self._episode_length * ended).sum()
        self._total_count = self._total_count + ended.sum()
        self._episode_return = torch.where(
            ended, torch.zeros_like(self._episode_return), self._episode_return
        )
        self._episode_length = torch.where(
            ended, torch.zeros_like(self._episode_length), self._episode_length
        )

    @property
    def episodes(self) -> Episodes:
        """Returns the totals of the finished episodes, which waits for the device once."""
        return Episodes(int(self._total_count), float(self._total_return), int(self._total_length))

    # Swapping, running and closing.

    def set(self, left: Any = None, right: Any = None) -> None:
        """Gives a side another controller, which each environment takes at its next episode."""
        self._check_open()
        for index, given in enumerate((left, right)):
            if given is None:
                continue
            side = self._sides[index]
            if side.controller is None:
                raise ConfigurationError(
                    f'expected a side that has a controller, found the {("left", "right")[index]} '
                    'side yours or built-in, so make a new Match to change it'
                )
            if not isinstance(side.controller, Pool) or not side.controller.swappable:
                pool = Pool([side.controller], control=side.control)
                pool.swappable = True
                pool.owned = {0} if side.owned else set()
                if self._started:
                    pool.adopt(side.spec)
                self._owned = [c for c in self._owned if c is not side.controller]
                self._owned.append(pool)
                side.controller, side.owned = pool, True
            new = load(given)
            wanted = getattr(new, 'representation', 'raw')
            if wanted != side.representation:
                raise ConfigurationError(
                    f'expected a controller of the representation {side.representation!r}, '
                    f'found {wanted!r}'
                )
            owned = not (
                isinstance(given, Controller)
                or (hasattr(given, 'act') and not isinstance(given, type))
            )
            side.controller.set_pool([new], owned=[owned])

    def run(self, episodes: int = 1, record: bool = False) -> Result:
        """Plays `episodes` complete episodes with every side under a controller.

        A breach of the contract, or an exception of a controller, stops the run and is returned
        in `Result.error`, with the episodes that finished before it.
        """
        self._check_open()
        if self.yours:
            names = ', '.join(('left', 'right')[i] for i in self.yours)
            raise ConfigurationError(
                f'expected a controller on every side to run, found {names} without one, so give '
                'it a controller or step the match yourself'
            )
        if self._inner._raw_fields is None:
            raise ConfigurationError('expected a batch with raw=True to run, which reads the score')
        count = self.environments
        live = torch.zeros(count, dtype=torch.bool)
        launched = min(count, episodes)
        live[:launched] = True
        self._live = live.to(self._device)
        scores: list[tuple[int, int]] = []
        lengths: list[int] = []
        recordings: list[Any] = []
        recorder = None
        if record:
            try:
                from cuju.recording import record as make_recorder
            except ImportError:
                raise ConfigurationError(
                    'expected cuju.recording for record=True, found it missing, so install '
                    'cuju-recording'
                ) from None
            recorder = make_recorder(self.batch, list(range(launched)), recordings.append)
        error: BaseException | None = None
        try:
            self.reset()
            flipped = not self._agents[0]
            while bool(live.any()):
                _, _, done, information = self.step()
                finished = (done.cpu() & live).nonzero().flatten().tolist()
                if not finished:
                    continue
                final = information['final_raw']
                taken = final['score'][:, 0].cpu().numpy()
                steps = self._finished_steps.cpu().numpy()
                restart = []
                for environment in finished:
                    own, other = (int(taken[environment][0]), int(taken[environment][1]))
                    scores.append((other, own) if flipped else (own, other))
                    lengths.append(int(steps[environment]))
                    if launched < episodes:
                        launched += 1
                        restart.append(environment)
                    else:
                        live[environment] = False
                if restart and not self._inner.auto_reset:
                    mask = torch.zeros(count, dtype=torch.bool)
                    mask[restart] = True
                    self.reset(mask=mask)
                self._live = live.to(self._device)
        except Exception as caught:
            error = caught
        finally:
            if recorder is not None:
                recorder.close()
            self._live = torch.ones(count, dtype=torch.bool, device=self._device)
        return Result(
            np.asarray(scores, dtype=np.int64).reshape(-1, 2),
            np.asarray(lengths, dtype=np.int64),
            error,
            recordings,
        )

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError('the match is closed')

    def close(self) -> None:
        """Closes every controller that the match built, and the processes they started."""
        if self._closed:
            return
        self._closed = True
        for controller in self._owned:
            controller.close()
        self._owned = []

    def __enter__(self) -> 'Match':
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()


def _batch_for(
    controllers: list[Any],
    scenario: str,
    environments: int,
    seed: int,
    device: Device,
    controls: dict[str, dict[str, Any]],
) -> Any:
    import os

    import cuju

    from .._environment import BatchEnvironment

    path = scenario
    if not os.path.isfile(path):
        path = os.path.join(os.path.dirname(cuju.__file__), 'data', 'scenarios', f'{scenario}.py')
    sizes = []
    rank = {'default': 0, 'v2': 1, 'full': 2}
    action = 'default'
    for index, controller in enumerate(controllers):
        if isinstance(controller, Builtin):
            sizes.append(0)
            continue
        table = {**(getattr(controller, 'control', None) or {}), **controls.get(SIDES[index], {})}
        control = normalize(table)
        sizes.append(_control.agents_of(control, _control.formation(path, index)))
        needed = 'v2' if control['assignment'] == 'fixed' else control['action_set']
        if rank[needed] > rank[action]:
            action = needed
    return BatchEnvironment(
        scenario,
        environment_count=environments,
        device=device,
        seed=seed,
        left_agents=sizes[0],
        right_agents=sizes[1],
        representation='raw',
        auto_reset=False,
        action_set=action,
        raw=True,
    )


def play_batch(
    left: Any,
    right: Any,
    scenario: str,
    environments: int = 64,
    episodes: int | None = None,
    seed: int = 0,
    device: Device = 'cpu',
    record: bool = False,
    control: dict[str, dict[str, Any]] | None = None,
) -> Result:
    """Plays episodes in many environments at once and returns the `Result`."""
    controllers = [load(side) for side in (left, right)]
    batch = _batch_for(controllers, scenario, environments, seed, device, control or {})
    try:
        with Match(batch, *controllers, control=control, seed=seed) as match:
            return match.run(episodes or environments, record=record)
    finally:
        batch.close()
        for controller, given in zip(controllers, (left, right), strict=True):
            if controller is not given and not isinstance(given, Controller):
                controller.close()


def play(
    left: Any,
    right: Any,
    scenario: str,
    seed: int = 0,
    device: Device = 'cpu',
    episodes: int = 1,
    record: bool = False,
    control: dict[str, dict[str, Any]] | None = None,
) -> Result:
    """Plays `episodes` episodes of two controllers, one environment for each, and returns them."""
    return play_batch(left, right, scenario, episodes, episodes, seed, device, record, control)
