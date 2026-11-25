# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numbers
from typing import Any

import torch

from . import _config
from ._assets import _split_name
from ._batch import log as _log
from ._batch.environment import Batch
from .errors import ConfigurationError, LibraryError

__all__ = ['BatchEnvironment']

_autos_logged = False


def _log_automatic(device: Any, batch: Batch) -> None:
    global _autos_logged
    if _autos_logged or not (isinstance(device, str) and device == 'auto'):
        return
    _autos_logged = True
    _log._handler()
    _log.LOGGER.info("device='auto' picked %s", batch.torch_device)


def _agents(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
        raise ConfigurationError(
            f'expected {name} to be a whole number, 0 or more, found {value!r}'
        )
    return int(value)


def _frame_size(value: Any) -> tuple[int, int]:
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return int(value), 720
    try:
        width, height = value
    except (TypeError, ValueError):
        raise ConfigurationError(
            f'expected frame_size to be (width, height) or a width, found {value!r}'
        ) from None
    return width, height


class BatchEnvironment:
    """A batch of football environments on the GPU or the CPU, which plays GRF bit for bit.

    Every tensor that a method returns views a buffer of the batch on its device, without a copy,
    and the next `reset` or `step` overwrites it, so a caller clones what it keeps. The agent slots
    are the left agents first, then the right agents, and each agent sees the pitch from its own
    end.

    What changes the results goes in the arguments of the constructor. What changes only the speed
    goes in the machine config, the file `engine.toml` that `CUJU_CONFIG` names, or that is
    `~/.config/cuju/engine.toml` on Linux and macOS and `%APPDATA%\\cuju\\engine.toml` on Windows.
    Its table `[run]` holds `threads`, `debug`, `allocator`, `pin_threads`, `numa`, `pin_memory`,
    `render_threads`, `render_workers` and `log`, and the argument `config` overrides them for one
    batch.

    The methods `step` and `reset_done` compile under `torch.compile(fullgraph=True)`, and the
    steps of a GPU batch can be captured in CUDA graphs. A batch whose memory exceeds the free
    memory of its GPU warns, which is the sign on Windows and WSL2 that the driver moved it to
    shared memory.

    Actions are integers from 0 to `action_count - 1`. The CPU and a batch with `debug` on refuse
    others. With `debug` off, a GPU batch plays an action outside that range on the device as the
    idle action.

    Attributes:
        environment_count: The number of environments.
        agents: The size of the agent axis, `left_agents + right_agents`.
        left_agents: The agents of the left team.
        right_agents: The agents of the right team.
        action_count: The number of actions, 19 for `default`, 20 for `v2` and 33 for `full`.
        left_players: The players of the left team, as `len(observation['left_team'])` in GRF for
            a left agent.
        right_players: The players of the right team.
        device: The `torch.device` that holds the buffers of the batch, such as `cuda:0`.
        scenario_name: The name of the scenario, without its folder or suffix.
        difficulty: `[environment_count, 2]` floats, the difficulty of the left and of the right
            team of each environment, the one that the batch sets, else the scenario's.
        stacked: Whether the observation stacks the last 4.
        memory_bytes: The bytes of memory that the batch has requested so far.
        single_observation_space, single_action_space, observation_space, action_space: The spaces
            of one environment and of the batch, as gymnasium describes them. `raw` has no
            observation space.
        run_config: The resolved machine settings, a plain dict from each key of `[run]` to its
            `value` and its `source`, which is `default`, `file` with the path, or `call`.

    Examples:
        >>> import torch
        >>> import cuju
        >>> batch = cuju.BatchEnvironment('5_vs_5', environment_count=1024, device='cpu')
        >>> observation = batch.reset()
        >>> tuple(observation.shape)
        (1024, 1, 115)
        >>> actions = torch.zeros((1024, 1), dtype=torch.int32)
        >>> observation, reward, done, information = batch.step(actions)
        >>> tuple(reward.shape), tuple(done.shape)
        ((1024, 1), (1024,))
    """

    def __init__(
        self,
        scenario: str,
        environment_count: int = 1024,
        device: str | torch.device = 'auto',
        seed: int = 0,
        left_agents: int = 1,
        right_agents: int = 0,
        difficulty: tuple[float | None, float | None] | None = None,
        representation: str = 'simple115v2',
        rewards: str = 'scoring',
        action_set: str = 'default',
        auto_reset: bool = True,
        stacked: bool = False,
        channel_dimensions: tuple[int, int] = (96, 72),
        raw: bool = False,
        frames: bool = False,
        frame_size: tuple[int, int] | int = (1280, 720),
        config: dict[str, Any] | None = None,
    ) -> None:
        if isinstance(scenario, str) and _split_name(scenario) is not None:
            raise ConfigurationError(
                'expected a scenario without agent counts, such as '
                f'{_split_name(scenario)[0]!r}, with left_agents and right_agents, found '
                f'{scenario!r}'
            )
        left_agents, right_agents = (
            _agents('left_agents', left_agents),
            _agents('right_agents', right_agents),
        )
        if difficulty is not None and not (
            isinstance(difficulty, (tuple, list)) and len(difficulty) == 2
        ):
            raise ConfigurationError(
                f'expected difficulty as a (left, right) pair, found {difficulty!r}'
            )
        width, height = _frame_size(frame_size)
        options = _config.machine_options(config)
        pixels = representation in ('pixels', 'pixels_gray')
        self._config = _config.resolve(config)
        self._batch = Batch(
            scenario,
            environment_count=environment_count,
            representation=representation,
            rewards=rewards,
            action_set=action_set,
            seed=seed,
            auto_reset=auto_reset,
            agent_counts=[(left_agents, right_agents)],
            with_raw=raw,
            difficulty=None if difficulty is None else [tuple(difficulty)],
            device=device,
            channel_dimensions=channel_dimensions,
            stacked=stacked,
            render_frames=frames,
            render_resolution_x=width,
            render_resolution_y=height,
            signed_actions=False,
            **options,
        )
        batch = self._batch
        try:
            self.environment_count = batch.environment_count
            self.left_agents, self.right_agents = left_agents, right_agents
            self.agents = batch.agents
            self.action_count = batch.action_count
            self.left_players = self._uniform(batch.left_players, 'left_players')
            self.right_players = self._uniform(batch.right_players, 'right_players')
            self.device = batch.torch_device
            self.scenario_name = batch.scenario_name
            self.difficulty = batch.difficulty
            self.stacked = batch.stacked
            self._raw_on = bool(raw) or representation == 'raw'
            self._frames_on = bool(frames) or pixels
            self._representation = representation
            self._final_raw = batch._raw('final_raw') if raw and representation != 'raw' else None
            self.run_config = self._applied(options, batch.device)
        except BaseException:
            batch.close()
            raise
        _log_automatic(device, batch)

    def _uniform(self, values: torch.Tensor, name: str) -> int:
        first = int(values[0])
        if not bool((values == first).all()):
            raise LibraryError(f'expected the same {name} in every environment, found {values}')
        return first

    def _applied(self, options: dict[str, Any], backend: str) -> dict[str, dict[str, Any]]:
        resolved = {key: dict(entry) for key, entry in self._config.items()}
        resolved['pin_memory']['value'] = bool(options['pin_memory']) and backend == 'cpu'
        return resolved

    def __repr__(self) -> str:
        if self._batch._native.environment is None:
            return 'BatchEnvironment(closed)'
        return (
            f'BatchEnvironment({self.scenario_name!r}, environment_count={self.environment_count}, '
            f'device={str(self.device)!r}, agents={self.agents})'
        )

    def __reduce__(self):
        raise TypeError(
            'a batch cannot be pickled; create it inside the worker process that uses it'
        )

    def __enter__(self) -> BatchEnvironment:
        self._batch._handle()
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()

    def close(self) -> None:
        self._batch.close()

    @property
    def action_space(self) -> Any:
        return self._batch.action_space

    @property
    def single_action_space(self) -> Any:
        return self._batch.single_action_space

    @property
    def observation_space(self) -> Any:
        return self._batch.observation_space

    @property
    def single_observation_space(self) -> Any:
        return self._batch.single_observation_space

    @property
    def memory_bytes(self) -> int:
        return self._batch.memory_bytes

    def reset(self, seeds: Any = None, mask: Any = None, reverse: Any = None) -> Any:
        return self._batch.reset(seeds=seeds, reverse=reverse, mask=mask)

    def step(self, actions: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
        return self._outputs(self._batch.step(actions))

    def step_async(self, actions: Any) -> None:
        self._batch.step_async(actions)

    def step_wait(self) -> tuple[Any, Any, Any, dict[str, Any]]:
        return self._outputs(self._batch.step_wait())

    def reset_done(self) -> None:
        self._batch.reset_done()

    def set_difficulty(self, values: Any, mask: Any = None) -> None:
        self._batch.set_difficulty(values, mask)
        self.difficulty = self._batch.difficulty

    def _outputs(self, outputs):
        observation, reward, done, information = outputs
        final = {
            'final_observation': information['final_observation'],
            'score_reward': information['score_reward'],
        }
        if self._raw_on:
            final['final_raw'] = (
                final['final_observation'] if self._representation == 'raw' else self._final_raw
            )
        return observation, reward, done.view(torch.bool), final

    def get_state(self, index: int, stacked: bool = False) -> bytes:
        return self._batch.get_state(index, stacked)

    def set_state(self, index: int, state: bytes, stacked: bool = False) -> None:
        self._batch.set_state(index, state, stacked)

    def raw(self) -> dict[str, torch.Tensor]:
        if not self._raw_on:
            raise LibraryError(
                "expected a batch with raw=True or representation='raw' to return raw(), found "
                'neither'
            )
        return self._batch.raw()

    def frame(self) -> torch.Tensor:
        if not self._frames_on:
            raise LibraryError(
                "expected a batch with frames=True or representation='pixels' to return frame(), "
                'found neither'
            )
        return self._batch.frame()


BatchEnvironment.__module__ = 'cuju'
