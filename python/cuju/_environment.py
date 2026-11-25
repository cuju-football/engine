# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""`BatchEnvironment`, the public batch of football environments, which maps onto `Batch`."""

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
    """Logs once per process which device `auto` picked."""
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
    """Returns the `(width, height)` of the frames, 720 high when only a width is set."""
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
        """Creates the batch and allocates its buffers.

        Args:
            scenario: A shipped scenario such as `5_vs_5` or `11_vs_11_kaggle`, or the path of a
                GRF scenario file. `scenario_names()` lists the shipped ones.
            environment_count: The number of environments.
            device: `auto`, `cpu`, `cuda`, `cuda:N` or a `torch.device`. Both backends give the
                same outputs bit for bit, and the device decides only where the batch runs. `auto`
                takes the current CUDA device where PyTorch sees a GPU and the CUDA library is
                installed, else the CPU, and logs its choice once.
            seed: The seed of the batch, from 0 to 2**64 - 1. Environment `index` plays as GRF's
                `create_environment` after `random.seed(seed * 2**32 + index)`.
            left_agents: The agents of the left team, which the policy controls.
            right_agents: The agents of the right team. The players that no agent controls play
                GRF's built-in AI.
            difficulty: The `(left, right)` difficulty of the teams, each from 0 to 1 or `None`
                for the scenario's.
            representation: `simple115`, `simple115v2`, `extracted` (the SMM), `raw` (a dictionary
                of named tensors), `pixels` or `pixels_gray`.
            rewards: `scoring`, or `scoring,checkpoints` to add the checkpoint reward.
            action_set: `default`, `v2` or `full`.
            auto_reset: Whether a step resets every environment whose episode ended. `False`
                leaves each such environment at its end until `reset_done` or `reset`.
            stacked: Whether the observation stacks the last 4 along its last axis, the oldest
                first, as GRF's `stacked`.
            channel_dimensions: The width and the height of the SMM in cells, which `pixels` and
                `pixels_gray` shrink their frames to as well.
            raw: Whether the batch also writes the raw observation, which `raw` returns and
                `information['final_raw']` holds, whatever the representation.
            frames: Whether the batch also renders GRF's frame of each environment, which `frame`
                returns.
            frame_size: The `(width, height)` of GRF's frames in pixels, the width a multiple of
                4. GRF's height stays 720 when only a width is given.
            config: Keys of the table `[run]` of the machine config that this batch overrides.

        Raises:
            ChoiceError: If `representation`, `action_set` or `device` names no known value. It is
                a `ValueError` and a `KeyError`.
            ConfigurationError: If an option is refused, such as a `seed` out of range, a
                `difficulty` outside 0 to 1, more agents than a team has players, a `stacked`
                `raw` batch or a machine config that holds another table or key. It is a
                `ValueError`, and its subclass `scenarios.ScenarioError` reports a refused
                scenario file.
            MissingFileError: If the scenario names no file or a library is missing. It is a
                `FileNotFoundError`.
            DeviceError: If `device` names a CUDA GPU and PyTorch sees none or fewer, the CPU
                lacks AVX2, BMI2 or FMA, or the device runs out of memory, which the message
                reports with the bytes requested and the bytes free. It is a `RuntimeError`.
            LibraryError: If the library fails to create the batch. It is a `RuntimeError`.
        """
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
            # The views never move and a compiled step cannot make them, so they are made now.
            self._final_raw = batch._raw('final_raw') if raw and representation != 'raw' else None
            self.run_config = self._applied(options, batch.device)
        except BaseException:
            batch.close()
            raise
        _log_automatic(device, batch)

    def _uniform(self, values: torch.Tensor, name: str) -> int:
        """Returns the one value that `values` holds, which a batch of one scenario has."""
        first = int(values[0])
        if not bool((values == first).all()):
            raise LibraryError(f'expected the same {name} in every environment, found {values}')
        return first

    def _applied(self, options: dict[str, Any], backend: str) -> dict[str, dict[str, Any]]:
        """Returns the resolved machine settings, with the `pin_memory` that applies."""
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
        # A batch owns device memory and threads, which no pickle can carry to another process.
        raise TypeError(
            'a batch cannot be pickled; create it inside the worker process that uses it'
        )

    def __enter__(self) -> BatchEnvironment:
        self._batch._handle()
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()

    def close(self) -> None:
        """Frees the batch and its buffers at once. A second call does nothing.

        The tensors that the batch returned view those buffers, so none may be used after it, and
        every later call on the batch raises `ClosedError`, a `RuntimeError`. A batch dropped
        without `close` frees them once no tensor that it returned remains.
        """
        self._batch.close()

    @property
    def action_space(self) -> Any:
        """The action space of the batch, `single_action_space` for each environment."""
        return self._batch.action_space

    @property
    def single_action_space(self) -> Any:
        """The action space of one environment, `MultiDiscrete` over the agents. It needs
        gymnasium or gym."""
        return self._batch.single_action_space

    @property
    def observation_space(self) -> Any:
        """The observation space of the batch, or `None` for `raw`, which has none in GRF."""
        return self._batch.observation_space

    @property
    def single_observation_space(self) -> Any:
        """The observation space of one environment, as GRF's with the same options, or `None`
        for `raw`. It needs gymnasium or gym."""
        return self._batch.single_observation_space

    @property
    def memory_bytes(self) -> int:
        """The bytes of memory that the batch has requested so far, its tables included.

        It counts device memory on the GPU and host memory on the CPU. The allocators round each
        request up. The first batch of the process on a GPU also makes the driver reserve the
        local memory of the kernels, which this count leaves out.
        """
        return self._batch.memory_bytes

    def reset(self, seeds: Any = None, mask: Any = None, reverse: Any = None) -> Any:
        """Starts the next episode of every environment, or of those that `mask` selects.

        Args:
            seeds: One seed per environment for this episode, as `game_engine_random_seed` in the
                configuration of GRF, or one seed for every environment. A seed of -1 draws that
                environment's seed as `None` does. `None` draws every seed.
            mask: `[environment_count]` booleans. Each environment where it is true starts its
                next episode as GRF's `reset()` does, and the others keep their episode, outputs
                and done flags. A mask on the device of the batch takes no copy and the call then
                waits for nothing, so a CUDA graph can capture it once every environment has had
                its first reset. `None` resets every environment and waits.
            reverse: One `reverse_team_processing` flag per environment, or one for every
                environment. `None` follows the seeds, as GRF does. Only the first reset of an
                environment reads it.

        Returns:
            The observation of every agent, `[environment_count, agents, 115]` for `simple115`
            and `simple115v2`, `[environment_count, agents, height, width, 4]` for `extracted`, or
            a dictionary of `[environment_count, agents, ...]` tensors for `raw`. A stack
            multiplies the last axis by 4.

        Raises:
            TypeError: If `seeds`, `reverse` or `mask` holds floating-point values.
            ArgumentError: If `seeds`, `reverse` or `mask` holds neither one value nor one per
                environment, or a seed or a flag lies outside its range.
            LibraryError: If the library fails to reset the batch.
            UndefinedPathError: If an environment reaches a path that GRF leaves undefined.
            DeviceError: If the device fails in a `debug` batch.
        """
        return self._batch.reset(seeds=seeds, reverse=reverse, mask=mask)

    def step(self, actions: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
        """Runs one step of every environment.

        On the GPU the step runs on the current stream of PyTorch and does not wait for the device.
        Under `torch.compile` it traces as the operator `cuju::step`, so a compiled function that
        steps the batch needs no graph break, and `mode='reduce-overhead'` may capture it.

        Args:
            actions: `[environment_count, agents]` indices from 0 to `action_count - 1`, or
                `[environment_count]` for one agent. A tensor on the device of the batch takes no
                copy. Actions in host memory may be reused as soon as the call returns.

        Returns:
            The observation, the reward `[environment_count, agents]`, the done flags
            `[environment_count]` as `bool`, and the information. The information maps
            `final_observation` to the observation of the step that ended each episode,
            `score_reward` to GRF's `info['score_reward']`, the left team's scoring reward,
            `[environment_count]` `int32`, and, with the raw observation on, `final_raw` to the
            raw observation of that step.

        Raises:
            TypeError: If the actions are not integers.
            ArgumentError: If the actions have another shape, or lie outside the action set in
                host memory or, with `debug` on, in GPU memory. It is a `ValueError` and an
                `IndexError`.
            LibraryError: If the batch has had no `reset`, or the library fails to run the step.
            UndefinedPathError: If an environment reaches a path that GRF leaves undefined. On
                the GPU the device traps, which ends the CUDA context of the process.
            DeviceError: If the device fails in a `debug` batch.
        """
        return self._outputs(self._batch.step(actions))

    def step_async(self, actions: Any) -> None:
        """Starts a step of every environment and returns at once. `step_wait` ends the step.

        On the CPU the batch steps in a thread of its own, with a copy of the actions, while the
        caller goes on, for example to run a policy on the GPU. On the GPU the step runs on the
        current stream, as `step` does. Until `step_wait` returns, the outputs are not ready, and
        any other call on the batch waits for the step first.

        Args:
            actions: As for `step`, and the caller may reuse them at once.

        Raises:
            TypeError, ArgumentError: As `step` raises them, before the step starts.
            RuntimeError: If a step is already in flight, or the library fails to start the step.
        """
        self._batch.step_async(actions)

    def step_wait(self) -> tuple[Any, Any, Any, dict[str, Any]]:
        """Waits for the step that `step_async` started and returns what `step` returns.

        Raises:
            RuntimeError: If no step is in flight, or the step failed as `step` would.
        """
        return self._outputs(self._batch.step_wait())

    def reset_done(self) -> None:
        """Starts the next episode of every environment whose episode ended.

        A batch with `auto_reset=False` leaves such environments at their end. The reset reads
        the done flags in the memory of the batch, so on the GPU it does not wait for the device.
        It writes the new observations and clears the done flags of the environments it resets.

        Raises:
            LibraryError: If the library fails to run the reset.
            UndefinedPathError: If an environment reaches a path that GRF leaves undefined.
            DeviceError: If the device fails in a `debug` batch.
        """
        self._batch.reset_done()

    def set_difficulty(self, values: Any, mask: Any = None) -> None:
        """Sets the difficulty that each environment plays from its next reset.

        Args:
            values: A `(left, right)` pair for the selected environments, each from 0 to 1 or
                `None` for the scenario's, or `[environment_count, 2]` values with NaN for the
                scenario's.
            mask: `[environment_count]` booleans that select the environments, or `None` for all.
                The others keep their difficulty.

        Raises:
            ConfigurationError: If a value lies outside 0 to 1.
            ArgumentError: If the shape of `values` or `mask` does not fit.
        """
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
        """Returns the state of one environment in the byte format of GRF.

        The bytes hold the length of a pickle as `int32`, the pickle of the episode logic and the
        serialized match.

        Args:
            index: The index of the environment.
            stacked: Whether the pickle also holds GRF's `FrameStack`, as the state of GRF with
                `stacked=True` does. It needs numpy, and a batch with `stacked=True`.

        Returns:
            The bytes, which `set_state` here or in GRF reads.

        Raises:
            ArgumentError: If the index is out of range. It is a `ValueError` and an
                `IndexError`.
            StateError: If `stacked` differs from the batch's `stacked`. It is a `ValueError` and
                a `RuntimeError`.
        """
        return self._batch.get_state(index, stacked)

    def set_state(self, index: int, state: bytes, stacked: bool = False) -> None:
        """Sets one environment to a state in the byte format of GRF.

        The observation and the sticky actions follow from the state. The reward is 0 and the
        done flag is clear until the next step.

        Args:
            index: The index of the environment.
            state: The bytes that `get_state` of this package or of GRF returned.
            stacked: Whether the state holds GRF's `FrameStack`, and its four frames become the
                observation stack. It needs numpy, and a batch with `stacked=True`.

        Raises:
            ArgumentError: If the index is out of range. It is a `ValueError` and an
                `IndexError`.
            StateError: If the bytes hold no pickle of the episode logic or do not load, the state
                belongs to another scenario or to other difficulties, or `stacked` differs from
                the batch's. It is a `ValueError` and a `RuntimeError`.
        """
        self._batch.set_state(index, state, stacked)

    def raw(self) -> dict[str, torch.Tensor]:
        """Returns GRF's raw observation of every agent after the last `reset` or `step`.

        The tensors view buffers of the batch, which the next call to `reset` or `step`
        overwrites.

        Raises:
            LibraryError: If the batch writes no raw observation, which needs `raw=True` or the
                `raw` representation.
        """
        if not self._raw_on:
            raise LibraryError(
                "expected a batch with raw=True or representation='raw' to return raw(), found "
                'neither'
            )
        return self._batch.raw()

    def frame(self) -> torch.Tensor:
        """Returns GRF's frame of every environment after the last `reset` or `step`.

        Returns:
            `[environment_count, height, width, 3]` `uint8`, rows from the top, as GRF's
            observation `frame` holds them.

        Raises:
            LibraryError: If the batch renders nothing, which needs `frames=True` or a `pixels`
                representation.
        """
        if not self._frames_on:
            raise LibraryError(
                "expected a batch with frames=True or representation='pixels' to return frame(), "
                'found neither'
            )
        return self._batch.frame()


BatchEnvironment.__module__ = 'cuju'
