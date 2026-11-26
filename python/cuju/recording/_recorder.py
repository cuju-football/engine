# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

import weakref
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch

from ..errors import Error
from . import _views
from ._errors import RecordingError
from ._recording import Recording, engine_version

_ACTION_SETS = {19: 'default', 20: 'v2', 33: 'full'}


class _Track:
    def __init__(self, environment: int, position: int, left_agents: int, players: tuple[int, int]):
        self.environment = environment
        self.position = position
        self.left_agents = left_agents
        self.players = players
        self.active = False
        self.clear()

    def clear(self) -> None:
        self.actions: list[np.ndarray] = []
        self.rewards: list[np.ndarray] = []
        self.rows: list[dict[str, np.ndarray]] = []
        self.arrays: dict[str, dict[int, torch.Tensor]] = {}
        self.state: bytes | None = None
        self.left = 0
        self.difficulty: list[float] | None = None


class Recorder:
    """Follows environments of a batch and hands each finished episode to a function."""

    def __init__(
        self,
        batch: Any,
        environments: int | Sequence[int],
        on_episode: Callable[[Recording], Any],
        steps: bool = True,
        provenance: Mapping[str, Any] | None = None,
        teams: Sequence[str] | None = None,
    ) -> None:
        from .._batch import unwrap

        inner = unwrap(batch)
        if not callable(on_episode):
            raise RecordingError(f'expected a function for on_episode, found {on_episode!r}')
        chosen = [environments] if isinstance(environments, (int, np.integer)) else environments
        chosen = list(dict.fromkeys(int(index) for index in chosen))
        count = inner.environment_count
        outside = [index for index in chosen if not 0 <= index < count]
        if not chosen or outside:
            raise RecordingError(
                f'expected environments from 0 to {count - 1}, found {outside or "none"}'
            )
        if steps:
            try:
                inner.raw()
            except Exception as error:
                raise RecordingError(
                    'expected a batch with raw=True or the raw representation, or steps=False'
                ) from error
        self.environments = chosen
        self.steps = steps
        self.episodes = 0
        self._on_episode = on_episode
        self._provenance = dict(provenance or {})
        self._teams = None if teams is None else list(teams)
        self._scenario = inner.scenario_name
        self._action_set = _ACTION_SETS[inner.action_count]
        self._terms = 'scoring,checkpoints' if getattr(inner, '_checkpoints', False) else 'scoring'
        self._count = count
        self._tracks = [
            _Track(
                environment,
                position,
                int(inner.left_agents[environment]),
                (int(inner.left_players[environment]), int(inner.right_players[environment])),
            )
            for position, environment in enumerate(chosen)
        ]
        self._pending: dict[str, torch.Tensor] = {}
        self._shapes: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
        self._events: dict[torch.device, Any] = {}
        self._closed = False
        # The batch holds the recorder, so the recorder holds the batch weakly to let it free.
        self._batch = weakref.ref(inner)
        inner.add_observer(self)
        try:
            self._begin(inner, self._tracks)
        except Error:
            # A batch that was not reset yet has no state to read, and its first reset begins.
            for track in self._tracks:
                track.active = False

    def __enter__(self) -> 'Recorder':
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()

    def __call__(self, batch: Any, done: Any, information: Mapping[str, Any]) -> None:
        if self._closed:
            return
        if done is not None:
            self._step(batch, done, information)
        elif information.get('close'):
            self._end_all()
            self._closed = True
        elif 'reset' in information:
            self._reset(batch, information['reset'])

    def add(self, name: str, values: Any) -> None:
        """Attaches values for the observation that the policy saw at the current step."""
        if self._closed:
            raise RecordingError('expected an open recorder, found it closed')
        if not isinstance(name, str) or not name or '/' in name:
            raise RecordingError(f'expected an array name without a slash, found {name!r}')
        try:
            tensor = (
                values if isinstance(values, torch.Tensor) else torch.as_tensor(np.asarray(values))
            )
        except (TypeError, ValueError, RuntimeError) as error:
            raise RecordingError(f'expected numbers for {name!r}, found {error}') from error
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        if tensor.dim() < 1 or tensor.dtype not in _NUMERIC:
            raise RecordingError(f'expected numbers for {name!r} with an environment axis')
        if tensor.shape[0] == self._count:
            index = torch.as_tensor(self.environments, dtype=torch.long, device=tensor.device)
            tensor = tensor.detach()[index]
        elif tensor.shape[0] == len(self.environments):
            tensor = tensor.detach().clone()
        else:
            raise RecordingError(
                f'expected the first axis of {name!r} to be {self._count} environments or '
                f'{len(self.environments)} recorded ones, found {tuple(tensor.shape)}'
            )
        shape = (tuple(tensor.shape[1:]), tensor.dtype)
        if self._shapes.setdefault(name, shape) != shape:
            raise RecordingError(
                f'expected {name!r} to keep the shape and type {self._shapes[name]}, found {shape}'
            )
        if tensor.device.type == 'cuda':
            host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
            host.copy_(tensor, non_blocking=True)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(tensor.device))
            self._events[tensor.device] = event
            tensor = host
        self._pending[name] = tensor

    def capture(self, information: Mapping[str, Any]) -> None:
        """Takes a step or a reset that ran without calling the observers of the batch."""
        batch = self._batch()
        if batch is None:
            raise RecordingError('expected the batch of the recorder, found it gone')
        if 'reset' in information:
            self._reset(batch, information['reset'])
            return
        missing = [key for key in ('done', 'actions', 'reward') if key not in information]
        if missing:
            raise RecordingError(f'expected the outputs of a step, found {missing} missing')
        self._step(batch, information['done'], information)

    def close(self) -> None:
        """Hands the running episodes over as unfinished and stops following the batch."""
        if self._closed:
            return
        self._end_all()
        self._closed = True
        batch = self._batch()
        if batch is not None and any(observer is self for observer in batch.observers):
            batch.remove_observer(self)

    def _host(self, tensor: Any, environments: list[int]) -> np.ndarray:
        tensor = torch.as_tensor(tensor)
        index = torch.as_tensor(environments, dtype=torch.long, device=tensor.device)
        return tensor[index].cpu().numpy()

    def _raw(self, batch: Any, final: bool, environments: list[int]) -> dict[str, np.ndarray]:
        raw = batch.final_raw() if final else batch.raw()
        return {name: self._host(tensor, environments) for name, tensor in raw.items()}

    def _view(self, rows: Mapping[str, np.ndarray], k: int, track: _Track) -> dict[str, Any]:
        return _views.left_view(
            {name: array[k] for name, array in rows.items()}, track.left_agents, *track.players
        )

    def _begin(self, batch: Any, tracks: list[_Track]) -> None:
        from .. import _state

        environments = [track.environment for track in tracks]
        rows = self._raw(batch, False, environments) if self.steps else None
        for k, track in enumerate(tracks):
            track.clear()
            track.state = batch.get_state(track.environment)
            start, duration = _state.read(
                track.state, ['environment.step', 'environment.scenario.gameDuration']
            )
            track.left = int(duration) - int(start)
            track.difficulty = [float(value) for value in batch.difficulty[track.environment]]
            if self.steps:
                track.rows.append(self._view(rows, k, track))
            track.active = True

    def _step(self, batch: Any, done: Any, information: Mapping[str, Any]) -> None:
        live = [track for track in self._tracks if track.active]
        if not live:
            self._pending.clear()
            return
        environments = [track.environment for track in live]
        ended = self._host(done, environments).astype(bool)
        actions = self._host(information['actions'], environments)
        rewards = self._host(information['reward'], environments)
        rows = final = None
        if self.steps:
            rows = self._raw(batch, False, environments)
            if ended.any():
                final = self._raw(batch, True, environments)
        restart = []
        for k, track in enumerate(live):
            self._take(track)
            track.actions.append(actions[k])
            track.rewards.append(rewards[k])
            if rows is not None:
                track.rows.append(self._view(final if ended[k] else rows, k, track))
            if ended[k]:
                self._finish(track, 2 if track.left - len(track.actions) <= 0 else 1)
                if batch.auto_reset:
                    restart.append(track)
        self._pending.clear()
        if restart:
            self._begin(batch, restart)

    def _reset(self, batch: Any, mask: Any) -> None:
        self._pending.clear()
        if mask is None:
            chosen = list(self._tracks)
        else:
            flags = self._host(mask, [track.environment for track in self._tracks]).astype(bool)
            chosen = [track for track, flag in zip(self._tracks, flags, strict=True) if flag]
        if not chosen:
            return
        for track in chosen:
            if track.active:
                self._finish(track, 0)
        self._begin(batch, chosen)

    def _take(self, track: _Track) -> None:
        row = len(track.actions)
        for name, values in self._pending.items():
            track.arrays.setdefault(name, {})[row] = values[track.position]

    def _end_all(self) -> None:
        for track in self._tracks:
            if track.active:
                self._take(track)
                self._finish(track, 0)
        self._pending.clear()

    def _finish(self, track: _Track, end: int) -> None:
        track.active = False
        steps = len(track.actions)
        if not steps:
            return
        for event in self._events.values():
            event.synchronize()
        arrays = {}
        for name, rows in track.arrays.items():
            first = next(iter(rows.values())).numpy()
            fill = np.nan if first.dtype.kind == 'f' else 0
            array = np.full((steps + 1, *first.shape), fill, first.dtype)
            for row, value in rows.items():
                array[row] = value.numpy()
            arrays[name] = array
        fields = {}
        if self.steps:
            for name in track.rows[0]:
                fields[name] = np.stack([row[name] for row in track.rows])
        match = Recording(
            self._scenario,
            track.left_agents,
            np.stack(track.actions),
            np.stack(track.rewards),
            right_agents=len(track.actions[0]) - track.left_agents,
            action_set=self._action_set,
            difficulty=track.difficulty,
            players=track.players,
            end=end,
            state=track.state,
            step=fields,
            array=arrays,
            provenance={
                **self._provenance,
                'environment': track.environment,
                'episode': self.episodes,
            },
            teams=self._teams,
            reward_terms=self._terms,
            engine=engine_version(),
        )
        self.episodes += 1
        self._on_episode(match)


_NUMERIC = {
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.float32,
    torch.float64,
}


def record(
    batch: Any,
    environments: int | Sequence[int],
    on_episode: Callable[[Recording], Any],
    steps: bool = True,
    provenance: Mapping[str, Any] | None = None,
    teams: Sequence[str] | None = None,
) -> 'Recorder':
    """Watches a batch and calls the function with each finished episode of the environments."""
    return Recorder(batch, environments, on_episode, steps, provenance, teams)
