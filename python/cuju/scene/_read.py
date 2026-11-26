# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

import collections
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import ArgumentError
from ._events import Event, between
from ._frames import Frames, check_library, environments_of


def of_batch(batch: Any, environments: Iterable[int] | int | None = None) -> Frames:
    """Returns the scene of environments of a batch as it is now.

    The scene is read from the worlds without changing them, only when this is called.

    Args:
        batch: A `BatchEnvironment`.
        environments: The indices to read, one index, or `None` for every environment.

    Returns:
        The `Frames` of the chosen environments, in the order given, one frame for each.
    """
    from .._batch import unwrap

    inner = unwrap(batch)
    chosen = environments_of(inner, environments)
    team_players = check_library(inner._library)
    return Frames.from_bytes(inner._scene_states(chosen), team_players, len(chosen))


class Follower:
    """Follows environments of a batch and keeps the scene after each of its steps.

    Iterating gives `(frames, events)` first for the scene at the start, with no events, and then
    for each step or reset of the batch since the last item, so a loop may step the batch inside
    its body or the batch may run first. The iteration ends when no step is waiting.

    An environment that resets itself at the end of an episode shows the first scene of the next
    episode after that step, without events, as the old world is gone then.
    """

    def __init__(self, batch: Any, environments: Iterable[int] | int | None = None) -> None:
        from .._batch import unwrap

        self._batch = unwrap(batch)
        self.environments = environments_of(self._batch, environments)
        self._last = of_batch(self._batch, self.environments)
        self._queue: collections.deque[tuple[Frames, list[list[Event]]]] = collections.deque(
            [(self._last, [[] for _ in self.environments])]
        )
        self._closed = False
        self._batch.add_observer(self)

    def __call__(self, batch: Any, done: Any, information: dict[str, Any]) -> None:
        if self._closed:
            return
        if information.get('close'):
            self._closed = True
            return
        frames = of_batch(batch, self.environments)
        self._queue.append((frames, between(self._last, frames)))
        self._last = frames

    def __iter__(self) -> 'Follower':
        return self

    def __next__(self) -> tuple[Frames, list[list[Event]]]:
        if not self._queue:
            raise StopIteration
        return self._queue.popleft()

    def __enter__(self) -> 'Follower':
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()

    def close(self) -> None:
        """Stops following the batch."""
        if self._closed:
            return
        self._closed = True
        if any(observer is self for observer in self._batch.observers):
            self._batch.remove_observer(self)


def follow(batch: Any, environments: Iterable[int] | int | None = None) -> Follower:
    """Follows environments of a batch, for the scene and the events after each step.

    Args:
        batch: A `BatchEnvironment`.
        environments: The indices to follow, one index, or `None` for every environment.

    Returns:
        A `Follower`, which iterates over `(frames, events)`, where `frames` holds one frame for
        each followed environment and `events` lists the events of each, since the step before.
    """
    return Follower(batch, environments)


def of_recording(
    match: Any, device: Any = 'cpu', scenario: Any = None
) -> tuple[Frames, list[list[Event]]]:
    """Returns the scene of every step of a recording, by replaying its state and actions.

    Args:
        match: A `cuju.recording.Recording` with a state and the actions of every step, or the path
            of a file that holds one.
        device: The device that replays the match.
        scenario: A scenario to replay on in place of the recording's own.

    Returns:
        The `Frames` of the start and of each step, shaped `[steps + 1, 1]`, and a list of one list
        of the events of the match.
    """
    import torch

    from .. import recording
    from .._batch import unwrap
    from .._environment import BatchEnvironment

    if isinstance(match, (str, Path)):
        match = recording.read(match)
    if match.state is None:
        raise ArgumentError('expected a recording with a state, found none')
    if bool((match.actions < 0).any()):
        raise ArgumentError('expected the actions of every step, found a step without')
    batch = BatchEnvironment(
        scenario or match.scenario,
        environment_count=1,
        device=device,
        representation='raw',
        left_agents=match.left_agents,
        right_agents=match.right_agents,
        action_set=match.action_set,
        auto_reset=False,
        difficulty=None if match.difficulty is None else tuple(match.difficulty),
        rewards=match.reward_terms,
    )
    try:
        batch.reset()
        batch.set_state(0, match.state)
        inner = unwrap(batch)
        frames = [of_batch(inner, 0)]
        events: list[Event] = []
        for step in range(match.steps):
            actions = torch.as_tensor(match.actions[step].astype(np.int32)).reshape(1, -1)
            batch.step(actions)
            frames.append(of_batch(inner, 0))
            events.extend(between(frames[-2], frames[-1])[0])
    finally:
        batch.close()
    return Frames.stack(frames), [events]
