# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np

from . import _views
from ._errors import RecordingError
from ._recording import Recording


def _shown(value: np.ndarray) -> str:
    if value.size <= 6:
        return repr(value.tolist())
    return f'an array of the shape {value.shape} and the type {value.dtype}'


def _differs(replayed: np.ndarray, stored: np.ndarray) -> bool:
    # Bytes tell apart the zeros of both signs and every pattern of NaN.
    return (
        replayed.dtype != stored.dtype
        or replayed.shape != stored.shape
        or replayed.tobytes() != np.ascontiguousarray(stored).tobytes()
    )


def _verify(match: 'Recording', row: dict[str, np.ndarray], index: int) -> None:
    for name, stored in match.step.items():
        if name in row and _differs(row[name], stored[index]):
            raise RecordingError(
                f'the replay differs from the recording at step {index} in step/{name}, the '
                f'replay gives {_shown(row[name])} and the recording holds {_shown(stored[index])}'
            )


def _snapshot(observation: Any) -> Any:
    if isinstance(observation, dict):
        return {name: tensor[0].cpu().numpy().copy() for name, tensor in observation.items()}
    return observation[0].cpu().numpy().copy()


def _play(
    match: 'Recording', representation: str, device: Any, scenario: Any, check: bool
) -> tuple[Any, dict[str, np.ndarray]]:
    import torch

    from .._batch import unwrap
    from .._environment import BatchEnvironment
    from ..errors import StateError

    if match.state is None:
        raise RecordingError('expected a recording with a state, found none')
    unknown = np.argwhere(match.actions < 0)
    if len(unknown):
        raise RecordingError(
            f'expected the actions of every step, found step {unknown[0][0]} without'
        )
    batch = BatchEnvironment(
        scenario or match.scenario,
        environment_count=1,
        device=device,
        representation=representation,
        raw=representation != 'raw',
        left_agents=match.left_agents,
        right_agents=match.right_agents,
        action_set=match.action_set,
        auto_reset=False,
        difficulty=None if match.difficulty is None else tuple(match.difficulty),
        rewards=match.reward_terms,
    )
    try:
        observation = batch.reset()
        try:
            batch.set_state(0, match.state)
        except StateError as error:
            raise RecordingError(
                f'expected a state that the scenario loads, found {error}'
            ) from error
        players = (int(unwrap(batch).left_players[0]), int(unwrap(batch).right_players[0]))

        def view() -> dict[str, np.ndarray]:
            raw = {name: tensor[0].cpu().numpy().copy() for name, tensor in batch.raw().items()}
            return _views.left_view(raw, match.left_agents, *players)

        observations = [_snapshot(observation)]
        rows = [view()]
        if check:
            _verify(match, rows[0], 0)
        for step in range(match.steps):
            actions = torch.as_tensor(match.actions[step].astype(np.int32)).reshape(1, -1)
            observation, _, done, _ = batch.step(actions)
            observations.append(_snapshot(observation))
            rows.append(view())
            if check:
                if bool(done[0]) != bool(match.done[step]):
                    raise RecordingError(
                        f'the replay differs from the recording at step {step + 1}, the replay '
                        f'{"ends" if bool(done[0]) else "does not end"} the episode and the '
                        f'recording {"does not" if bool(done[0]) else "does"}'
                    )
                _verify(match, rows[-1], step + 1)
    finally:
        batch.close()
    if isinstance(observations[0], dict):
        stacked = {
            name: np.stack([item[name] for item in observations]) for name in observations[0]
        }
    else:
        stacked = np.stack(observations)
    return stacked, {name: np.stack([row[name] for row in rows]) for name in rows[0]}


def replay(match: 'Recording', representation: str, device: Any, scenario: Any) -> Any:
    return _play(match, representation, device, scenario, True)[0]


def with_steps(match: 'Recording', device: Any, scenario: Any) -> 'Recording':
    return match._copy(step=_play(match, 'raw', device, scenario, False)[1])
