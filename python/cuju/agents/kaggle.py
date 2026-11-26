# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""Kaggle's observation and answers for a batch, which the `kaggle` controller builds on."""

from typing import Any

import numpy as np

from .._batch import unwrap
from .._raw import grf_observation, team_sizes
from .client import (
    OVERAGE,
    Struct,
    accepts_configuration,
    action_index,
    action_indices,
)

__all__ = [
    'Struct',
    'accepts_configuration',
    'action_index',
    'action_indices',
    'observation',
    'players_raw',
]


def _lists(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value.tolist() if isinstance(value, np.ndarray) else value
        for name, value in entry.items()
    }


def players_raw(
    batch: Any, environment: int, agent: int, final: bool = False
) -> list[dict[str, Any]]:
    """Returns `obs['players_raw']` of a Kaggle agent that plays an agent of a batch."""
    batch = unwrap(batch)
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
) -> Struct:
    """Returns a Kaggle agent's observation, a dictionary whose keys are also attributes."""
    batch = unwrap(batch)
    agents = [int(index) for index in agent] if isinstance(agent, (list, tuple)) else [int(agent)]
    if raw is None:
        raw = batch.final_raw() if final else batch.raw()
    return Struct(
        players_raw=[
            _lists(grf_observation(raw, environment, index, *team_sizes(batch, environment, index)))
            for index in agents
        ],
        controlled_players=len(agents),
        step=step,
        remainingOverageTime=OVERAGE,
    )
