# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""Which players of a scenario a controller plays, checked against the scenario itself."""

import os
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import ConfigurationError
from ._contract import ROLES

BUILTIN_AI = 19


def scenario_path(batch: Any) -> str:
    """Returns the scenario file of a batch, a shipped scenario or the file that built it."""
    inner = batch
    for asset in getattr(inner, '_requested', []):
        text = os.fspath(asset)
        if text.endswith('.py') and os.path.isfile(text):
            return text
    name = inner.scenario_name.split(',')[0].strip()
    import cuju

    return str(Path(cuju.__file__).parent / 'data' / 'scenarios' / f'{name}.py')


def formation(path: str, side: int) -> list[tuple[int, bool]]:
    """Returns the role and whether it is controllable of each player of a side's formation."""
    from ..scenarios import Builder, load_scenario

    builder = Builder(0)
    load_scenario(path)(builder)
    config = builder.config()
    team = config.left_team if side == 0 else config.right_team
    return [(int(player.role), bool(player.controllable)) for player in team]


def controllable(players: list[tuple[int, bool]]) -> list[int]:
    """Returns the formation indices of the controllable players."""
    return [index for index, (_, able) in enumerate(players) if able]


def selected(control: dict[str, Any], players: list[tuple[int, bool]]) -> list[int]:
    """Returns the formation index of each controlled player, in the order of the agents.

    Raises:
        ConfigurationError: If the control does not fit the scenario's controllable players.
    """
    available = controllable(players)
    keeper = [index for index, (role, _) in enumerate(players) if role == 0]
    count = control['players']
    if count is not None and count > len(available):
        raise ConfigurationError(
            f'expected at most {len(available)} players, the controllable players of the '
            f'scenario, found {count}'
        )
    if control['assignment'] == 'dynamic':
        if control['goalkeeper'] == 'agent' and not any(i in available for i in keeper):
            raise ConfigurationError("expected a controllable goalkeeper for goalkeeper 'agent'")
        return available
    chosen: list[int] = []
    for item in control['selection']:
        candidates = (
            [item]
            if type(item) is int
            else [i for i in available if players[i][0] == ROLES.index(item)]
        )
        found = next((i for i in candidates if i in available and i not in chosen), None)
        if found is None:
            raise ConfigurationError(f'expected a controllable player for {item!r}, found none')
        chosen.append(found)
    owns = any(i in keeper for i in chosen)
    if control['goalkeeper'] == 'builtin' and not any(i in available for i in keeper):
        raise ConfigurationError("expected a controllable goalkeeper for goalkeeper 'builtin'")
    if control['goalkeeper'] == 'agent' and not owns:
        raise ConfigurationError("expected the goalkeeper in the selection for goalkeeper 'agent'")
    if control['goalkeeper'] in ('builtin', 'scenario') and owns:
        raise ConfigurationError(
            f"expected the goalkeeper outside the selection for goalkeeper "
            f"{control['goalkeeper']!r}"
        )
    if control['goalkeeper'] == 'scenario' and any(i in available for i in keeper):
        raise ConfigurationError(
            "expected a scenario whose goalkeeper is not controllable for goalkeeper 'scenario'"
        )
    return chosen


def agents_of(control: dict[str, Any], players: list[tuple[int, bool]]) -> int:
    """Returns the agents that a side's batch needs for the control."""
    if control['assignment'] == 'fixed':
        return len(controllable(players))
    return 1 if control['players'] is None else control['players']


def columns(
    active: np.ndarray, chosen: list[int], held: dict[int, int]
) -> tuple[list[int], list[int]] | None:
    """Finds the agents that hold the selected players, in the order of the selection.

    GRF clears the selection at a set piece, and then an agent reports no player and the result
    is `None`. A selected player that no agent reports, after a red card, keeps the agent it held
    last, or else a free agent, and that agent is in the second list. `held` remembers the agent
    of each selected player.
    """
    indices = [int(value) for value in active]
    if -1 in indices:
        return None
    by_player: dict[int, int] = {}
    for column, index in enumerate(indices):
        if index in chosen and index not in by_player:
            by_player[index] = column
    held.update(by_player)
    free = [column for column in range(len(indices)) if column not in by_player.values()]
    missing: list[int] = []
    for index in chosen:
        if index in by_player:
            continue
        column = held.get(index)
        if column not in free:
            column = free[0]
        free.remove(column)
        by_player[index] = column
        missing.append(column)
    return [by_player[index] for index in chosen], missing
