# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import numpy as np

from . import _files, _views
from ._errors import RecordingError
from ._recording import ACTION_SETS, Recording, engine_version


def _views_of(entry: Any) -> list[dict[str, Any]]:
    raw = entry['observation']['players_raw']
    if not isinstance(raw, list) or not all(isinstance(view, dict) for view in raw):
        raise TypeError('players_raw')
    return raw


def _actions(entries: list[Any], counts: tuple[int, int]) -> list[int]:
    row: list[int] = []
    for entry, count in zip(entries, counts, strict=False):
        value = entry.get('action') if isinstance(entry, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = [value]
        value = value if isinstance(value, list) else []
        for index in range(count):
            item = value[index] if index < len(value) else None
            ok = isinstance(item, (int, float)) and not isinstance(item, bool) and item >= 0
            row.append(int(item) if ok and int(item) == item else -1)
    return row


def _raw(views: list[list[dict[str, Any]]]) -> dict[str, np.ndarray]:
    # The team fields come from the first agent that plays, and the agents' own from every one.
    agents = [view for side in views for view in side]
    raw: dict[str, np.ndarray] = {}
    for name in agents[0]:
        try:
            kind = _views.raw_type(name)
        except KeyError:
            continue
        if name in _views.PER_AGENT:
            raw[name] = np.asarray([view[name] for view in agents], kind)
        else:
            raw[name] = np.asarray(agents[0][name], kind)[None]
    return raw


def from_kaggle(source: Any) -> 'Recording':
    """Converts a Kaggle episode of the GRF competition, which holds no engine state."""
    try:
        document = json.loads(_files.read_bytes(source))
    except (ValueError, RecursionError) as error:
        raise RecordingError(f'expected a Kaggle episode in JSON, found {error}') from error
    try:
        return _convert(document)
    except RecordingError:
        raise
    except (KeyError, TypeError, IndexError, ValueError, AttributeError) as error:
        raise RecordingError(
            f'expected a Kaggle episode with raw observations, found {type(error).__name__} {error}'
        ) from error


def _convert(document: dict[str, Any]) -> 'Recording':
    steps = document['steps']
    if not isinstance(steps, list) or len(steps) < 2 or not all(steps):
        raise RecordingError('expected a Kaggle episode with at least two steps of agents')
    sizes = [len(_views_of(entry)) for entry in steps[0][:2]]
    counts = (sizes[0], sizes[1] if len(sizes) > 1 else 0)
    left_agents = counts[0]
    if not sum(counts):
        raise RecordingError('expected raw observations of the agents, found none')
    rows, fields = [], []
    for index, entries in enumerate(steps):
        views = [_views_of(entry) for entry in entries[:2]]
        if tuple(len(side) for side in views) != counts[: len(views)]:
            raise RecordingError(f'expected the same agents at every step, found others at {index}')
        raw = _raw(views)
        base = views[0][0] if left_agents else views[1][0]
        own, other = len(base['left_team']), len(base['right_team'])
        players = (own, other) if left_agents else (other, own)
        fields.append(_views.left_view(raw, left_agents, *players))
        if index:
            rows.append(_actions(entries, counts))
    actions = np.asarray(rows)
    largest = int(actions.max())
    action_set = next((name for name, size in ACTION_SETS.items() if largest < size), None)
    if action_set is None:
        raise RecordingError(f'expected actions of the sets of GRF, found {largest}')
    last = steps[-1]
    statuses = [entry.get('status') for entry in last if isinstance(entry, dict)]
    ended = bool(statuses) and all(isinstance(s, str) and s != 'ACTIVE' for s in statuses)
    end = (2 if int(fields[-1]['steps_left']) <= 0 else 1) if ended else 0
    configuration = document.get('configuration')
    scenario = (configuration or {}).get('scenario_name') or '11_vs_11_kaggle'
    names = (document.get('info') or {}).get('TeamNames')
    teams = names if isinstance(names, list) and len(names) == 2 else None
    try:
        step = {name: np.stack([row[name] for row in fields]) for name in fields[0]}
    except (KeyError, ValueError) as error:
        raise RecordingError(f'expected the same fields at every step, found {error}') from error
    return Recording(
        str(scenario),
        left_agents,
        actions,
        right_agents=counts[1],
        action_set=action_set,
        players=(step['left_team'].shape[1], step['right_team'].shape[1]),
        end=end,
        step=step,
        provenance={'source': 'kaggle'},
        teams=teams,
        engine=engine_version(),
    )
