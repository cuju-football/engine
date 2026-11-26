# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from typing import Any

import numpy as np

PER_AGENT = ('active', 'designated', 'sticky_actions')
_TEAM_TYPES = {
    'team': np.float32,
    'team_direction': np.float32,
    'team_tired_factor': np.float32,
    'team_yellow_card': np.bool_,
    'team_active': np.bool_,
    'team_roles': np.int32,
}
_TYPES = {
    'ball': np.float32,
    'ball_direction': np.float32,
    'ball_rotation': np.float32,
    'ball_owned_team': np.int32,
    'ball_owned_player': np.int32,
    'score': np.int32,
    'steps_left': np.int32,
    'game_mode': np.int32,
    'active': np.int32,
    'designated': np.int32,
    'sticky_actions': np.uint8,
}


def raw_type(name: str) -> Any:
    if name in _TYPES:
        return _TYPES[name]
    return _TEAM_TYPES[name.partition('_')[2]]


def left_view(
    raw: Mapping[str, np.ndarray], left_agents: int, left_players: int, right_players: int
) -> dict[str, np.ndarray]:
    # A right agent sees the pitch from its own end, so a batch of right agents alone turns back
    # by the half turn that GRF applies, with the teams swapped.
    flip = left_agents == 0
    view: dict[str, np.ndarray] = {}
    for name, value in raw.items():
        if name in PER_AGENT:
            view[name] = np.asarray(value)
        elif name.startswith(('left_', 'right_')):
            side, _, rest = name.partition('_')
            target = {'left': 'right', 'right': 'left'}[side] if flip else side
            data = np.asarray(value[0])[: left_players if target == 'left' else right_players]
            if flip and rest in ('team', 'team_direction'):
                data = -data
            view[f'{target}_{rest}'] = data
        else:
            data = np.asarray(value[0])
            if flip and name in ('ball', 'ball_direction'):
                data = data.copy()
                data[:2] = -data[:2]
            elif flip and name == 'ball_owned_team':
                data = np.where(data >= 0, 1 - data, data).astype(data.dtype)
            elif flip and name == 'score':
                data = data[::-1].copy()
            view[name] = data
    return view
