# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks `cuju.recording.load`, which joins recordings into arrays."""

import numpy as np
import pytest

import cuju.recording as cr


def make(path, steps, value):
    actions = np.full((steps, 2), value)
    match = cr.Recording(
        '5_vs_5',
        1,
        actions,
        rewards=np.full((steps, 2), float(value)),
        right_agents=1,
        end=1,
        step={'extra': np.arange(steps + 1)[:, None] * value},
    )
    match.write(path)
    return match


def test_load_joins_episodes_and_marks_their_starts(tmp_path):
    make(tmp_path / 'a.cjr', 3, 1)
    make(tmp_path / 'b.cjr', 5, 2)
    found = cr.load([tmp_path / 'a.cjr', tmp_path / 'b.cjr'])
    assert set(found) == {'actions', 'rewards', 'done', 'episode_start'}
    assert found['episode_start'].tolist() == [0, 3, 8]
    assert found['actions'].shape == (8, 2) and found['actions'][3:].min() == 2
    assert found['done'].tolist() == [False, False, True] + [False] * 4 + [True]
    assert cr.load(tmp_path)['episode_start'].tolist() == [0, 3, 8]


def test_load_decodes_only_the_fields_asked_for(tmp_path):
    make(tmp_path / 'a.cjr', 3, 1)
    make(tmp_path / 'b.cjr', 2, 3)
    found = cr.load(tmp_path, ['actions', 'step/extra'])
    assert set(found) == {'actions', 'step/extra', 'episode_start', 'table_start'}
    assert found['table_start'].tolist() == [0, 4, 7]
    assert found['step/extra'].shape == (7, 1)
    assert found['step/extra'][4:, 0].tolist() == [0, 3, 6]


def test_load_refuses_what_it_cannot_join(tmp_path):
    make(tmp_path / 'a.cjr', 3, 1)
    with pytest.raises(cr.RecordingError, match='nothing'):
        cr.load(tmp_path, ['step/nothing'])
    with pytest.raises(cr.RecordingError, match='field'):
        cr.load(tmp_path, ['state'])
    with pytest.raises(cr.RecordingError, match='at least one'):
        cr.load([])
