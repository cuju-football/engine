# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks replay and the step fields that it fills."""

import numpy as np
import pytest
from helpers import SHORT, make_batch, record_episodes, same_steps

import cuju.recording as cr


@pytest.fixture(scope='module')
def match():
    batch = make_batch(environments=1)
    found, _ = record_episodes(batch, [0], 400)
    batch.close()
    return max((m for m in found if m.end), key=lambda m: m.steps)


def test_a_replay_gives_the_observations_in_any_representation(match):
    simple = match.replay()
    assert simple.shape == (match.steps + 1, 1, 115) and simple.dtype == np.float32
    raw = match.replay('raw')
    assert raw['left_team'].shape == (match.steps + 1, 1, 2, 2)
    assert raw['left_team'][:, 0].shape == match.step['left_team'].shape
    assert raw['left_team'][:, 0].tobytes() == match.step['left_team'].tobytes()
    assert match.replay('extracted').shape[:2] == (match.steps + 1, 1)


def test_a_replay_on_either_backend_equals_the_stored_steps(match, device):
    assert match.replay(device=device).shape[0] == match.steps + 1


def test_a_replay_survives_a_file(match, tmp_path):
    match.write(tmp_path / 'match.cjr')
    cr.read(tmp_path / 'match.cjr').replay()


def test_a_tampered_action_fails_the_replay(match):
    tampered = cr.read(match.write())
    tampered.actions[2:] = 5
    with pytest.raises(cr.RecordingError, match=r'differs from the recording at step \d+ in step/'):
        tampered.replay()


def test_a_tampered_step_names_its_row_and_field(match):
    tampered = cr.read(match.write())
    tampered.step['ball'][4, 0] += np.float32(2**-10)
    with pytest.raises(cr.RecordingError, match='at step 4 in step/ball'):
        tampered.replay()
    tampered = cr.read(match.write())
    tampered.step['score'][0, 1] = 3
    with pytest.raises(cr.RecordingError, match='at step 0 in step/score'):
        tampered.replay()


def test_the_end_of_the_episode_is_checked(match):
    early = cr.read(match.write())
    early.done[:] = False
    with pytest.raises(
        cr.RecordingError, match='the replay ends the episode and the recording does'
    ):
        early.replay()
    late = cr.read(match.write())
    late.done[-2] = True
    late.done[-1] = False
    with pytest.raises(cr.RecordingError, match='does not end the episode and the recording does'):
        late.replay()


def test_a_replay_needs_a_state_and_every_action(match):
    bare = cr.Recording(match.scenario, 1, match.actions)
    with pytest.raises(cr.RecordingError, match='with a state'):
        bare.replay()
    unknown = cr.read(match.write())
    unknown.actions[3, 0] = -1
    with pytest.raises(cr.RecordingError, match='step 3 without'):
        unknown.replay()
    with pytest.raises(cr.RecordingError, match='step 3 without'):
        unknown.with_steps()


def test_a_state_of_another_scenario_is_refused(match):
    other = cr.read(match.write())
    other.scenario = 'academy_corner'
    with pytest.raises(cr.RecordingError, match='a state that the scenario loads'):
        other.replay()


def test_with_steps_equals_a_recording_made_with_steps(match):
    batch = make_batch(environments=1, raw=False)
    found, _ = record_episodes(batch, [0], 400, steps=False)
    batch.close()
    bare = max((m for m in found if m.end), key=lambda m: m.steps)
    assert bare.step == {} and bare.state == match.state
    filled = bare.with_steps()
    assert bare.step == {}
    same_steps(filled.step, match.step)
    assert filled.actions.tobytes() == match.actions.tobytes()
    assert filled.write() == match.write()


def test_with_steps_on_the_other_backend_gives_the_same_bits(match, device):
    same_steps(match.with_steps(device=device).step, match.step)


def test_a_scenario_file_replays_when_it_is_named(scenario_file):
    batch = make_batch(scenario=scenario_file, environments=1)
    found, _ = record_episodes(batch, [0], 12)
    batch.close()
    assert found[0].scenario == 'ten_steps'
    found[0].replay(scenario=scenario_file)
    assert found[0].with_steps(scenario=scenario_file).write() == found[0].write()


def test_the_scenario_of_the_header_names_the_shipped_one(match):
    assert match.scenario == SHORT and match.reward_terms == 'scoring'
