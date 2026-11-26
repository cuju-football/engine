# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks recording from a batch."""

import numpy as np
import pytest
import torch
from helpers import SHORT, actions_for, make_batch, record_episodes, same_steps

import cuju
import cuju.recording as cr
from cuju._batch import unwrap

STEP_FIELDS = {
    'ball',
    'ball_direction',
    'ball_rotation',
    'ball_owned_team',
    'ball_owned_player',
    'left_team',
    'left_team_direction',
    'left_team_tired_factor',
    'left_team_yellow_card',
    'left_team_active',
    'left_team_roles',
    'right_team',
    'right_team_direction',
    'right_team_tired_factor',
    'right_team_yellow_card',
    'right_team_active',
    'right_team_roles',
    'score',
    'steps_left',
    'game_mode',
    'active',
    'designated',
    'sticky_actions',
}


def test_episodes_are_recorded_across_an_auto_reset(device):
    batch = make_batch(device)
    found, recorder = record_episodes(batch, [0, 2], 400)
    batch.close()
    ended = [match for match in found if match.end]
    assert len(ended) >= 4 and found[-1].end == 0 and found[-2].end == 0
    assert recorder.episodes == len(found)
    assert [match.provenance['episode'] for match in found] == list(range(len(found)))
    assert {match.provenance['environment'] for match in found} == {0, 2}
    for match in found:
        assert match.scenario == SHORT and match.left_agents == 1 and match.right_agents == 0
        assert match.action_set == 'default' and match.players == [2, 1]
        assert match.state is not None and match.engine == cuju.__version__
        assert set(match.step) == STEP_FIELDS
        assert match.actions.shape == (match.steps, 1) and match.rewards.shape == (match.steps, 1)
        assert match.step['left_team'].shape == (match.steps + 1, 2, 2)
        assert match.step['right_team'].shape == (match.steps + 1, 1, 2)
        assert match.step['sticky_actions'].shape == (match.steps + 1, 1, 10)
        assert match.step['score'].shape == (match.steps + 1, 2)
        assert match.step['steps_left'][0] > match.step['steps_left'][-1]
        assert match.done.tolist() == [False] * (match.steps - 1) + [bool(match.end)]
    # The step that ends an episode shows the first state of the next, so each episode starts at
    # its kick off and the last row of an ended one is its final state.
    assert all(match.step['steps_left'][0] == 401 for match in found)


def test_steps_false_records_the_state_and_actions_alone(device):
    batch = make_batch(device, raw=False)
    found, _ = record_episodes(batch, [1], 250, steps=False)
    batch.close()
    assert len(found) >= 2
    for match in found:
        assert match.step == {} and match.state is not None
        assert match.done.tolist() == [False] * (match.steps - 1) + [bool(match.end)]


def test_steps_true_needs_a_batch_with_raw():
    batch = make_batch(raw=False)
    with pytest.raises(cr.RecordingError, match='raw=True'):
        cr.record(batch, [0], print)
    batch.close()


def test_the_raw_representation_serves_as_well():
    batch = make_batch(representation='raw', raw=False)
    found, _ = record_episodes(batch, [0], 60)
    batch.close()
    assert found and set(found[0].step) == STEP_FIELDS


def test_the_environments_are_checked():
    batch = make_batch()
    for bad in ([], [3], [-1]):
        with pytest.raises(cr.RecordingError, match='expected environments from 0 to 2'):
            cr.record(batch, bad, print)
    with pytest.raises(cr.RecordingError, match='on_episode'):
        cr.record(batch, [0], None)
    batch.close()


def test_an_integer_names_one_environment_and_the_caller_adds_provenance():
    batch = make_batch()
    found = []
    generator = torch.Generator().manual_seed(1)
    with cr.record(
        batch, 1, found.append, provenance={'run': 'a'}, teams=['red', 'blue']
    ) as recorder:
        batch.reset()
        for _ in range(5):
            batch.step(actions_for(generator, batch))
    batch.close()
    assert recorder.environments == [1] and len(found) == 1
    assert found[0].provenance == {'run': 'a', 'environment': 1, 'episode': 0}
    assert found[0].teams == ['red', 'blue'] and found[0].steps == 5


def test_a_recorder_stops_when_closed_and_when_the_batch_closes():
    batch = make_batch()
    found = []
    recorder = cr.record(batch, [0], found.append)
    batch.reset()
    batch.step(torch.zeros(3, 1, dtype=torch.int32))
    recorder.close()
    recorder.close()
    assert len(found) == 1 and found[0].end == 0 and not unwrap(batch).observers
    batch.step(torch.zeros(3, 1, dtype=torch.int32))
    assert len(found) == 1
    with pytest.raises(cr.RecordingError, match='closed'):
        recorder.add('value', torch.zeros(3, 1))
    other = []
    cr.record(batch, [0, 1], other.append)
    batch.step(torch.zeros(3, 1, dtype=torch.int32))
    batch.close()
    assert [match.end for match in other] == [0, 0]


def test_a_recorder_attached_after_the_reset_starts_from_the_current_state():
    batch = make_batch()
    generator = torch.Generator().manual_seed(2)
    batch.reset()
    for _ in range(3):
        batch.step(actions_for(generator, batch))
    found = []
    with cr.record(batch, [0], found.append):
        for _ in range(4):
            batch.step(actions_for(generator, batch))
    batch.close()
    (match,) = found
    assert match.steps == 4 and match.step['steps_left'][0] == 398
    assert match.replay() is not None


def test_a_reset_hands_over_the_running_episode_as_unfinished():
    batch = make_batch()
    found = []
    with cr.record(batch, [0, 1], found.append):
        batch.reset()
        for _ in range(3):
            batch.step(torch.zeros(3, 1, dtype=torch.int32))
        batch.reset(mask=torch.tensor([True, False, False]))
        batch.step(torch.zeros(3, 1, dtype=torch.int32))
    batch.close()
    assert [(m.provenance['environment'], m.steps, m.end) for m in found] == [
        (0, 3, 0),
        (0, 1, 0),
        (1, 4, 0),
    ]


def test_the_values_of_add_align_with_the_observation_the_policy_saw(device):
    batch = make_batch(device, raw=True)
    found = []
    generator = torch.Generator().manual_seed(5)
    with cr.record(batch, [0, 2], found.append) as recorder:
        observation = batch.reset()
        for _ in range(300):
            recorder.add('observation', observation)
            recorder.add('count', torch.arange(3).reshape(3, 1) + 100)
            observation, *_ = batch.step(actions_for(generator, batch))
    batch.close()
    assert len(found) >= 3
    for match in found:
        rows = match.array['observation']
        assert rows.shape == (match.steps + 1, 1, 115) and rows.dtype == np.float32
        replayed = match.replay('simple115v2')
        # Rows 0 to T-1 are what the policy saw before each step. The observation after the last
        # step was never seen, so its row is filled.
        np.testing.assert_array_equal(rows[: match.steps], replayed[: match.steps])
        assert np.isnan(rows[-1]).all()
        count = match.array['count'][:, 0]
        assert (count[:-1] == 100 + match.provenance['environment']).all() and count[-1] == 0


def test_add_checks_its_values():
    batch = make_batch()
    with cr.record(batch, [0, 1], print) as recorder:
        batch.reset()
        recorder.add('value', np.zeros((2, 4)))
        with pytest.raises(cr.RecordingError, match='the first axis'):
            recorder.add('other', np.zeros((5, 1)))
        with pytest.raises(cr.RecordingError, match='keep the shape'):
            recorder.add('value', np.zeros((3, 5)))
        with pytest.raises(cr.RecordingError, match='without a slash'):
            recorder.add('a/b', np.zeros((3, 1)))
        with pytest.raises(cr.RecordingError, match='numbers'):
            recorder.add('text', np.array(['a'] * 3))
    batch.close()


def test_capture_takes_steps_that_call_no_observer():
    batch = make_batch()
    generator = torch.Generator().manual_seed(11)
    by_observer, by_hand = [], []
    recorder = cr.record(batch, [0, 2], by_hand.append)
    # A second recorder of the same batch stands in for the steps that the observers see.
    watcher = cr.record(batch, [0, 2], by_observer.append)
    unwrap(batch).remove_observer(recorder)
    mask = torch.ones(3, dtype=torch.bool)
    batch.reset()
    recorder.capture({'reset': mask})
    for _ in range(150):
        actions = actions_for(generator, batch)
        _, reward, done, _ = batch.step(actions)
        recorder.capture({'actions': actions, 'reward': reward, 'done': done})
    recorder.close()
    watcher.close()
    batch.close()
    assert len(by_hand) == len(by_observer) >= 2
    for mine, theirs in zip(by_hand, by_observer, strict=True):
        assert mine.write() == theirs.write()
    other = make_batch()
    with pytest.raises(cr.RecordingError, match='missing'):
        cr.record(other, [0], print).capture({'done': 1})
    other.close()


def test_the_time_limit_and_the_end_by_the_scenario_differ(scenario_file):
    batch = make_batch(scenario=scenario_file, environments=1)
    found, _ = record_episodes(batch, [0], 25)
    batch.close()
    assert [(match.steps, match.end) for match in found] == [(11, 2), (11, 2), (3, 0)]
    assert all(match.scenario == 'ten_steps' for match in found)
    assert all(match.step['steps_left'][-1] == 0 for match in found if match.end)
    for match in found:
        match.replay(scenario=scenario_file)
    short = make_batch(environments=1)
    ended, _ = record_episodes(short, [0], 400)
    short.close()
    assert {match.end for match in ended} == {0, 1}
    assert all(match.step['steps_left'][-1] > 0 for match in ended if match.end)


def test_agents_of_both_sides_are_recorded_in_the_view_of_the_left_team():
    batch = make_batch(scenario='academy_3_vs_1_with_keeper', left_agents=1, right_agents=1)
    found, _ = record_episodes(batch, [0], 30)
    batch.close()
    (match,) = found
    assert (match.left_agents, match.right_agents) == (1, 1) and match.actions.shape == (30, 2)
    assert match.step['sticky_actions'].shape == (31, 2, 10)
    assert match.step['active'].shape == (31, 2)
    assert match.step['left_team'].shape[1] == int(match.players[0])
    assert match.replay() is not None


def test_agents_of_the_right_alone_turn_back_to_the_left_view():
    both = make_batch(scenario='academy_3_vs_1_with_keeper', environments=1, left_agents=1)
    right = make_batch(
        scenario='academy_3_vs_1_with_keeper', environments=1, left_agents=0, right_agents=1
    )
    found, _ = record_episodes(right, [0], 20)
    right.close()
    both.close()
    (match,) = found
    assert (match.left_agents, match.right_agents) == (0, 1)
    # The pitch is the same whoever plays, so the kick off holds the same left team as a left
    # agent's batch sees.
    left = make_batch(scenario='academy_3_vs_1_with_keeper', environments=1, left_agents=1)
    kickoff, _ = record_episodes(left, [0], 1)
    left.close()
    for name in ('ball', 'left_team', 'right_team', 'score', 'ball_owned_team'):
        assert match.step[name][0].tobytes() == kickoff[0].step[name][0].tobytes(), name
    assert match.replay() is not None
    same_steps(match.with_steps().step, match.step)


def test_an_environment_without_auto_reset_waits_for_its_reset():
    batch = make_batch(environments=2, auto_reset=False)
    found = []
    generator = torch.Generator().manual_seed(4)
    with cr.record(batch, [0, 1], found.append):
        batch.reset()
        done = torch.zeros(2, dtype=torch.bool)
        for _ in range(400):
            _, _, done, _ = batch.step(actions_for(generator, batch))
            if done.all():
                break
        assert done.all()
        # An environment that ended plays nothing at the steps that follow, and its recording
        # ends at the step that ended it.
        batch.step(actions_for(generator, batch))
        batch.reset_done()
        for _ in range(5):
            batch.step(actions_for(generator, batch))
    batch.close()
    assert sorted((m.provenance['environment'], m.end) for m in found) == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    for match in found:
        assert (match.steps == 5) == (match.end == 0)
        match.replay()
