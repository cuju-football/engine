# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks the events of cuju.scene on frames set by hand, and on situations that matches play."""

import collections
import itertools

import numpy as np
import pytest
import torch

import cuju
import cuju.scene as scene
from cuju.scene._events import between
from cuju.scene._frames import frame_dtype

SHOT = 12


def frames_of(count=1):
    data = np.zeros(count, frame_dtype(11))
    data['players']['team'] = -1
    data['foul'] = -1
    data['foul'][..., 0] = 0
    data['possession'] = -1
    data['last_touch'] = -1
    data['offside_player'] = -1
    data['goal'] = -1
    data['half'] = 1
    data['duration'] = 3000
    return scene.Frames(data)


def changed(before, **fields):
    after = scene.Frames(before.data.copy())
    for name, value in fields.items():
        after.data[name] = value
    return after


def kinds(events):
    return [(event.kind, event.data) for event in events]


def test_a_goal_comes_from_the_score_with_its_team_and_scorer():
    before = frames_of()
    after = changed(before, score=[0, 1], goal=[1, 12], step=4)
    (events,) = between(before, after)
    assert kinds(events) == [('goal', {'team': 1, 'scorer': 12})]
    assert events[0].step == 4


def test_a_card_comes_from_the_count_of_the_player():
    before = frames_of()
    yellow = changed(before)
    yellow.data['players']['cards'][0, 3] = [1, 0]
    red = changed(yellow)
    red.data['players']['cards'][0, 3] = [2, 1]
    second_yellow = [('card', {'player': 3, 'kind': 1}), ('card', {'player': 3, 'kind': 2})]
    assert kinds(between(before, yellow)[0]) == [('card', {'player': 3, 'kind': 1})]
    assert kinds(between(yellow, red)[0]) == second_yellow
    assert between(red, red) == [[]]


def test_an_offside_comes_from_the_referee_calling_it():
    before = frames_of()
    after = changed(before, offside_called=1, offside_player=7, set_piece=3)
    names = [(event.kind, event.data) for event in between(before, after)[0]]
    assert names == [('restart', {'set_piece': 3}), ('offside', {'player': 7})]
    assert between(after, after) == [[]]


def test_a_foul_comes_from_the_referee_and_names_both_players():
    before = frames_of()
    after = changed(before, foul=[1, 4, 13], set_piece=3)
    names = kinds(between(before, after)[0])
    assert ('foul', {'kind': 1, 'offender': 4, 'victim': 13}) in names
    assert ('restart', {'set_piece': 3}) in names


def test_restarts_kickoffs_and_the_halves_come_from_the_state():
    before = frames_of()
    kick = changed(before, game_mode=1, set_piece=1)
    assert kinds(between(before, kick)[0]) == [
        ('kickoff', {}),
        ('restart', {'set_piece': 1}),
    ]
    corner = changed(kick, game_mode=0, set_piece=4)
    assert kinds(between(kick, corner)[0]) == [('restart', {'set_piece': 4})]
    second = changed(before, half=2)
    assert kinds(between(before, second)[0]) == [('halftime', {})]
    over = changed(before, step=3000)
    assert kinds(between(before, over)[0]) == [('fulltime', {})]


def test_two_matches_share_no_events():
    before = frames_of()
    after = changed(before, score=[1, 0], episode=1)
    assert between(before, after) == [[]]


def play_goal(device):
    batch = cuju.BatchEnvironment(
        'academy_empty_goal_close', environment_count=1, device=device, seed=3, auto_reset=False
    )
    batch.reset()
    found = []
    with scene.follow(batch) as follower:
        for _ in range(40):
            _, _, done, _ = batch.step(torch.full((1, 1), SHOT, dtype=torch.int32))
            if bool(done[0]):
                break
        found = list(follower)
    batch.close()
    return found


def test_a_shot_into_an_empty_goal_gives_a_goal_event(device):
    found = play_goal(device)
    goals = [
        (frames, event) for frames, events in found for event in events[0] if event.kind == 'goal'
    ]
    assert len(goals) == 1
    frames, event = goals[0]
    assert event.data['team'] == 0 and tuple(frames.score[0]) == (1, 0)
    assert event.step == frames.step[0]
    assert sum(len(events[0]) for _, events in found) == 1


@pytest.fixture(scope='module')
def seen():
    """Plays 11 against 11 with random actions, and keeps the frames and events of three matches."""
    batch = cuju.BatchEnvironment(
        '11_vs_11_stochastic', environment_count=32, device='cpu', seed=1, auto_reset=False
    )
    batch.reset()
    watched = [0, 10, 16]
    generator = torch.Generator().manual_seed(5)
    steps = []
    with scene.follow(batch, watched) as follower:
        for _ in range(100):
            actions = torch.randint(0, 19, (32, batch.agents), generator=generator).int()
            batch.step(actions)
        steps = list(follower)
    batch.close()
    return steps


def test_matches_play_fouls_cards_restarts_and_offsides(seen):
    counts = collections.Counter(
        event.kind for _, events in seen for per_environment in events for event in per_environment
    )
    assert counts['foul'] >= 1 and counts['card'] >= 1 and counts['restart'] >= 2
    assert counts['offside'] >= 1


def test_every_event_agrees_with_the_frames_that_it_comes_from(seen):
    for (before, _), (frames, events) in itertools.pairwise(seen):
        for index, per_environment in enumerate(events):
            now, was = frames[index], before[index]
            for event in per_environment:
                if event.kind == 'card':
                    assert (now.cards[event.data['player']] > was.cards[event.data['player']]).any()
                if event.kind == 'foul':
                    assert now.foul[0] == event.data['kind'] != 0
                    assert now.foul.tolist() != was.foul.tolist()
                if event.kind == 'offside':
                    assert now.offside_called == 1 and was.offside_called == 0
                if event.kind == 'restart':
                    assert now.set_piece == event.data['set_piece'] != 0
                assert event.step == now.step
