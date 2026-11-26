# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks the frames of cuju.scene, that reading them changes nothing, and both backends."""

import numpy as np
import pytest
from scene_support import make_batch, play

import cuju.scene as scene
from conftest import CUDA
from cuju import errors
from cuju._batch import unwrap


def test_a_scene_holds_the_match_the_ball_and_every_player():
    batch = make_batch(environments=3)
    batch.reset()
    frames = scene.of_batch(batch)
    batch.close()
    assert frames.shape == (3,) and frames.version == scene.VERSION == 1
    assert (frames.step == -1).all() and (frames.steps_per_second == 10).all()
    assert (frames.duration == 3000).all() and (frames.score == 0).all()
    assert (frames.half == 1).all() and (frames.possession == -1).all()
    assert frames.ball_position.shape == (3, 3) and frames.ball_rotation.shape == (3, 4)
    assert frames.players.shape == (3, 25)
    assert frames.joints.shape == (3, 25, 13, 3) and frames.joint_rotations.shape == (3, 25, 13, 4)
    assert frames.root_position.shape == (3, 25, 3) and frames.cards.shape == (3, 25, 2)
    teams = frames.team[0]
    assert (teams[:5] == 0).all() and (teams[11:16] == 1).all() and (teams[22:] == 2).all()
    assert (teams[5:11] == -1).all() and (frames.slot[0][:5] == np.arange(5)).all()
    assert frames.present[0].tolist() == [1] * 5 + [0] * 6 + [1] * 5 + [0] * 6 + [1] * 3
    assert (frames.role[0][teams == 2] == -1).all() and frames.role[0][0] == 0
    assert frames.designated[0].sum() >= 1 and frames.controlled[0].sum() == 1
    assert (frames.height[0][frames.present[0] == 1] > 1.5).all()
    assert set(frames.names) >= {'step', 'ball_position', 'joints', 'animation', 'height'}
    with pytest.raises(AttributeError, match='no field'):
        _ = frames.nothing


def test_frames_index_and_stack_like_arrays():
    batch = make_batch(environments=3)
    batch.reset()
    frames = scene.of_batch(batch, [2, 0])
    single = scene.of_batch(batch, 1)
    stacked = scene.Frames.stack([frames, frames])
    batch.close()
    assert frames.shape == (2,) and single.shape == (1,) and stacked.shape == (2, 2)
    assert frames[1].ball_position.shape == (3,) and len(frames) == 2
    assert stacked[:, 0].joints.shape == (2, 25, 13, 3)
    assert stacked.joints.shape == (2, 2, 25, 13, 3)


def test_a_bad_environment_is_an_argument_error():
    batch = make_batch(environments=2)
    batch.reset()
    with pytest.raises(errors.ArgumentError):
        scene.of_batch(batch, [2])
    batch.close()
    with pytest.raises(errors.ClosedError):
        scene.of_batch(batch)


def test_reading_the_scene_changes_no_output_and_no_state():
    plain, followed = make_batch(), make_batch()
    for batch in (plain, followed):
        batch.reset()
    follower = scene.follow(followed)
    before = followed.get_state(0)
    for _ in range(3):
        scene.of_batch(followed)
    assert followed.get_state(0) == before
    assert play(plain, 40) == play(followed, 40)
    assert [plain.get_state(index) for index in range(3)] == [
        followed.get_state(index) for index in range(3)
    ]
    assert len(list(follower)) == 41
    follower.close()
    plain.close()
    followed.close()


def test_the_scene_does_not_depend_on_render_mode():
    plain, rendering = make_batch(), make_batch()
    unwrap(rendering)._set_render(render_mode=1, referee_animations=True)
    for batch in (plain, rendering):
        batch.reset()
        play(batch, 25)
    first, second = scene.of_batch(plain), scene.of_batch(rendering)
    plain.close()
    rendering.close()
    assert first.data.tobytes() == second.data.tobytes()


def test_the_same_match_gives_the_same_frames_on_both_backends():
    if not CUDA:
        pytest.skip('no CUDA device or library')
    found = {}
    for device in ('cpu', 'cuda'):
        batch = make_batch(device, '11_vs_11_stochastic')
        batch.reset()
        found[device] = []
        for _ in range(3):
            play(batch, 20)
            found[device].append(scene.of_batch(batch).data.tobytes())
        batch.close()
    assert found['cpu'] == found['cuda']


def test_the_frames_of_a_followed_step_are_those_of_the_scene_after_it():
    batch = make_batch(environments=2)
    batch.reset()
    with scene.follow(batch, [1]) as follower:
        first = next(follower)
        assert first[0].shape == (1,) and first[1] == [[]]
        play(batch, 2)
        assert [next(follower)[0].step[0] for _ in range(2)] == [0, 1]
        with pytest.raises(StopIteration):
            next(follower)
        play(batch, 1)
        assert next(follower)[0].data.tobytes() == scene.of_batch(batch, [1]).data.tobytes()
    assert not any(observer is follower for observer in unwrap(batch).observers)
    batch.close()
