# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks that a match followed live and the same match replayed from its recording agree."""

import numpy as np
import pytest
import torch

import cuju
import cuju.recording as cr
import cuju.scene as scene
from cuju.errors import ArgumentError


def live(device, steps=40):
    """Plays a match, and returns its frames and events as followed, and its recording."""
    batch = cuju.BatchEnvironment(
        '5_vs_5', environment_count=2, device=device, seed=3, auto_reset=False
    )
    found = []
    generator = torch.Generator().manual_seed(2)
    with cr.record(batch, [1], found.append, steps=False):
        batch.reset()
        with scene.follow(batch, [1]) as follower:
            for _ in range(steps):
                actions = torch.randint(0, 19, (2, batch.agents), generator=generator).int()
                batch.step(actions)
            items = list(follower)
    batch.close()
    frames = scene.Frames.stack([frames for frames, _ in items])
    events = [event for _, per_step in items for event in per_step[0]]
    return frames, events, found[0]


def test_a_match_followed_live_equals_its_replay_bit_for_bit(device):
    frames, events, match = live(device)
    replayed, replayed_events = scene.of_recording(match, device=device)
    assert frames.shape == replayed.shape == (41, 1)
    assert frames.data.tobytes() == replayed.data.tobytes()
    assert events == replayed_events[0]


def test_the_replay_gives_the_same_frames_on_either_backend(device):
    _, _, match = live('cpu')
    other, _ = scene.of_recording(match, device=device)
    reference, _ = scene.of_recording(match, device='cpu')
    assert other.data.tobytes() == reference.data.tobytes()


def test_a_replay_reads_a_file(tmp_path):
    frames, _, match = live('cpu', 12)
    path = tmp_path / 'match.cjr'
    match.write(path)
    replayed, _ = scene.of_recording(path)
    assert replayed.data.tobytes() == frames.data.tobytes()


def test_a_recording_without_a_state_has_no_scene():
    match = cr.Recording('5_vs_5', 1, np.zeros((3, 1), np.int32))
    with pytest.raises(ArgumentError, match='a state'):
        scene.of_recording(match)
