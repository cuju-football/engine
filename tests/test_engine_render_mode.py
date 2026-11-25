# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks GRF's render mode in cuju's engines, the matches a rendering GRF plays, and RenderState.

The digests are of matches that equal the clean GRF build's rendered matches outside GRF's padding.
"""

import ctypes
import hashlib

import pytest

import cuju.gfootball_engine._render_hooks as hooks
from conftest import CUDA
from cuju._batch import Batch
from cuju.gfootball_engine import _gameplayfootball, _native


class _Session:
    """A renderer that draws nothing, so render mode plays as GRF's rendering engine does."""

    def __init__(self, env):
        self.env = env

    def reset(self, env, animations):
        pass

    def render(self, env, swap_buffer):
        library = _native.library()
        state = ctypes.create_string_buffer(library.cuju_engine_render_state_size())
        processed = ctypes.c_uint()
        assert (
            library.cuju_engine_render_state(
                env._engine, state, len(state.raw), ctypes.byref(processed)
            )
            == 0
        )
        _RENDERS.append((state.raw, processed.value))

    def frame(self):
        config = self.env.game_config
        return bytes(config.render_resolution_x * config.render_resolution_y * 3)


class _Provider:
    @staticmethod
    def open_session(env):
        return _Session(env)


# The RenderState and the count of saves and loads at each render. GRF keeps one rendering engine
# per process and hands it to the next environment, so the renders gather here.
_RENDERS = []


@pytest.fixture
def renderer(monkeypatch):
    from cuju.gfootball.env import football_env_core

    _RENDERS.clear()
    monkeypatch.setattr(hooks, 'provider', lambda: _Provider)
    # GRF keeps a released rendering engine for the next environment of the process, which would
    # hold another test's renderer, so each test starts a fresh one.
    monkeypatch.setattr(football_env_core, '_unused_rendering_engine', None)
    monkeypatch.setattr(football_env_core, '_active_rendering', False)
    return _RENDERS


def match_digest(state):
    """The SHA-256 of a state's match, its bytes after the length and the pickle."""
    return hashlib.sha256(state[4 + int.from_bytes(state[:4], 'little') :]).hexdigest()[:16]


def play(scenario, seed, steps, mode, saves=True):
    """Plays GRF's episode through `cuju.gfootball` as GRF's harness does and returns its states."""
    from cuju.gfootball.env import config, football_env

    values = {
        'level': scenario,
        'players': [],
        'game_engine_random_seed': seed,
        'render_resolution_x': 64,
        'render_resolution_y': 36,
    }
    env = football_env.FootballEnv(config.Config(values))
    switch = int(mode[4:]) if mode.startswith('mid:') else None
    if mode == 'switch':
        env.render()
    env.reset()
    states = [env.get_state()] if saves else []
    for step in range(steps):
        if step == switch:
            env.render()
            if saves:
                states.append(env.get_state())
        done = env.step([])[2]
        if saves:
            states.append(env.get_state())
        if done:
            break
    if not saves:
        states.append(env.get_state())
    # GRF: one rendering environment per process.
    env.close()
    return states


def test_render_mode_plays_the_corner_of_a_rendering_grf(renderer):
    # GRF: the stale root of the taker after the reset of the set piece moves the ball.
    states = play('academy_corner', 0, 60, 'switch')
    assert len(states) == 33
    assert match_digest(states[-1]) == '2723d171d84c551b'


def test_without_render_mode_the_corner_differs(renderer, monkeypatch):
    def plain(self):
        _native.library().cuju_engine_set_render(self._engine, 0)
        self._render_mode = False

    monkeypatch.setattr(_gameplayfootball.GameEnv, '_sync_render', plain)
    states = play('academy_corner', 0, 60, 'switch')
    assert match_digest(states[-1]) == '352a4d98fba75eec'


def test_the_render_switch_loads_as_grf_does(renderer):
    # GRF: the switch loads the state into a fresh engine, which mirrors the mental images' ball
    # predictions once more under `reverse_team_processing`.
    states = play('11_vs_11_stochastic', 3, 40, 'mid:20', saves=False)
    assert match_digest(states[-1]) == '71ddab669a8bb93e'


def test_render_state_holds_the_goal_message(renderer):
    play('academy_pass_and_shoot_with_keeper', 0, 40, 'switch', saves=False)
    library = _native.library()
    assert library.cuju_engine_render_state_version() == 1
    last, _ = renderer[-1]
    assert len(last) == library.cuju_engine_render_state_size()
    assert b'GOAL for Frequentists United! Turing scores!\0' in last
    first, _ = renderer[0]
    assert b'GOAL' not in first


def overlay(state):
    """The scoreboard's goals, the message's visibility and its text in a RenderState."""
    import re

    clocks = [m.start() for m in re.finditer(rb'\d\d:\d\d\0', state)]
    assert len(clocks) == 1
    at = clocks[0] + 16
    goals = tuple(int.from_bytes(state[at + k : at + k + 4], 'little') for k in (0, 4))
    text = state[at + 12 : at + 12 + 128].split(b'\0')[0]
    return goals, bool(state[at + 8]), text


@pytest.mark.parametrize('disabled', [False, True])
def test_a_goal_without_rendering_leaves_the_captions(renderer, disabled):
    # GRF: a caption takes new text only while rendering, so a goal scored after `disable_render`
    # leaves the scoreboard and the message caption as they were, while the message shows.
    from cuju.gfootball.env import config, football_env

    values = {
        'level': 'academy_pass_and_shoot_with_keeper',
        'players': [],
        'game_engine_random_seed': 0,
        'render_resolution_x': 64,
        'render_resolution_y': 36,
    }
    env = football_env.FootballEnv(config.Config(values))
    env.render()
    env.reset()
    if disabled:
        env.disable_render()
    for _ in range(40):
        if env.step([])[2]:
            break
    env.render()
    env.close()
    goals, visible, text = overlay(renderer[-1][0])
    assert visible
    if disabled:
        assert (goals, text) == ((0, 0), b'')
    else:
        assert (goals, text) == ((1, 0), b'GOAL for Frequentists United! Turing scores!')


def test_saves_and_loads_count_for_the_nets(renderer):
    play('academy_empty_goal_close', 0, 3, 'switch')
    counts = [processed for _, processed in renderer]
    # A save follows the reset and each step, and a render ends each step.
    assert counts[-3:] == [counts[-3], counts[-3] + 1, counts[-3] + 2]


def batch_run(device, steps=40):
    import torch

    batch = Batch('5_vs_5_l1_r0', environment_count=3, device=device, seed=7, representation='raw')
    batch._set_render(render_mode=1, referee_animations=True)
    batch.reset()
    actions = torch.zeros((3, batch.agents), dtype=torch.int32, device=batch.torch_device)
    states = []
    for _ in range(steps):
        batch.step(actions)
        states.append(batch._render_states([2, 0]))
    saved = [batch.get_state(index) for index in range(3)]
    after = batch._render_states([0])
    batch.close()
    return states, saved, after


def test_batch_render_states(renderer):
    states, _, after = batch_run('cpu')
    raw, processed = states[-1]
    size = _native.library().cuju_engine_render_state_size()
    assert len(raw) == 2 * size and list(processed) == [0, 0]
    assert list(after[1]) == [1]


@pytest.mark.skipif(not CUDA, reason='no CUDA device or library')
def test_both_backends_give_the_same_render_states(renderer):
    cpu, cuda = batch_run('cpu'), batch_run('cuda')
    assert cpu[0] == cuda[0]
    assert [match_digest(state) for state in cpu[1]] == [match_digest(state) for state in cuda[1]]
