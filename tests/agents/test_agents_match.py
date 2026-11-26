# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks `Match`, `play`, the result summary and the Gymnasium adapter."""

import numpy as np
import pytest
import torch

import cuju
from cuju import agents
from cuju.agents import standard
from cuju.errors import ConfigurationError


def make(**options):
    values = dict(
        scenario='5_vs_5',
        environment_count=3,
        device='cpu',
        left_agents=1,
        right_agents=1,
        representation='raw',
        raw=True,
        auto_reset=False,
    )
    values.update(options)
    return cuju.BatchEnvironment(**values)


def zeros(controller_agents=1):
    return lambda o: torch.zeros(o.active.shape, dtype=torch.int64)


def test_a_match_with_two_controllers_runs_and_summarizes():
    batch = make()
    with agents.Match(batch, standard.Function(zeros()), standard.Random()) as match:
        result = match.run(episodes=4)
    batch.close()
    assert result.error is None
    assert result.score.shape == (4, 2) and (result.steps == 3001).all()
    summary = result.summary()
    assert summary['wins'] + summary['draws'] + summary['losses'] == 4
    low, high = summary['win_rate_interval']
    assert 0 <= low <= summary['win_rate'] <= high <= 1


def test_the_caller_keeps_the_sides_it_names_and_counts_episodes():
    batch = make(auto_reset=True, environment_count=2)
    with agents.Match(batch, right=standard.Random()) as match:
        observation = match.reset()
        assert observation['ball'].shape == (2, 1, 3)
        for _ in range(3001):
            observation, reward, done, _ = match.step(torch.zeros(2, 1, dtype=torch.int32))
        assert reward.shape == (2, 1) and bool(done.all())
        assert match.episodes.count == 2 and match.episodes.mean_length == 3001
        with pytest.raises(ConfigurationError, match='found none'):
            match.step()
        with pytest.raises(ConfigurationError, match='every side'):
            agents.Match(batch, standard.Random(), standard.Random()).step(torch.zeros(2, 1))
    batch.close()


def test_a_mismatch_of_agents_and_actions_is_refused_naming_both():
    batch = make(left_agents=2)
    with pytest.raises(ConfigurationError, match=r'expected 3 agents.*2'):
        agents.Match(batch, standard.Random(), standard.Random(), control={'left': {'players': 3}})
    with pytest.raises(ConfigurationError, match='action_set'):
        agents.Match(
            batch, standard.Random(), standard.Random(), control={'left': {'action_set': 'full'}}
        )
    plain = make(representation='simple115v2', raw=False)
    with pytest.raises(ConfigurationError, match='raw=True'):
        agents.Match(plain, standard.Random(), standard.Random())
    batch.close()
    plain.close()


def test_a_breach_stops_the_run_and_names_side_step_and_environment():
    batch = make()
    bad = standard.Function(lambda o: torch.full(o.active.shape, 99))
    with agents.Match(batch, bad, standard.Random()) as match:
        result = match.run(1)
        assert isinstance(result.error, agents.ControllerError)
        assert (result.error.side, result.error.step, result.error.reason) == (0, 0, 'range')
        with pytest.raises(agents.ControllerError, match='side left'):
            match.reset()
            match.step()
    batch.close()


def test_negative_actions_count_from_the_end_of_the_controllers_set():
    batch = make(environment_count=1)
    seen = []

    class Spy(standard.Function):
        def act(self, observation):
            return super().act(observation)

    controller = Spy(lambda o: torch.full(o.active.shape, -1))
    with agents.Match(batch, controller, standard.Random()) as match:
        match.reset()
        match.step()
        seen.append(match._applied[0, 0].item())
    batch.close()
    assert seen == [18]


def test_a_fixed_assignment_hands_the_rest_to_the_built_in_ai():
    batch = make(left_agents=4, right_agents=0, action_set='v2', environment_count=2)
    calls = []

    def play_selected(o):
        calls.append(tuple(o.active.shape))
        return torch.zeros(o.active.shape, dtype=torch.int64)

    controller = standard.Function(play_selected, control={'assignment': 'fixed', 'selection': [1, 2]})
    with agents.Match(batch, controller, 'builtin') as match:
        match.reset()
        for _ in range(5):
            match.step()
        applied = match._applied
    batch.close()
    assert calls[0] == (2, 2)
    assert applied[:, 2:].eq(19).all() and applied[:, :2].eq(0).all()


def test_fixed_assignment_needs_the_v2_set():
    batch = make(left_agents=4, right_agents=0)
    controller = standard.Random(control={'assignment': 'fixed', 'selection': [1]})
    with pytest.raises(ConfigurationError, match="action_set='v2'"):
        agents.Match(batch, controller, 'builtin')
    batch.close()


def test_a_swap_takes_effect_at_the_next_episode_and_keeps_callers_objects_open():
    batch = make(auto_reset=True, environment_count=1)
    closed = []

    class Probe(standard.Function):
        def close(self):
            closed.append(True)

    first = Probe(lambda o: torch.zeros(o.active.shape, dtype=torch.int64))
    second = standard.Function(lambda o: torch.full(o.active.shape, 3, dtype=torch.int64))
    with agents.Match(batch, first, standard.Random()) as match:
        match.reset()
        match.step()
        match.set(left=second)
        match.step()
        assert match._applied[0, 0].item() == 0
        for _ in range(3000):
            match.step()
        assert match._applied[0, 0].item() == 3
    batch.close()
    assert closed == []


def test_async_steps_equal_sync_steps():
    results = []
    for use_async in (False, True):
        batch = make(environment_count=1)
        with agents.Match(batch, standard.Function(zeros()), standard.Random(), seed=4) as match:
            match.reset()
            for _ in range(10):
                if use_async:
                    match.step_async()
                    match.step_wait()
                else:
                    match.step()
            results.append(match._applied.clone())
        batch.close()
    assert results[0].equal(results[1])


def test_play_and_play_batch_return_results():
    result = agents.play('random', 'random', scenario='5_vs_5', episodes=2)
    assert result.episodes == 2 and result.error is None
    again = agents.play_batch('random', 'random', environments=2, scenario='5_vs_5')
    assert again.episodes == 2


def test_gymnasium_adapter_resets_in_the_same_step():
    pytest.importorskip('gymnasium')
    batch = make(
        left_agents=2, right_agents=0, representation='simple115v2', auto_reset=True, environment_count=2
    )
    with agents.Match(batch, right='builtin') as match:
        env = agents.to_gymnasium(match)
        observation, _ = env.reset(seed=1)
        assert tuple(observation.shape) == (2, 2, 115)
        for _ in range(3001):
            observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert bool(truncated.all()) and bool(info['_final_observation'].all())
        assert tuple(info['final_observation'].shape) == (2, 2, 115)
    batch.close()
    assert np.isfinite(reward.numpy()).all()
