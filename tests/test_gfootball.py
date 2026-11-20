# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Runs GRF's own usage against `cuju.gfootball`, GRF's Python package over cuju's engine."""

import glob
import os
import random

import pytest

np = pytest.importorskip('numpy')
pytest.importorskip('absl')

from cuju import _dump_to_txt as dump_to_txt  # noqa: E402
from cuju import dumps, gfootball_engine  # noqa: E402
from cuju.gfootball import _gym  # noqa: E402
from cuju.gfootball import env as football_env  # noqa: E402
from cuju.gfootball.env import script_helpers  # noqa: E402

football_action_set = football_env.football_action_set


def grf_draws(seed, count):
    """Returns the first `count` seeds that GRF draws after `random.seed(seed)`."""
    generator = random.Random(seed)
    return [generator.randint(0, 2000000000) for _ in range(count)]


def test_grf_example_runs():
    # GRF's example of its README, with fewer steps.
    env = football_env.create_environment(
        env_name='academy_empty_goal_close',
        stacked=False,
        logdir='',
        write_goal_dumps=False,
        write_full_episode_dumps=False,
        render=False,
    )
    env.reset()
    for _ in range(20):
        obs, _, done, info = env.step(env.action_space.sample())
        if done:
            break
    assert obs.shape == (72, 96, 4) and isinstance(done, bool) and 'score_reward' in info
    env.close()


def test_grf_internals_are_cujus_engine():
    random.seed(9)
    env = football_env.create_environment('academy_empty_goal_close', representation='simple115v2')
    draws = grf_draws(9, 4)
    core = env.unwrapped._env
    assert isinstance(core._env, gfootball_engine.GameEnv)
    cfg = env.unwrapped._config
    # GRF: the creation leaves episode 1 with the third draw, and the second sets the team order.
    assert cfg['episode_number'] == 1 and cfg['game_engine_random_seed'] == draws[2]
    assert cfg['reverse_team_processing'] == bool(draws[1] % 2)
    assert core._env.config.game_engine_random_seed == draws[2]
    env.reset()
    assert cfg['episode_number'] == 2 and cfg['game_engine_random_seed'] == draws[3]
    env.close()


def test_grf_ids_make_grf_environments():
    name = 'GFootball-academy_empty_goal_close-simple115v2-v0'
    spec = _gym.spec(name)
    assert spec.entry_point == 'cuju.gfootball.env:create_environment'
    env = _gym.make(name)
    assert env.reset().shape == (115,)
    assert len(env.step(0)) == 4
    env.close()


def test_config_follows_grf():
    config = football_env.config
    random.seed(7)
    values = {
        'level': 'academy_empty_goal_close',
        'players': ['agent:left_players=2', 'bot:right_players=1'],
    }
    cfg = config.Config(values)
    draws = grf_draws(7, 2)
    # GRF: creating the configuration builds the scenario of episode 1, which draws once.
    assert cfg['game_engine_random_seed'] == draws[0]
    assert random.randint(0, 2000000000) == draws[1]
    assert cfg['episode_number'] == 1
    assert cfg['reverse_team_processing'] == bool(draws[0] % 2)
    scenario = cfg.ScenarioConfig()
    assert isinstance(scenario, gfootball_engine.ScenarioConfig)
    assert (scenario.left_agents, scenario.right_agents) == (2, 1)


@pytest.mark.parametrize('scenario', ['academy_3_vs_1_with_keeper', '5_vs_5'])
def test_conversions_of_raw_observations_equal_the_representations(scenario):
    trajectories = {}
    for representation in ('raw', 'simple115', 'simple115v2', 'extracted'):
        random.seed(13)
        env = football_env.create_environment(scenario, representation=representation)
        observations = [env.reset()]
        for step in range(30):
            observation, _, done, _ = env.step((step * 7) % 19)
            observations.append(observation)
            if done:
                break
        trajectories[representation] = observations
        env.close()
    convert = football_env.wrappers.Simple115StateWrapper.convert_observation
    generate_smm = football_env.observation_preprocessing.generate_smm
    for index, raw in enumerate(trajectories['raw']):
        simple115 = convert(raw, False)
        assert simple115.dtype == np.float32 and simple115.shape == (1, 115)
        assert simple115[0].tobytes() == trajectories['simple115'][index].tobytes()
        assert convert(raw, True)[0].tobytes() == trajectories['simple115v2'][index].tobytes()
        smm = generate_smm(raw)
        assert smm.dtype == np.uint8 and smm.shape == (1, 72, 96, 4)
        assert smm[0].tobytes() == trajectories['extracted'][index].tobytes()


def test_dump_reads_and_prints_as_grf(tmp_path):
    random.seed(23)
    env = football_env.create_environment(
        'academy_empty_goal_close',
        representation='simple115v2',
        write_full_episode_dumps=True,
        logdir=str(tmp_path),
    )
    env.reset()
    for step in range(10):
        _, _, done, _ = env.step(football_action_set.action_right if step == 0 else 5)
        if done:
            break
    env.close()
    paths = sorted(glob.glob(os.path.join(str(tmp_path), '*.dump')))
    assert len(paths) == 1
    steps = script_helpers.ScriptHelpers().load_dump(paths[0])
    assert steps[0]['debug']['action'] == [football_action_set.action_right]
    # cuju's reader takes the dump that GRF's Python wrote, and runs no code from it.
    assert len(dumps.read_dump(paths[0])) == len(steps)
    output = tmp_path / 'dump.txt'
    dump_to_txt.main(['--trace_file', paths[0], '--output', str(output)])
    text = output.read_text()
    assert "'action': [right]" in text and 'array([' in text
