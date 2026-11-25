# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks `cuju.gfootball_engine` against the behaviour of GRF's compiled `_gameplayfootball`."""

import copy
import os
import pickle

import pytest

import cuju
from cuju import gfootball_engine as lg
from cuju.gfootball_engine import SignatureError
from cuju.scenarios import ScenarioError

E = lg.e_PlayerRole


def entry(x, y, role, lazy=False, controllable=True):
    return lg.FormationEntry(x, y, role, lazy, controllable)


def five_a_side(left_agents=1, right_agents=0, seed=7):
    """A scenario as GRF's `5_vs_5` builds it, before the reset scales its ball."""
    config = lg.ScenarioConfig.make()
    for team in (config.left_team, config.right_team):
        team.append(entry(-1.0, 0.0, E.e_PlayerRole_GK))
        team.append(entry(-0.5, 0.2, E.e_PlayerRole_CB))
        team.append(entry(-0.5, -0.2, E.e_PlayerRole_CB))
        team.append(entry(0.0, 0.0, E.e_PlayerRole_CM))
        team.append(entry(0.5, 0.0, E.e_PlayerRole_CF))
    config.game_duration = 3000
    config.left_agents = left_agents
    config.right_agents = right_agents
    config.game_engine_random_seed = seed
    return config


def started(config=None, animations=False):
    env = lg.GameEnv()
    env.start_game()
    env.state = lg.GameState.game_running
    env.reset(config or five_a_side(), animations)
    return env


def test_module_names():
    names = {
        'ControllerInfo',
        'ControllerInfoVec',
        'FloatVec',
        'FormationEntry',
        'FormationEntryVec',
        'GameConfig',
        'GameEnv',
        'GameState',
        'IntVec',
        'PlayerInfo',
        'PlayerInfoVec',
        'Position',
        'ScenarioConfig',
        'SharedInfo',
        'StringVector',
        'Vector3',
        'e_BackendAction',
        'e_GameMode',
        'e_PlayerRole',
        'e_RenderingMode',
        'e_Team',
        'data_dir',
        'font_file',
        'game_path',
        'gfootball_dir',
    }
    assert names <= set(dir(lg))


def test_enumerations_print_and_compare_as_grf():
    assert repr(E.e_PlayerRole_GK) == '_gameplayfootball.e_PlayerRole.e_PlayerRole_GK'
    assert str(E.e_PlayerRole_GK) == 'e_PlayerRole_GK' == E.e_PlayerRole_GK.name
    assert format(E.e_PlayerRole_CB) == 'e_PlayerRole_CB' and format(E.e_PlayerRole_CB, 'd') == '1'
    assert isinstance(E.e_PlayerRole_CF, int) and E.e_PlayerRole_CF == 9
    assert hash(E.e_PlayerRole_CF) == 9 and type(E.e_PlayerRole_GK + 1) is int
    assert E(3) == E.e_PlayerRole_RB and E(3) is not E.e_PlayerRole_RB
    assert repr(E(42)) == str(E(42)) == '_gameplayfootball.e_PlayerRole(42)'
    assert not hasattr(E(3), 'name')
    assert repr(E(1.5)) == '_gameplayfootball.e_PlayerRole(1)' and int(E()) == 0
    with pytest.raises(ValueError, match='invalid literal for int'):
        E('x')
    assert list(E.names) == [f'e_PlayerRole_{r}' for r in 'GK CB LB RB DM CM LM RM AM CF'.split()]
    assert E.values[9] is E.e_PlayerRole_CF
    assert lg.e_BackendAction.top_right == 4 and lg.e_BackendAction.builtin_ai == 32
    states = ['game_created', 'game_initiated', 'game_running', 'game_done']
    assert list(lg.GameState.names) == states
    assert [kind.__name__ for kind in (lg.e_GameMode, lg.e_Team, lg.e_RenderingMode)] == [
        'e_GameMode',
        'e_Team',
        'e_RenderingMode',
    ]
    assert E.__base__.__name__ == 'enum' and E.__base__.__module__ == 'Boost.Python'


def test_enumerations_pickle_as_grf():
    loaded = pickle.loads(pickle.dumps(E.e_PlayerRole_CB))
    assert loaded == 1 and repr(loaded) == '_gameplayfootball.e_PlayerRole(1)'
    assert repr(copy.deepcopy(E.e_PlayerRole_CB)) == '_gameplayfootball.e_PlayerRole(1)'
    with pytest.raises(pickle.PicklingError):
        pickle.dumps(E.e_PlayerRole_CB, 0)


def test_conversions_follow_boost_python():
    v = lg.Vector3(1.1, 2.2, 3.3)
    assert (v[0], v[1], v[2]) == (1.100000023841858, 2.200000047683716, 3.299999952316284)
    assert lg.Vector3(True, 2, 3)[0] == 1.0
    with pytest.raises(SignatureError, match=r'Vector3.__init__\(Vector3, int, int\)'):
        lg.Vector3(1, 2)
    with pytest.raises(TypeError):
        lg.Vector3()
    with pytest.raises(SignatureError, match=r'__getitem__\(blunted::Vector3 \{lvalue\}, int\)'):
        v[1.0]
    with pytest.raises(IndexError):
        v[3]
    import numpy as np

    v[0] = np.float64(0.2)
    assert v[0] == 0.20000000298023224
    with pytest.raises(SignatureError):
        lg.Vector3(np.float32(0.1), 0, 0)
    config = lg.ScenarioConfig.make()
    with pytest.raises(SignatureError) as raised:
        config.game_duration = 400.0
    assert str(raised.value) == (
        'Python argument types in\n    None.None(ScenarioConfig, float)\n'
        'did not match C++ signature:\n    None(ScenarioConfig {lvalue}, int)'
    )
    config.game_duration = True
    assert config.game_duration == 1
    with pytest.raises(OverflowError, match='positive overflow'):
        config.game_duration = 2**31
    with pytest.raises(OverflowError, match="can't convert negative value to unsigned int"):
        config.game_engine_random_seed = -1
    config.game_engine_random_seed = 2**32 - 1
    assert config.game_engine_random_seed == 4294967295
    config.right_team_difficulty = 0.1
    assert config.right_team_difficulty == 0.10000000149011612
    config.left_team_difficulty = 2**200
    assert config.left_team_difficulty == float('inf')
    config.deterministic = None
    assert config.deterministic is False
    config.deterministic = 2
    assert config.deterministic is True
    for value in ('x', 0.0, np.bool_(True)):
        with pytest.raises(SignatureError):
            config.deterministic = value
    with pytest.raises(SignatureError):
        config.offsides = np.int64(0)


def test_formation_entries_hold_float_positions():
    f = entry(0.5, 0.1, E.e_PlayerRole_CB)
    assert (f.position[0], f.position[1], f.position[2]) == (0.5, 0.10000000149011612, 0.0)
    f.position[0] = 0.9
    assert f.position[0] == 0.5
    with pytest.raises(AttributeError, match='has no setter'):
        f.role = E.e_PlayerRole_CF
    with pytest.raises(SignatureError):
        lg.FormationEntry(0.5, 0.1, 1, False, True)
    with pytest.raises(SignatureError):
        lg.FormationEntry(x=0.5, y=0.1, role=E.e_PlayerRole_CB, lazy=False, controllable=True)
    assert entry(0.5, 0.1, E.e_PlayerRole_CB, 1).lazy is True
    assert entry(0.5, 0.1, E.e_PlayerRole_CB, None).lazy is False
    assert f != entry(0.5, 0.1, E.e_PlayerRole_CB)
    f.zzz = 1
    assert f.zzz == 1
    with pytest.raises(RuntimeError, match=r'Pickling of "_gameplayfootball\.FormationEntry"'):
        pickle.dumps(f)


def test_vectors_follow_the_indexing_suite():
    config = lg.ScenarioConfig.make()
    team = config.left_team
    team.append(entry(-1.0, 0.0, E.e_PlayerRole_GK))
    team.append(entry(0.5, 0.0, E.e_PlayerRole_CF))
    assert len(config.left_team) == 2 and team[-1].role == E.e_PlayerRole_CF
    assert type(team[0:1]).__name__ == 'FormationEntryVec' and len(team[-1:]) == 1
    with pytest.raises(IndexError, match='Index out of range'):
        team[2]
    with pytest.raises(TypeError, match='Invalid index type'):
        team[1.0]
    with pytest.raises(IndexError, match=r'slice step size not supported\.'):
        team[::2]
    with pytest.raises(TypeError, match='Attempting to append an invalid type'):
        team.append(3)
    with pytest.raises(TypeError, match='Incompatible Data Type'):
        team.extend([entry(0.1, 0.1, E.e_PlayerRole_CB), 1])
    assert len(team) == 2
    assert entry(-1.0, 0.0, E.e_PlayerRole_GK) in team and 3 not in team
    team[0].lazy = True
    assert config.left_team[0].lazy is True
    appended = entry(0.1, 0.1, E.e_PlayerRole_CB)
    team.append(appended)
    appended.lazy = True
    assert team[-1].lazy is False
    with pytest.raises(TypeError, match='Invalid assignment'):
        team[0] = 5
    del team[-1]
    assert len(team) == 2
    with pytest.raises(SignatureError):
        config.left_team = [entry(-1.0, 0.0, E.e_PlayerRole_GK)]
    config.left_team = lg.FormationEntryVec()
    assert len(config.left_team) == 0
    numbers = lg.FloatVec()
    numbers.append(1)
    numbers.append(0.1)
    assert list(numbers) == [1.0, 0.10000000149011612] and 1 in numbers
    strings = lg.StringVector()
    strings.append('ab')
    strings.append(b'cd')
    assert list(strings) == ['ab', 'cd']


def test_configurations_and_their_defaults():
    with pytest.raises(RuntimeError, match='This class cannot be instantiated from Python'):
        lg.ScenarioConfig()
    with pytest.raises(RuntimeError, match='This class cannot be instantiated from Python'):
        lg.GameConfig()
    config = lg.ScenarioConfig.make()
    assert (config.left_agents, config.right_agents, config.game_engine_random_seed) == (1, 0, 42)
    assert config.right_team_difficulty == 0.6000000238418579 and config.second_half == 999999999
    assert (config.use_magnet, config.offsides, config.real_time) == (True, True, False)
    game = lg.GameConfig.make()
    assert (game.render, game.physics_steps_per_frame) == (False, 10)
    assert (game.render_resolution_x, game.render_resolution_y) == (1280, 720)
    assert not hasattr(game, 'data_dir')
    config.left_team.append(entry(-1.0, 0.0, E.e_PlayerRole_GK))
    assert config.controllable_left_players == 1 and config.dynamic_player_selection is False
    config.left_team.append(entry(0.3, 0.2, E.e_PlayerRole_CF))
    # GRF: the counts stay cached until a state is saved or loaded.
    assert config.controllable_left_players == 1
    with pytest.raises(AttributeError, match="property of 'ScenarioConfig' object has no setter"):
        config.controllable_left_players = 3
    for value in (config, game, lg.Vector3(0, 0, 0)):
        with pytest.raises(RuntimeError, match='Pickling of'):
            copy.deepcopy(value)


def test_game_env_before_a_reset():
    env = lg.GameEnv()
    assert repr(env.state) == '_gameplayfootball.GameState.game_created'
    assert env.waiting_for_game_count == 0 and env.config.left_agents == 1
    with pytest.raises(SignatureError, match=r'None\(GameEnv_Python \{lvalue\}, GameState\)'):
        env.state = 2
    env.config.offsides = False
    assert env.config.offsides is False
    env.config.foo = 1
    assert not hasattr(env.config, 'foo')
    env.start_game()
    with pytest.raises(RuntimeError, match='start_game'):
        env.start_game()
    with pytest.raises(cuju.errors.LibraryError, match='reset'):
        env.get_info()
    with pytest.raises(RuntimeError, match=r'Pickling of "_gameplayfootball\.GameEnv"'):
        pickle.dumps(env)


def test_reset_scales_the_ball_of_the_caller():
    # GRF's Python builds the scenario, as `football_env_core` does.
    pytest.importorskip('cuju.gfootball.env.football_env_core')
    from cuju.gfootball.env import config as grf_config

    values = {'level': '5_vs_5', 'game_engine_random_seed': 7}
    config = grf_config.Config(dict(values, players=['agent:left_players=1'])).ScenarioConfig()
    assert (config.ball_position[0], config.ball_position[1]) == (0.0, 0.0)
    env = started(config)
    assert (config.ball_position[0], str(config.ball_position[1])) == (0.0, '-0.0')
    assert str(env.config.ball_position[1]) == '-0.0'
    assert env.config.game_engine_random_seed == 7
    info = env.get_info()
    assert (info.step, info.is_in_play, int(info.game_mode)) == (-1, False, 0)
    assert [c.controlled_player for c in info.left_controllers] == [2, -1, -1, -1, -1]
    goalkeeper = info.left_team[0]
    assert goalkeeper.position[0] == -0.4595588147640228
    assert str(goalkeeper.position) == '-25.000000,0.000000,0.000000'
    assert repr(goalkeeper.role) == '_gameplayfootball.e_PlayerRole.e_PlayerRole_GK'
    steps = 0
    while not env.get_info().is_in_play:
        env.step()
        steps += 1
    assert steps == 19


def test_actions_and_sticky_actions():
    env = started()
    while not env.get_info().is_in_play:
        env.step()
    right = lg.e_BackendAction.right
    assert env.sticky_action_state(int(right), True, 0) is False
    env.perform_action(right, True, 0)
    assert env.sticky_action_state(right, True, 0) is True
    with pytest.raises(SignatureError):
        env.perform_action(1.0, True, 0)
    with pytest.raises(cuju.errors.ArgumentError, match='sticky'):
        env.sticky_action_state(int(lg.e_BackendAction.short_pass), True, 0)
    with pytest.raises(cuju.errors.ArgumentError, match='controller'):
        env.perform_action(0, True, 30)


def test_states_round_trip():
    env = started()
    for _ in range(30):
        env.step()
    state = env.get_state(b'abc')
    assert state[:7] == b'\x03\x00\x00\x00abc' and state[7:11] == b'\x02\x00\x00\x00'
    assert env.set_state(state) == b'abc'
    assert env.get_state(b'abc') == state
    assert env.get_state('abc') == state


def test_states_of_another_scenario_refused():
    first, second = started(), started(five_a_side(left_agents=2))
    with pytest.raises(cuju.errors.StateError, match='left_agents'):
        second.set_state(first.get_state(b''))


def raw_digest(observation):
    """The SHA-256 of GRF's raw observation, every value widened, as the clean build gave it."""
    import hashlib

    import numpy as np

    digest = hashlib.sha256()
    for key in sorted(observation):
        value = np.asarray(observation[key])
        if value.dtype.kind == 'f':
            value = value.astype(np.float64)
        elif value.dtype.kind in 'iub':
            value = value.astype(np.int64)
        digest.update(key.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


@pytest.mark.parametrize(
    ('skip', 'seed', 'saved_at', 'loaded_at', 'steps', 'expected'),
    [
        (0, 1, 82, 2, 14, 'c9f98555f98f12b6460e0ead9a6bc737bb96205fd0281e606b96494feff59045'),
        (1, 2, 100, 3, 115, '972032a1ce518fe4c7731505fdeeef79799e3d7cabbffc6f53148232ac478759'),
    ],
)
def test_a_state_of_the_other_parity_plays_as_in_grf(
    skip, seed, saved_at, loaded_at, steps, expected
):
    # GRF: `11_vs_11_stochastic` swaps the teams' formations with the parity of the episode.
    from cuju.gfootball.env import config, football_env

    values = {'level': '11_vs_11_stochastic', 'players': [], 'game_engine_random_seed': seed}
    env = football_env.FootballEnv(config.Config(values))
    for _ in range(1 + skip):
        env.reset()
    for _ in range(saved_at):
        env.step([])
    state = env.get_state()
    env.reset()
    for _ in range(loaded_at):
        env.step([])
    env.set_state(state)
    for _ in range(steps):
        env.step([])
    observation = env._env._observation
    env.close()
    assert raw_digest(observation) == expected


def test_scenarios_grf_cannot_play_refused():
    config = five_a_side()
    config.left_team[0] = entry(-1.0, 0.0, E.e_PlayerRole_CB)
    with pytest.raises(ScenarioError, match='goalkeeper'):
        started(config)
    config = five_a_side(left_agents=6)
    with pytest.raises(ScenarioError, match='controllers'):
        started(config)


def test_config_writes_reach_the_match():
    env = started()
    for _ in range(25):
        env.step()
    env.config.game_engine_random_seed = 99
    state = env.get_state(b'')
    other = started()
    for _ in range(25):
        other.step()
    assert state != other.get_state(b'')
    with pytest.raises(ScenarioError, match='reverse_team_processing'):
        env.config.reverse_team_processing = True
        env.step()


def test_the_tracker_compares_two_engines():
    import threading

    errors = []

    def play(env):
        try:
            for step in range(15):
                env.perform_action(step % 19, True, 0)
                env.step()
        except Exception as error:
            errors.append(error)

    engines = [started(), started()]
    for env in engines:
        env.tracker_setup(0, 10**9)
    threads = [threading.Thread(target=play, args=(env,)) for env in engines]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert not errors and not any(thread.is_alive() for thread in threads)


def test_rendering_names_the_renderer(monkeypatch):
    from cuju.gfootball_engine._render_hooks import RendererMissingError

    monkeypatch.setenv('CUJU_RENDER_LIBRARY', os.path.join(os.sep, 'nowhere', 'libcuju_render'))
    env = lg.GameEnv()
    env.game_config.render = True
    with pytest.raises(RendererMissingError, match=r'cuju-engine\[pixels\]'):
        env.start_game()
    plain = started()
    with pytest.raises(RuntimeError):
        plain.render(True)


def test_animations_reach_the_referee():
    # GRF: with animations every stoppage plays out in full, which moves the clock of the match.
    pytest.importorskip('cuju.gfootball.env.football_env_core')
    from cuju.gfootball.env import config as grf_config

    values = {'level': '11_vs_11_stochastic', 'game_engine_random_seed': 3}
    states = []
    for animations in (False, True):
        config = grf_config.Config(dict(values, players=['agent:left_players=1'])).ScenarioConfig()
        env = started(config, animations)
        for _ in range(600):
            env.step()
        states.append(env.get_state(b''))
    assert states[0] != states[1]
