# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""GRF's `create_environment` and gymnasium ids, each environment a batch of one."""

from __future__ import annotations

import collections
import importlib
import importlib.metadata
import importlib.util
import io
import os
import pickle
import platform
import random
import re
import struct
import sys
import warnings
from typing import Any

import numpy as np

try:
    import gymnasium as _gym
except ImportError:
    try:
        import gym as _gym
    except ImportError:
        raise ImportError(
            "cuju.env needs gymnasium, from pip install 'cuju-engine[spaces]', or GRF's gym"
        ) from None

from . import _replays
from ._assets import scenario_names
from ._batch import Batch
from ._batch import frame_stack as _frame_stack
from ._batch.state import _pickle_episode, _StateUnpickler
from ._players import Bot, Lazy
from ._raw import grf_observation, grf_observations
from ._wrappers import MultiAgentToSingleAgent
from .errors import ChoiceError, StateError

__all__ = ['FootballEnvironment', 'create_environment', 'grf_observation', 'register_environments']

_APIS = ('gym', 'gymnasium')

# GRF: `observation_preprocessing.SMM_WIDTH` and `SMM_HEIGHT`.
_SMM_SIZE = (96, 72)
# GRF: every scenario that `scenario_builder` builds draws `random.randint(0, 2000000000)`.
_SEED_LIMIT = 2000000000
# GRF: the core actions in the order of the `full` set.
_ACTION_NAMES = (
    'idle',
    'left',
    'top_left',
    'top',
    'top_right',
    'right',
    'bottom_right',
    'bottom',
    'bottom_left',
    'long_pass',
    'high_pass',
    'short_pass',
    'shot',
    'sprint',
    'release_direction',
    'release_sprint',
    'sliding',
    'dribble',
    'release_dribble',
    'builtin_ai',
    'keeper_rush',
    'pressure',
    'team_pressure',
    'switch',
    'release_long_pass',
    'release_high_pass',
    'release_short_pass',
    'release_shot',
    'release_keeper_rush',
    'release_sliding',
    'release_pressure',
    'release_team_pressure',
    'release_switch',
)
_ACTION_COUNTS = {'default': 19, 'v2': 20, 'full': 33}
# GRF: options of `Config` that only rendering reads, so without it they change nothing.
_DISPLAY_OPTIONS = frozenset(
    {
        'custom_display_stats',
        'display_game_stats',
        'real_time',
        'render_resolution_x',
        'render_resolution_y',
        'video_format',
        'video_quality_level',
    }
)
_OPTIONS = _DISPLAY_OPTIONS | {
    'action_set',
    'dump_full_episodes',
    'dump_scores',
    'episode_number',
    'game_engine_random_seed',
    'level',
    'physics_steps_per_frame',
    'players',
    'reverse_team_processing',
    'tracesdir',
    'write_video',
}
_PLAYERS = ('agent', 'bot', 'lazy')
# GRF: any other player is a module of `gfootball.env.players`, here one so named or a module path.
_PLAYER_MODULE = re.compile(r'\w+(\.\w+)*')


def _numpy(tensor):
    return tensor.detach().cpu().numpy().copy()


def _parse_player(definition):
    name, parameters = definition, {}
    if ':' in definition:
        # GRF: on Windows the path of a replay may hold a colon.
        if (
            platform.system() == 'Windows'
            and definition.startswith('replay:')
            and len(definition.split(':')) > 2
        ):
            name, text = 'replay', definition.split('replay:')[-1]
        else:
            name, text = definition.split(':')
        for parameter in text.split(','):
            key, value = parameter.split('=')
            parameters[key] = value
    counted = 'left_players' in parameters or 'right_players' in parameters
    left = int(parameters.pop('left_players', 0))
    right = int(parameters.pop('right_players', 0))
    return name, left if counted else 1, right, parameters


def _scenario_facts(asset):
    if isinstance(asset, str) and asset.endswith('.py') and os.path.isfile(asset):
        from . import scenarios

        # GRF: `create_environment` reads the scenario of episode 1, the odd form.
        odd = scenarios.describe(asset)[1]
        control = None
        if odd['values']['control_all_players']:
            control = tuple(
                sum(1 for player in odd[side] if player[4]) for side in ('left', 'right')
            )
        return bool(odd['values']['deterministic']), control
    if isinstance(asset, str) and not asset.endswith('.cjb') and os.sep not in asset:
        # GRF: no shipped scenario is deterministic or sets `control_all_players`.
        return False, None
    return None, None


def _numpy_values(value):
    from . import dumps

    if isinstance(value, dumps._Array):
        numpy_type, _ = value.kind
        return np.array(value.values, dtype=np.dtype(numpy_type.order + numpy_type.code)).reshape(
            value.shape
        )
    if isinstance(value, dict):
        return {key: _numpy_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_numpy_values(item) for item in value]
    return value


def _space_shape(base, agents, single, stacked):
    # GRF: the representation wrappers size the agent axis from the action space.
    shape = (agents if agents > 1 else 1, *base)
    if single:
        shape = shape[1:]
    if stacked:
        shape = (*shape[:-1], shape[-1] * 4)
    return shape


class _Dumps:
    """GRF's dump options, which `cuju.dumps.DumpWriter` writes after each step, outside `step`."""

    def __init__(self, batch, full_episodes, goals, frequency, directory):
        try:
            from . import dumps
        except ImportError:
            raise NotImplementedError(
                "GRF's dump options write through cuju.dumps, which this installation lacks"
            ) from None
        self._frequency = frequency
        self._resets = 0
        # GRF: an empty `logdir` writes the dumps in the working directory.
        self._writer = dumps.DumpWriter(
            batch,
            directory or os.curdir,
            environments=[0],
            write_full_episode_dumps=bool(full_episodes),
            write_goal_dumps=bool(goals),
        )
        # GRF: one environment names its dumps by their kind and the time they start.
        self._writer._grf_names = True

    def after_step(self, actions):
        return [{'dump': path, 'name': kind} for kind, path in self._writer._record(actions)]

    def follow(self, config):
        # GRF: `create_environment` applies `PeriodicDumpWriter` only for a frequency above 1.
        kept = self._frequency <= 1 or self._resets % self._frequency == 0
        self._resets += 1
        self._writer.write_full_episode_dumps = kept and bool(config['dump_full_episodes'])
        self._writer.write_goal_dumps = kept and bool(config['dump_scores'])

    def close(self):
        self._writer.close()


# GRF: the keys of a state's pickle, of the episode logic and of the wrappers that keep states.
_CORE, _CHECKPOINTS, _FRAMES = 'FootballEnvCore', 'CheckpointRewardWrapper', 'FrameStack'
# GRF: the attributes of `EnvState`, the episode logic.
_LOGIC = ('previous_score_diff', 'previous_game_mode', 'prev_ball_owned_team')


def _split_state(state):
    if len(state) < 4 or not 0 < struct.unpack_from('<i', state)[0] <= len(state) - 4:
        raise StateError(
            'expected the bytes of a state: a pickle length, the pickle and the match, '
            f'found {len(state)} bytes that hold none'
        )
    (length,) = struct.unpack_from('<i', state)
    unpickler = _StateUnpickler(io.BytesIO(state[4 : 4 + length]))
    unpickler.stacked = True
    try:
        entries = unpickler.load()
        [vars(entries[_CORE])[name] for name in _LOGIC]
    except (pickle.UnpicklingError, KeyError, TypeError, EOFError) as error:
        raise StateError(
            f'expected the pickle of the episode logic in the state, found {error!r}'
        ) from error
    return entries, state[4 + length :]


def _pickled(value):
    return pickle.dumps(value, protocol=2)[2:-1]


def _own_frame(frame):
    if not isinstance(frame, np.ndarray) or frame.dtype.kind not in 'biufc':
        return frame
    return frame.astype(np.dtype(frame.dtype.str))


def _join_state(entries, match):
    episode = [vars(entries[_CORE])[name] for name in _LOGIC]
    collected = None
    if _CHECKPOINTS in entries:
        counts = entries[_CHECKPOINTS]
        collected = [counts.get(agent, 0) for agent in range(max(counts, default=-1) + 1)]
    frames = None
    if _FRAMES in entries:
        stack = entries[_FRAMES]
        frames = _pickled(
            collections.deque(map(_own_frame, stack), maxlen=getattr(stack, 'maxlen', None))
        )
    data = _pickle_episode(episode, collected, frames)
    others = b''.join(
        _pickled(key) + _pickled(value)
        for key, value in entries.items()
        if key not in (_CORE, _CHECKPOINTS, _FRAMES)
    )
    data = data[:-2] + others + data[-2:]
    return struct.pack('<i', len(data)) + data + match


class FootballEnvironment(_gym.Env):
    """One GRF environment, from `create_environment`."""

    # GRF: gym's `Env`, whose rewards are unbounded.
    reward_range = (-float('inf'), float('inf'))

    def __init__(self, batch, settings):
        self.batch = batch
        self._settings = settings
        self._config = settings['config']
        self._slots = settings['slots']
        # GRF: a configuration without an `agent` player steps with `step([])`.
        self._agent = settings.get('agent', True)
        # GRF: the single-agent wrappers follow the agent counts of the arguments.
        self._single = settings['single']
        self._representation = settings['representation']
        self._fixed = settings['fixed_seed'] is not None
        self._done = False
        self._gymnasium = settings['api'] == 'gymnasium'
        self._dumps = settings['dumps']
        self._wrapper = settings['wrapper']
        # GRF: the scripted players of `extra_players`, which act on their own observations.
        self._scripted = [
            (
                Bot(batch, slots[0])
                if name == 'bot'
                else Lazy(batch, slots)
                if name == 'lazy'
                else name.bind(batch),
                slots,
            )
            for name, slots in settings['scripted']
        ]
        self._python_players = settings.get('python_players')
        # GRF: the steps of the episode that `write_dump` writes.
        self._episode_steps = _replays.EpisodeSteps()
        self._played = self._episode_steps.reset()
        count = settings['action_count']
        agents = len(self._slots) if self._wrapper is None else self._wrapper.players
        # GRF's gym interface takes gym's spaces where an older gym is installed.
        spaces = (_legacy_gym() or _gym).spaces if settings['api'] == 'gym' else _gym.spaces
        self.action_space = (
            spaces.MultiDiscrete([count] * agents) if agents > 1 else spaces.Discrete(count)
        )
        width, height = settings['channel_dimensions']
        stacked = settings['stacked']
        if self._representation == 'extracted':
            shape = _space_shape((height, width, 4), agents, self._single, stacked)
            self.observation_space = spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        elif self._representation != 'raw':
            shape = _space_shape((115,), agents, self._single, stacked)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=shape, dtype=np.float32
            )
        else:
            # GRF: the raw representation has no space, and gymnasium's `Env` declares none.
            self.observation_space = None

    @property
    def _num_actions(self) -> int:
        return self._settings['action_count']

    def _observe(self):
        if self._wrapper is not None:
            if self._representation == 'raw':
                return self._raw(self._wrapper._index[0].tolist())
            array = _numpy(self._wrapper.observation()[0])
        elif self._representation == 'raw':
            return self.observation()
        else:
            array = _numpy(self.batch.observation()[0, self._slots])
        return array[0] if self._single else array

    def _raw(self, slots):
        return grf_observations(self.batch, 0, slots)

    def observation(self) -> list[dict[str, Any]] | dict[str, Any] | None:
        """Returns the raw observation of each of the agent's players, as GRF's unwrapped one."""
        if not self._agent:
            return self._match_observation()
        return self._raw(self._slots)

    def _match_observation(self):
        from . import dumps

        batch = self.batch
        if batch.agents == 0:
            return None
        raw = batch.raw()
        rows = [
            {name: tensor[0, agent].tolist() for name, tensor in raw.items()}
            for agent in range(batch.agents)
        ]
        values, _ = dumps._episode_values(batch, 0)
        designated = values[15:17] if self._done else values[13:15]
        episode = dumps._Episode(0, values, None, False, False)
        sticky = dumps._sticky_names(batch.action_count)
        return _numpy_values(dumps._match_observation(rows, episode, designated, sticky))

    def _actions(self, action):
        # GRF: `FootballEnv._action_to_list`, then one action per player of the agent.
        if isinstance(action, np.ndarray):
            action = action.tolist()
        if not isinstance(action, list):
            action = [action]
        count = len(self._slots) if self._wrapper is None else self._wrapper.players
        if len(action) != count:
            if not self._agent:
                # GRF: `FootballEnv.step` without an agent asserts that it gets no action.
                raise ValueError(f'step() received {len(action)} actions, but no agent is playing')
            raise ValueError(
                f'expected {count} actions, one for each player of the agent, found {len(action)}'
            )
        names = _ACTION_NAMES[: self._settings['action_count']]
        values = []
        for value in action:
            if hasattr(value, '_name') and hasattr(value, '_backend_action'):
                # GRF: a `CoreAction` stands for itself.
                if value._name not in names:
                    raise ValueError(
                        f'expected an action of the set {self._settings["action_set"]!r}, found '
                        f'{value._name!r}; create the environment with that set'
                    )
                value = names.index(value._name)
            elif isinstance(value, (bool, int, np.int32, np.int64)):
                value = int(value)
            else:
                raise TypeError(f'expected an integer action, found {value!r}')
            values.append(value)
        scripted = [
            (slot, value)
            for player, slots in self._scripted
            for slot, value in zip(slots, player.act().reshape(-1).tolist(), strict=True)
        ]
        if self._wrapper is not None:
            return values, scripted
        actions = [0] * self.batch.agents
        for slot, value in [*zip(self._slots, values, strict=True), *scripted]:
            actions[slot] = value
        return actions, scripted

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> Any:
        """Starts the next episode, with a seed drawn from Python's `random` as GRF draws it."""
        if seed is not None:
            # GRF: gym 0.21 takes no seed in `reset`, so only gymnasium's base class seeds.
            super().reset(seed=seed)
        target = self.batch if self._wrapper is None else self._wrapper
        if self._dumps is not None:
            # GRF: the dump options hold for the episode that the reset starts.
            self._dumps.follow(self._config)
        episode_seed = None if self._fixed else random.randint(0, _SEED_LIMIT)
        if episode_seed is None:
            target.reset()
        else:
            target.reset(seeds=episode_seed)
        # GRF: each reset builds the scenario of the next episode number with the seed it drew.
        self._config._new_scenario(1, episode_seed)
        if self._python_players is not None:
            self._python_players.reset()
        self._played = self._episode_steps.reset()
        self._done = False
        if self._gymnasium:
            return self._observe(), {}
        return self._observe()

    def step(self, action: Any) -> tuple[Any, ...]:
        """Runs one step."""
        if self._done:
            raise RuntimeError("can't call step() once the episode finished; call reset() first")
        actions, scripted = self._actions(action)
        if self._wrapper is None:
            # GRF: a negative index counts from the end, which the batch plays as the same action.
            count = self._settings['action_count']
            _, reward, done, information = self.batch.step(
                [[value + count if value < 0 else value for value in actions]]
            )
            played = [actions]
        else:
            _, reward, done, information = self._wrapper.step([actions], _others=scripted)
            played = self._wrapper.last_actions
        # GRF: the reward of each of the agent's players, as `MultiAgentToSingleAgent` passes on.
        reward = _numpy(reward[0, self._slots])
        self._played.append(played)
        self._done = bool(done[0])
        score_reward = int(information['score_reward'][0])
        info = {'score_reward': score_reward}
        if self._dumps is not None:
            # GRF: the information lists the dumps that the step finished.
            written = self._dumps.after_step(played)
            if written:
                info['dumps'] = written
        observation = self._observe()
        if not self._agent:
            # GRF: without an agent the reward is the scoring reward itself.
            reward = np.array(score_reward, dtype=np.float32)
        elif self._single:
            reward = reward[0]
        if self._gymnasium:
            return observation, reward, self._done, False, info
        return observation, reward, self._done, info

    def seed(self, seed: int | None = None) -> None:
        """Does nothing and returns `None`, as gym's `Env.seed` under GRF's environment does."""
        return None

    def disable_render(self) -> None:
        """Does nothing, since cuju renders nothing."""

    def _refuse_wrapper_state(self):
        if self._wrapper is not None:
            raise ValueError(
                "GRF's MultiAgentToSingleAgent supports no states, so a scenario with "
                'control_all_players played by one player a side has none'
            )

    def get_state(self, to_pickle: dict[str, Any] | None = None) -> bytes:
        """Returns the state of the match in GRF's byte format, as GRF's `get_state(to_pickle)`."""
        self._refuse_wrapper_state()
        batch = self.batch
        entries, match = _split_state(batch.get_state(0, stacked=batch.stacked))
        own = {_CORE: entries[_CORE]}
        if _CHECKPOINTS in entries:
            # GRF: the counts go by the agent's players, which play the batch's slots.
            counts = entries[_CHECKPOINTS]
            own[_CHECKPOINTS] = {
                index: counts[slot] for index, slot in enumerate(self._slots) if slot in counts
            }
        if batch.stacked:
            # GRF: `FrameStack` holds the observations of the agent's players.
            stack = batch.observation()[0, self._slots].cpu()
            own[_FRAMES] = collections.deque(
                _frame_stack.frames_of(stack, len(self._slots)), maxlen=_frame_stack.FRAMES
            )
        return _join_state({**(to_pickle or {}), **own}, match)

    def set_state(self, state: bytes) -> dict[str, Any]:
        """Loads a state in GRF's byte format, as GRF's `set_state(state)`."""
        self._refuse_wrapper_state()
        batch = self.batch
        entries, match = _split_state(state)
        core = {_CORE: entries[_CORE]}
        own = dict(core)
        if batch._checkpoints and _CHECKPOINTS in entries:
            counts = entries[_CHECKPOINTS]
            own[_CHECKPOINTS] = {
                slot: counts[index] for index, slot in enumerate(self._slots) if index in counts
            }
        if batch.stacked and _FRAMES in entries:
            own[_FRAMES] = self._batch_frames(entries[_FRAMES])
        batch.set_state(0, _join_state(own, match), stacked=batch.stacked)
        self._played = self._episode_steps.set_state(_join_state(core, match), self._done)
        self._done = False
        return entries

    def _batch_frames(self, frames):
        batch = self.batch
        agents = batch.agents
        if list(self._slots) == list(range(agents)):
            return frames
        own = list(frames) if isinstance(frames, (collections.deque, list, tuple)) else []
        current = _frame_stack.frames_of(batch.observation()[0, :agents].cpu(), agents)
        if len(own) != _frame_stack.FRAMES:
            raise StateError(
                f"expected the {_frame_stack.FRAMES} frames of GRF's FrameStack in the state"
            )
        shape = current[0].shape[1:]
        expected = shape if len(self._slots) == 1 else (len(self._slots), *shape)
        stack = []
        for frame, mine in zip(current, own, strict=True):
            mine = np.asarray(mine)
            if mine.shape != expected or mine.dtype != frame.dtype:
                raise StateError(
                    f'expected frames of the shape {expected} and the type {frame.dtype} in the '
                    f'state, found {mine.shape} and {mine.dtype}'
                )
            frame = frame.copy()
            frame[self._slots] = mine.reshape((len(self._slots), *shape))
            stack.append(frame)
        return collections.deque(stack, maxlen=_frame_stack.FRAMES)

    def write_dump(self, name: str) -> str | None:
        """Writes the last steps of the episode as a dump, as GRF's `write_dump(name)`."""
        return self._episode_steps.write_dump(self, name)

    def close(self) -> None:
        """Frees the batch."""
        if self._dumps is not None:
            self._dumps.close()
            self._dumps = None
        self.batch.close()


def create_environment(
    env_name: str = '',
    stacked: bool = False,
    representation: str = 'extracted',
    rewards: str = 'scoring',
    write_goal_dumps: bool = False,
    write_full_episode_dumps: bool = False,
    render: bool = False,
    write_video: bool = False,
    dump_frequency: int = 1,
    logdir: str = '',
    extra_players: list[str] | None = None,
    number_of_left_players_agent_controls: int = 1,
    number_of_right_players_agent_controls: int = 0,
    channel_dimensions: tuple[int, int] = _SMM_SIZE,
    other_config_options: dict[str, Any] | None = None,
    *,
    device: str = 'cpu',
    threads: int = 1,
    api: str = 'gym',
) -> FootballEnvironment:
    """Creates one GRF environment, as `gfootball.env.create_environment` does."""
    if api not in _APIS:
        raise ValueError(f'expected api {" or ".join(map(repr, _APIS))}, found {api!r}')
    if not env_name:
        raise ValueError('expected the name of a scenario in env_name')
    if render or write_video:
        raise ValueError('cuju renders nothing, so it supports neither render nor write_video')
    if representation in ('pixels', 'pixels_gray'):
        # GRF: these need its render switch, which `cuju.gfootball` plays as GRF does.
        raise ChoiceError(
            f"cuju.env renders nothing, so it has no representation {representation!r}; GRF's "
            'pixels come from cuju.gfootball, and a BatchEnvironment draws them for its episodes'
        )
    options = dict(other_config_options or {})
    unknown = sorted(set(options) - _OPTIONS)
    if unknown:
        warnings.warn(
            f'other_config_options holds {unknown}, which GRF keeps and ignores, as cuju does',
            stacklevel=2,
        )
    left_agents = number_of_left_players_agent_controls
    right_agents = number_of_right_players_agent_controls
    players = [f'agent:left_players={left_agents},right_players={right_agents}']
    facts = _scenario_facts(env_name)
    control = facts[1]
    # GRF: `control_all_players` gives the agent every player, and the wrapper plays one a side.
    single_players = control is not None and left_agents in (0, 1) and right_agents in (0, 1)
    if single_players:
        if 'players' in options or not left_agents + right_agents:
            raise ValueError(
                'expected one player on one or both sides, and the players of extra_players '
                'rather than of other_config_options, for a scenario with control_all_players'
            )
        players = [
            f'agent:left_players={control[0] if left_agents else 0},'
            f'right_players={control[1] if right_agents else 0}'
        ]
    if extra_players is not None:
        players.extend(extra_players)
    # GRF: the values of the environment's `Config`, which `other_config_options` updates.
    values = {
        'dump_full_episodes': write_full_episode_dumps,
        'dump_scores': write_goal_dumps,
        'players': players,
        'level': env_name,
        'tracesdir': logdir,
        'write_video': write_video,
        **options,
    }
    batch, settings = _create(
        values,
        env_name,
        facts,
        config=None,
        representation=representation,
        rewards=rewards,
        stacked=stacked,
        channel_dimensions=channel_dimensions,
        dump_frequency=dump_frequency,
        sides=(left_agents, right_agents) if single_players else None,
        single=left_agents + right_agents == 1,
        device=device,
        threads=threads,
        api=api,
    )
    legacy = _legacy_gym() if api == 'gym' else None
    if legacy is None:
        return FootballEnvironment(batch, settings)
    return _legacy_class(legacy)(batch, settings)


def _legacy_gym():
    if _gym.__name__ == 'gym' or not _gym_before_026():
        return None
    try:
        return importlib.import_module('gym')
    except Exception:
        return None


_LEGACY_CLASSES: dict[str, type] = {}


def _legacy_class(gym) -> type:
    cls = _LEGACY_CLASSES.get(gym.__version__)
    if cls is None:
        cls = type('FootballEnvironment', (FootballEnvironment, gym.Env), {})
        cls.__module__ = __name__
        cls.__doc__ = FootballEnvironment.__doc__
        _LEGACY_CLASSES[gym.__version__] = cls
    return cls


def _create(
    values,
    env_name,
    facts,
    *,
    config,
    representation,
    rewards,
    stacked,
    channel_dimensions,
    dump_frequency,
    sides,
    single,
    device,
    threads,
    api,
):
    single_players = sides is not None
    players = list(values['players'])
    level = values['level']
    action_set = values.get('action_set', 'default')
    if action_set not in _ACTION_COUNTS:
        raise KeyError(f'expected the action set {", ".join(map(repr, _ACTION_COUNTS))}')
    # GRF: each player controls the next positions of each side, in the order of the list.
    parsed, left, right, slots, scripted = [], 0, 0, None, []
    for definition in players:
        name, player_left, player_right, parameters = _parse_player(definition)
        # GRF: another name is a module of players, which loads with the players, below.
        if name not in _PLAYERS and not _PLAYER_MODULE.fullmatch(name):
            raise ValueError(
                f'expected the players {", ".join(_PLAYERS)}, or a module that defines the player, '
                f'in {definition!r}'
            )
        if name == 'bot' and player_left + player_right != 1:
            raise ValueError(f"GRF's bot controls one player, found {definition!r}")
        # GRF: the bot asserts the `full` action set.
        if name == 'bot' and action_set != 'full':
            raise ValueError("GRF's bot needs other_config_options={'action_set': 'full'}")
        parsed.append((name, left, player_left, right, player_right, parameters))
        left += player_left
        right += player_right
    for name, left_start, player_left, right_start, player_right, _ in parsed:
        positions = list(range(left_start, left_start + player_left)) + [
            left + position for position in range(right_start, right_start + player_right)
        ]
        if name == 'agent':
            if slots is not None:
                raise ValueError("expected one 'agent' player, as GRF allows")
            slots = positions
        elif positions or name not in _PLAYERS:
            scripted.append((name, positions))
    # GRF: `FootballEnv` plays without an agent too, stepping with `step([])`.
    agent = slots is not None
    slots = slots or []
    if config is None:
        # GRF: `create_environment` builds a configuration that draws a seed anyway.
        random.randint(0, _SEED_LIMIT)
    fixed_seed = values.get('game_engine_random_seed')
    if fixed_seed is not None and not 0 <= fixed_seed < 2**32:
        raise ValueError(
            f'expected a game_engine_random_seed from 0 to 2**32 - 1, found {fixed_seed}'
        )
    reverse = values.get('reverse_team_processing')
    episode_number = values.get('episode_number', 0)
    deterministic = (facts if level == env_name else _scenario_facts(level))[0]
    # GRF: without a fixed seed, the creation draws the seeds, the first one setting the team order.
    creation = None
    if fixed_seed is None:
        first = (
            random.randint(0, _SEED_LIMIT) if config is None else config['game_engine_random_seed']
        )
        creation = [first]
    python_players = None
    if any(name not in _PLAYERS for name, _ in scripted):
        from ._python_players import PythonPlayers

        # GRF: `FootballEnv` builds its players between the configuration and the core.
        config = _grf_config_values(values, config, deterministic, fixed_seed, creation)
        python_players = PythonPlayers(players, parsed, scripted, config, action_set)
        scripted = python_players.scripted
    if creation is not None:
        creation.append(random.randint(0, _SEED_LIMIT))
    numpy_legacy = int(np.__version__.split('.')[0]) < 2
    batch = Batch(
        level,
        environment_count=1,
        representation=representation,
        rewards=rewards,
        action_set='v2' if single_players and action_set == 'default' else action_set,
        numpy_legacy=numpy_legacy,
        auto_reset=False,
        with_raw=True,
        device=device,
        threads=threads,
        physics_steps_per_frame=values.get('physics_steps_per_frame', 10),
        channel_dimensions=tuple(channel_dimensions),
        stacked=stacked and not single_players,
        game_engine_random_seed=fixed_seed,
        agent_counts=[(left, right)],
        episode_number=(episode_number + (0 if creation and deterministic is None else 1)) % 2,
    )
    try:
        if creation is None:
            batch.reset(reverse=None if reverse is None else int(bool(reverse)))
        elif deterministic is None:
            batch.reset(seeds=creation[0], reverse=None if reverse is None else int(bool(reverse)))
            batch.reset(seeds=creation[1])
        else:
            if reverse is None:
                reverse = 0 if deterministic else creation[0] % 2
            batch.reset(seeds=creation[1], reverse=int(bool(reverse)))
        dumps = None
        dump_full = values.get('dump_full_episodes', False)
        dump_goals = values.get('dump_scores', False)
        if values.get('write_video', False):
            raise ValueError('cuju renders nothing, so it writes no video')
        if dump_full or dump_goals:
            dumps = _Dumps(
                batch, dump_full, dump_goals, dump_frequency, values.get('tracesdir', '')
            )
        wrapper = None
        if single_players:
            # GRF: the wrapper covers the agent's players, `extra_players` play their own actions.
            wrapper = MultiAgentToSingleAgent(
                batch, *sides, action_set=action_set, stacked=stacked, _slots=slots
            )
        config = _grf_config(values, config, deterministic, fixed_seed, creation)
    except BaseException:
        batch.close()
        raise
    settings = {
        'config': config,
        'agent': agent,
        'slots': slots,
        'single': single,
        'wrapper': wrapper,
        'scripted': scripted,
        'representation': representation,
        'fixed_seed': fixed_seed,
        'action_set': action_set,
        'action_count': _ACTION_COUNTS[action_set],
        'channel_dimensions': tuple(channel_dimensions),
        'stacked': stacked,
        'dumps': dumps,
        'api': api,
        'python_players': python_players,
        'level': level,
    }
    return batch, settings


def _grf_config(values, config, deterministic, fixed_seed, creation):
    config = _grf_config_values(values, config, deterministic, fixed_seed, creation)
    # GRF: the creation of the core builds the scenario again with the same number and a new seed.
    config._new_scenario(0, None if creation is None else creation[1])
    return config


def _grf_config_values(values, config, deterministic, fixed_seed, creation):
    from . import _grf_config as grf_config

    if config is None:
        values = dict(values)
        # GRF: the `Config` numbers the scenario, and the parity of its seed sets the team order.
        values['episode_number'] = values.get('episode_number', 0) + 1
        if deterministic is False and 'reverse_team_processing' not in values:
            seed = fixed_seed if creation is None else creation[0]
            values['reverse_team_processing'] = bool(seed % 2)
        config = grf_config._without_draw(values)
        if creation is not None:
            # GRF: the seed that the configuration drew is a value of its scenario.
            config.set_scenario_value('game_engine_random_seed', creation[0])
    return config


# GRF: the ids of `gfootball/__init__.py`, by the suffix of each representation.
_GYM_REPRESENTATIONS = (
    ('SMM', 'extracted'),
    ('Pixels', 'pixels'),
    ('simple115', 'simple115'),
    ('simple115v2', 'simple115v2'),
)
_ENTRY_POINT = 'cuju.env:create_environment'


def _gym_version() -> tuple[int, int] | None:
    try:
        version = importlib.metadata.version('gym')
    except importlib.metadata.PackageNotFoundError:
        return None
    match = re.match(r'(\d+)\.(\d+)', version)
    return (int(match[1]), int(match[2])) if match else None


def _gym_before_026() -> bool:
    version = _gym_version()
    return version is not None and version < (0, 26)


def register_environments() -> None:
    """Registers GRF's ids `GFootball-<scenario>-<representation>-v0` for the shipped scenarios."""
    if importlib.util.find_spec('gymnasium') is not None:
        _register_with('gymnasium')


def _register_with(module: str) -> None:
    if module != 'gymnasium':
        return
    registration = importlib.import_module(f'{module}.envs.registration')
    registry = registration.registry
    specs = registry if isinstance(registry, dict) else registry.env_specs
    for scenario in scenario_names():
        for suffix, representation in _GYM_REPRESENTATIONS:
            name = f'GFootball-{scenario}-{suffix}-v0'
            spec = specs[name] if name in specs else None
            if spec is not None and spec.entry_point == _ENTRY_POINT:
                continue
            registration.register(
                id=name,
                entry_point=_ENTRY_POINT,
                kwargs={'env_name': scenario, 'representation': representation, 'api': module},
            )


if sys.modules.get('gymnasium') is not None:
    _register_with('gymnasium')
