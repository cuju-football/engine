# Migrating from GRF

cuju plays Google Research Football's matches, so code written for GRF moves over in steps. Three entry points exist, and they play the same matches.

| Entry point | What it is | Use it for |
| --- | --- | --- |
| `cuju.gfootball` | GRF's own Python package over cuju's engine | existing code, evaluation scripts, notebooks, GRF's tests |
| `cuju.env.create_environment` | GRF's function, one environment per call, several times faster than GRF | checking a port, comparisons, small experiments |
| `cuju.BatchEnvironment` | thousands of environments per call, outputs as tensors | training |

A port has two stages. The imports switch first, so existing code runs. The training loop then moves to a batch, while the rest stays on GRF's interface.

## GRF's Python package

`cuju.gfootball` holds every module of GRF 2.10.3 with its imports pointed at cuju. Only the imports change.

```python
import cuju.gfootball.env as football_env
from cuju.gfootball.env import observation_preprocessing

env = football_env.create_environment('11_vs_11_easy_stochastic', representation='raw')
observation = env.reset()
smm = observation_preprocessing.generate_smm(observation)
```

GRF's own modules are all there, internals such as `env.unwrapped._env` included, and GRF and `cuju.gfootball` run side by side in one script.

Some code cannot change its imports, for example a library that imports `gfootball` deeply, or a test suite worth running as GRF ships it. The package cuju-gfootball maps the names `gfootball` and `gfootball_engine` onto cuju, so code written for GRF runs as it is.

```sh
pip install cuju-gfootball
```

The alias stays off when a GRF is installed, so the two never fight over the name. `python -m cuju_gfootball` prints which case holds. `pip uninstall cuju-gfootball` removes it.

The extras `gfootball-video`, `gfootball-play`, `gfootball-remote` and `pixels` install OpenCV, pygame, gRPC and GRF's media for GRF's frames, players and remote environments, as in `pip install 'cuju-engine[pixels]'`.

Rendering works through `cuju.gfootball` as in GRF, `render=True`, `env.render()`, `write_video`, `pixels` and `pixels_gray` included. A program in which a person plays shows the frames itself, for example with OpenCV's `imshow`, since cuju's renderer draws into memory.

Importing `cuju.gfootball` registers GRF's gym ids `GFootball-<scenario>-<representation>-v0`, so `gym.make('GFootball-11_vs_11_kaggle-SMM-v0')` works as in GRF.

## GRF's interface in `cuju.env`

`cuju.env.create_environment` takes GRF's arguments with GRF's defaults and returns one environment with GRF's gym interface. After the same `random.seed(...)` it plays the same matches as GRF.

```python
import random

import cuju.env

random.seed(0)
env = cuju.env.create_environment(
    '11_vs_11_easy_stochastic',
    representation='extracted',
    stacked=True,
    rewards='scoring,checkpoints',
)
observation = env.reset()
observation, reward, done, info = env.step(12)
```

Three keyword-only arguments come on top of GRF's, `device`, which defaults to `'cpu'`, `threads`, which defaults to 1, and `api`. `api='gymnasium'` returns Gymnasium's interface, where `reset()` returns `(observation, info)` and `step` returns five values. `import cuju` registers the ids `GFootball-<scenario>-<representation>-v0` with Gymnasium for every shipped scenario, with the representations `SMM`, `simple115` and `simple115v2`.

The environment keeps GRF's negative action indices, `get_state`, `set_state`, `observation()` for GRF's raw dictionaries and `close`. `env.batch` is the batch of one environment underneath. `render=True` and `write_video=True` belong to `cuju.gfootball`, and frames from a batch are in [Observations, rewards and actions](observations-rewards-actions.md).

GRF's wrappers from `cuju.gfootball.env.wrappers` run over this interface too.

## Custom players

GRF loads scripted players from `extra_players`, and so does `cuju.env`. A name other than `agent`, `bot` and `lazy`, as in `extra_players=['my_package.my_player:left_players=1']`, loads that module and builds its class `Player(player_config, env_config)`. Its `take_action(observations)` receives GRF's raw observation of each of its players and returns one action each. `reset()` follows every `env.reset()`. `cuju.gfootball.env.player_base.PlayerBase` is GRF's base class, so a player written for GRF imports it from there.

The `bot` player is GRF's scripted player, needs the `full` action set and keeps its memory across episodes. The `lazy` player plays idle and makes a baseline opponent.

A Python player runs once per step, as in GRF, which suits an opponent in evaluation. For training, an opponent plays as the `right_agents` of a batch.

## `control_all_players`

GRF's `MultiAgentToSingleAgent` wrapper serves scenarios that set `control_all_players`. One policy plays the designated player of each side and the built-in AI plays every other player. The scenario needs every controllable player of the played sides as agents and the `v2` or `full` action set, whose action 19 is `builtin_ai`.

With `extra_players`, such as `['lazy:right_players=11']`, the wrapper covers the agent's players and the extra players act on the other slots. A batch hands the players no agent controls to the built-in AI, and `right_agents` makes a trained policy the opponent of the left agents. Scenario files that set `control_all_players` are in [Scenarios](scenarios.md).

## The batch

The batch replaces a list of GRF environments, or a pool of GRF processes, with one object.

A GRF loop with four agents on `5_vs_5` becomes one batch of 1,024 environments.```python
import cuju

batch = cuju.BatchEnvironment(
    '5_vs_5',
    environment_count=1024,
    left_agents=4,
    device='cpu',
    representation='simple115v2',
    rewards='scoring,checkpoints',
)
observation = batch.reset()
for _ in range(1000):
    actions = policy(observation)  # [1024, 4] actions, a tensor
    observation, reward, done, information = batch.step(actions)
    # No reset here, finished environments have already started the next episode.
    # information['final_observation'][done] holds their last observations.
```

Three things change in a loop. Outputs are tensors that the next step overwrites. Episodes reset inside `step`. Every output has an environment axis, and the agent axis is always present. Arguments, methods and attributes are in [API reference](api.md).

### Options of `create_environment`

| GRF argument | `BatchEnvironment` | Notes |
| --- | --- | --- |
| `env_name` | `scenario`, the first argument | a shipped scenario by name or the path of a scenario file |
| `stacked`, `rewards`, `channel_dimensions` | the same names | |
| `representation` | `representation` | default `simple115v2`, where GRF's is `extracted` |
| `number_of_left_players_agent_controls` | `left_agents` | any count from 0 to the team size |
| `number_of_right_players_agent_controls` | `right_agents` | same |
| `write_goal_dumps`, `write_full_episode_dumps`, `dump_frequency`, `logdir` | `cuju.dumps.DumpWriter` | see [Dumps from a batch](#dumps-from-a-batch) |
| `extra_players` | `right_agents`, or `cuju.env` | the built-in AI plays the sides without agents |
| `other_config_options` | constructor arguments and `config` | see below |

### Keys of `other_config_options`

| GRF key | cuju | Notes |
| --- | --- | --- |
| `action_set` | `action_set` | `default`, `v2`, `full` |
| `game_engine_random_seed` | `reset(seeds=...)` | one seed or one per environment |
| `reverse_team_processing` | `reset(reverse=...)` | read at an environment's first reset |
| `render_resolution_x`, `render_resolution_y` | `frame_size=(width, height)` | width a multiple of 4 |
| `physics_steps_per_frame`, `real_time` and the video options | `cuju.env` and `cuju.gfootball` only | a batch plays GRF's default of 10 steps per frame |

### Methods

| GRF | cuju | Notes |
| --- | --- | --- |
| `env.reset()` | `batch.reset()` | `reset(mask=...)` resets some environments |
| `env.step(action)` | `batch.step(actions)` | actions are `[environment_count, agents]` |
| `env.observation()` | `batch.raw()` | needs `raw=True` |
| `env.get_state(to_pickle)`, `env.set_state(state)` | `batch.get_state(index)`, `batch.set_state(index, state)` | GRF's format, one environment at a time |
| `env.observation_space`, `env.action_space` | `batch.single_observation_space`, `batch.single_action_space` | `batch.observation_space` for the batch |
| `env.render()` | `batch.frame()` | with `frames=True` |
| `info['score_reward']` | `information['score_reward']` | `[environment_count]` |

### Wrappers

A batch computes GRF's wrapper outputs inside the step, so each wrapper becomes an option.

| GRF wrapper | cuju |
| --- | --- |
| `Simple115StateWrapper` | `representation='simple115'` or `'simple115v2'` |
| `SMMWrapper` | `representation='extracted'` |
| `PixelsStateWrapper` | `representation='pixels'` or `'pixels_gray'` |
| `FrameStack` | `stacked=True` |
| `CheckpointRewardWrapper` | `rewards='scoring,checkpoints'` |
| `SingleAgentObservationWrapper`, `SingleAgentRewardWrapper` | `observation[:, 0]` and `reward[:, 0]` |
| `PeriodicDumpWriter` | `cuju.dumps.DumpWriter` |

### Shapes

For one agent per environment, `observation[:, 0]` and `reward[:, 0]` give GRF's shapes with an environment axis in front. The agent axis is always present, and the action space of a batch is `MultiDiscrete([19])` where GRF's is `Discrete(19)`.

### Things to check in a ported loop

| Situation | What to do |
| --- | --- |
| Appending `observation` to a list | each step overwrites the tensor, so append `observation.clone()` or write into a preallocated buffer |
| Bootstrapping after the last step | read `information['final_observation']`, since a finished environment already shows the next episode |
| Default representation | name it, or get `simple115v2` |
| Seeds | the default `seed=0` plays the same matches every run, so pass a different `seed` per run |

## Reproducing GRF results

A result made with cuju compares directly with one made on GRF when five matters agree.

The options. GRF's scripts for the PPO results of its paper play `11_vs_11_easy_stochastic` with one left agent, the stacked SMM, the default action set, and `scoring` or `scoring,checkpoints`. `BatchEnvironment('11_vs_11_easy_stochastic', representation='extracted', stacked=True, rewards='scoring,checkpoints')` is the same setup.

The count of steps. One `env.step` is one step of one match, and papers add them up over parallel environments. One `batch.step` is `environment_count` such steps.

The size of an update. PPO learns from `num_envs × nsteps` steps per update, 16 × 512 = 8,192 in GRF's scripts. A batch of 4,096 environments with the same `nsteps` collects 2,097,152 steps per update, so a curve against steps changes while every match is GRF's. Keep `num_envs × nsteps` and the minibatches to compare curves.

The evaluation. Evaluate against the built-in AI with fixed seeds in `reset(seeds=...)` and report an interval with the win rate. The same seeds play the same matches in GRF.

The version. cuju matches GRF 2.10.3, commit 3d9e754. Published results from older versions compare statistically, over many episodes. See [Exactness](exactness.md).

## Policies, Kaggle and Gymnasium

A policy trained with cuju plays in GRF unchanged, since observations and actions have the same meaning. On a GRF built another way the game is the same, and comparisons are made over many episodes.

A Kaggle football agent receives the raw observation as lists in `obs['players_raw'][0]`. `cuju.env.grf_observation(raw, environment, agent, own_players, other_players)` builds that dictionary from the raw tensors of a batch, and `tolist()` turns its arrays into those lists. GRF's `Simple115StateWrapper.convert_observation` and `observation_preprocessing.generate_smm` then give exactly the `simple115v2` and `extracted` observations of cuju. Kaggle's lists become `float64` arrays in numpy while cuju's raw arrays are `float32`, so cast with `.double()` in training to start from the values the submission sees.

In `11_vs_11_kaggle` the second half starts after step 1,500 with a kick-off by the other team, and the teams keep their ends.

For Gymnasium, use `api='gymnasium'` or the registered ids above. To play against the built-in AI, use a scenario with a side without agents, such as `5_vs_5_l4_r0`. Action 19, `builtin_ai`, of the `v2` and `full` sets hands one agent's player to the built-in AI for a step.

## Dumps

A dump is GRF's file of a match, readable by GRF's tools. The format is the same in cuju and GRF, and a dump written through the cuju-gfootball alias opens in GRF and the reverse.

### Dumps from one environment

`env.write_dump(name)` writes the last 100 steps as `<logdir>/<name>_<time>.dump` and returns the path without `.dump`. Each name writes once an episode, and a second call returns `None`. The options `write_full_episode_dumps` and `write_goal_dumps` write GRF's `episode_done_<time>.dump`, `score_<time>.dump` and `lost_score_<time>.dump`. The step that finishes a dump lists it in `info['dumps']`.

### Dumps from a batch

`cuju.dumps.DumpWriter` writes the matches of chosen environments of a batch as GRF dumps.

```python
import cuju
import cuju.dumps
import torch

batch = cuju.BatchEnvironment(
    '5_vs_5', environment_count=64, left_agents=4, device='cpu', raw=True
)
writer = cuju.dumps.DumpWriter(batch, 'dumps', environments=[0, 1], write_goal_dumps=True)
batch.reset()
for _ in range(3001):
    actions = torch.randint(0, 19, (64, 4), dtype=torch.int32)
    batch.step(actions)
    writer.record(actions)  # after every step, with the same actions
writer.close()  # writes the episodes still running
```

Files are named `episode_done_environment_<i>_episode_<n>.dump`, `score_environment_<i>_episode_<n>_step_<frame>.dump` and `lost_score_...`. `dump_frequency` selects episodes as GRF's `PeriodicDumpWriter` does, and the writer writes every selected episode, with no wall-clock limit. Follow a few environments, since `record` waits for the device at every step. Write the batch with `raw=True` for dumps meant for video, and a dump meant for replay needs no raw observation.

A difficulty that differs from the scenario's names the level `<scenario>_difficulty_<left>_<right>` and writes that scenario file next to the dumps. Copy it into GRF's `gfootball/scenarios/` for GRF's tools to find it.

### Reading, replaying and videos

`cuju.dumps.read_dump(path)` reads any dump, cuju's, GRF's or one of `cuju.gfootball`, and runs no code from it.

```sh
cuju replay match.dump
cuju dump match.dump -o match.txt
```

`cuju replay` plays a dump's actions again and compares every step with the dump. It prints the first difference and exits 1, or the steps found equal and exits 0. It takes dumps of cuju and of GRF that start at the start of an episode. `cuju dump` writes a dump as text. GRF's `dump_to_video`, `replay` and `ScriptHelpers().replay` run in `cuju.gfootball` and render with cuju's renderer, so a video of a dump comes from GRF's `dump_to_video` in `cuju.gfootball`.

A rendering replay plays the matches of render mode, so its video can part from an unrendered match that the dump came from. See [Exactness](exactness.md). Reports of dumps that behave differently in GRF are welcome as issues.

## See also

- [API reference](api.md), every argument, method and error of the batch
- [Observations, rewards and actions](observations-rewards-actions.md), representations, action sets and frames
- [Scenarios](scenarios.md), agents per side and scenario files
- [Exactness](exactness.md), what is checked and where cuju differs from GRF
- [Performance](performance.md), backends, threads and rendering speed
