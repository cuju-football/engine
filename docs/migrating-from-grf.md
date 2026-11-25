# Migrating from GRF

**Migration from GRF** moves code written for Google Research Football, or GRF, onto cuju's engine
through one of three entry points, GRF's own Python package in `cuju.gfootball`, GRF's
`create_environment` in `cuju.env`, and the batch, `cuju.BatchEnvironment`. The matches are the
same in all three, and every option keeps GRF's name. The first two entry points play one
environment per call with GRF's interface, and the batch plays thousands of environments per call
and takes a rewrite of the training loop.

The batch changes three properties of a loop. Its outputs are shared tensors that the next step
overwrites, its episodes reset inside `step` with the last observation in `final_observation`, and
every output carries an environment axis. GRF's dumps, scripted players and scenario configuration
carry over to both kinds of environment. A result made with cuju compares directly with one made on
GRF when five matters agree, the options, the count of steps, the size of an update, the evaluation
and the version of GRF.

## Entry points

cuju has three entry points, which trade compatibility with GRF's code for throughput.

**GRF's Python package, in `cuju.gfootball`.** GRF's own modules, ported verbatim with their import
lines changed, run over cuju's engine. The imports change and the rest of the code stays as it is,
its use of GRF's internals included.

**GRF's `create_environment`, in `cuju.env`.** cuju's own implementation of GRF's function, over a
batch of one environment, takes three keyword-only arguments beyond GRF's, `device`, `threads` and
`api`, the last of which chooses Gymnasium's interface.

**The batch, `cuju.BatchEnvironment`.** Thousands of environments advance per call, and their
outputs arrive as PyTorch tensors. The throughput of cuju lies here, and the loop is written for a
batch.

The first two play one environment per call, as GRF does, so they keep GRF's architecture and the
throughput of one environment at a time. In return, GRF's code runs on them as it stands.
Evaluation scripts, a Kaggle submission, a notebook that inspects raw observations or a test that
compares with published numbers keep working, and code with known results confirms that cuju
reproduces them. `cuju.gfootball` is GRF's code itself and runs everything that GRF's Python does,
at about GRF's speed, since GRF's Python costs about a millisecond a step on its own. `cuju.env`
skips that Python and runs several times as fast. The batch serves the part of a project where the
time goes, the collection of experience for training.

A port usually proceeds in two stages. The imports switch first, and the existing code runs, so that
any difference appears in code whose behaviour is known. The training loop then moves to a batch, a
rewrite of one function, while the rest of the project stays on GRF's interface for as long as it is
convenient. Both kinds of environment play GRF's matches, so the two coexist in one project.

## Import paths

`cuju.gfootball` is GRF's Python package, `gfootball` of GRF 2.10.3, with every module of it,
`gfootball.env`, `config`, `football_env`, `football_env_core`, `wrappers`, `observation_processor`,
`scenario_builder`, the players, the scenarios with their `tests` package, `play_game`, `replay`,
`dump_to_txt`, `dump_to_video`, `eval_server` and `examples`. Each module is GRF's file with its
import lines pointed at `cuju.gfootball`, `cuju.gfootball_engine` and two small modules of cuju, so
the code that runs is GRF's own, line for line. `cuju.gfootball_engine` stands in for GRF's compiled
engine with the same classes, functions, conversions and errors, so GRF's Python finds what it
expects, down to `env.unwrapped._env`, the `_unused_engines` pool and the private attributes of
GRF's classes. Code written for GRF runs unchanged once its imports point there.

```sh
pip install cuju-engine
```

A short program in GRF opens an environment and computes its SMM.

```python
import gfootball.env as football_env
from gfootball.env import observation_preprocessing

env = football_env.create_environment('11_vs_11_easy_stochastic', representation='raw')
observation = env.reset()
smm = observation_preprocessing.generate_smm(observation)
```

With cuju, the same code differs in its imports alone.

```python
import cuju.gfootball.env as football_env
from cuju.gfootball.env import observation_preprocessing

env = football_env.create_environment('11_vs_11_easy_stochastic', representation='raw')
observation = env.reset()
smm = observation_preprocessing.generate_smm(observation)
```

GRF's package needs `absl-py` and `six`, which `pip install cuju-engine` installs. It uses the
installed gym wherever GRF runs on it, gym 0.21, which GRF requires, and every gym up to 0.24, and gym 0.25
with numpy 1. With a later gym, or without one, cuju carries a copy of gym 0.21's core, which serves
in its place, as [Differences from GRF](differences-from-grf.md#grfs-python-package) describes.
OpenCV, which GRF uses for frames and videos, loads at its first use, so training without it works,
and the extra `gfootball-video` installs it together with GRF's media, which cuju's renderer draws
GRF's frames from. The extras `gfootball-play` and `gfootball-remote` install pygame for GRF's game
and keyboard players and gRPC for its remote environments.

These paths run on cuju alone, and with GRF installed as well, the two run side by side in one
script, since cuju's engine module never takes the name of GRF's. Each draws its seeds from Python's
`random`, so two environments in one process shift each other's seeds, as two GRF environments do,
a behaviour covered in [Episodes and seeds](episodes-and-seeds.md).

Importing `cuju.gfootball` registers GRF's ids `GFootball-<scenario>-<representation>-v0` with the
gym it uses, as importing `gfootball` does, so `gym.make('GFootball-11_vs_11_kaggle-SMM-v0')` gives
GRF's environment wherever GRF runs on the installed gym, and `cuju.gfootball._gym.make` gives it
everywhere.

### GRF's names through cuju-gfootball

Some code keeps its imports for good reasons. A library may import `gfootball` deep inside its own
modules, as Kaggle's football environment does, a test suite may be worth running exactly as GRF
ships it, and a dump written by GRF names GRF's modules inside its pickles. The package
cuju-gfootball serves these cases. It maps GRF's top-level names onto the port, `gfootball` and its
submodules onto `cuju.gfootball`, and `gfootball_engine` and `_gameplayfootball` onto
`cuju.gfootball_engine`, so the first program of this section runs on cuju exactly as written for
GRF.

```sh
pip install cuju-gfootball
```

Each of GRF's names is the module of cuju itself, so
`gfootball.env.football_env is cuju.gfootball.env.football_env` holds, and code that mixes both
spellings shares one module and one state. `importlib.util.find_spec('gfootball')` and
`from gfootball.scenarios import *` work as in GRF, and so do gym's ids of the form
`gfootball:GFootball-11_vs_11_kaggle-SMM-v0` on the gyms that accept that form for GRF, gym 0.21,
0.24 and 0.25 with numpy 1. Gym 0.22 and 0.23 refuse the form for GRF's own package as well, and on
them `import gfootball` followed by `gym.make('GFootball-11_vs_11_kaggle-SMM-v0')` gives the
environment, as it does on every gym that GRF runs on. The specification of an id names the entry
point `cuju.gfootball.env:create_environment`, the same function as GRF's
`gfootball.env:create_environment` through the alias.

The classes of the port report GRF's module names under the alias. `CoreAction` reports
`gfootball.env.football_action_set`, `EnvState` reports `gfootball.env.football_env_core`, and the
engine's enumerations report `_gameplayfootball`, which are the names that Python's `pickle`
writes. A dump written through the alias therefore opens in GRF, and a dump written by GRF opens in
the port, so dumps move freely between machines that run GRF and machines that run cuju, a
convenience for a project that keeps an old evaluation pipeline on GRF while it trains on cuju.
Without the alias each side writes names that the other cannot import, as
[Differences from GRF](differences-from-grf.md#grfs-python-package) describes, while cuju's own
readers, `cuju.dumps.read_dump`, `cuju replay` and `cuju dump-to-txt`, read every dump with or
without it.

GRF's own tests run through the alias exactly as GRF ships them, every file byte for byte, and
cuju's testing runs them on each gym that GRF supports. Testing is internal, and it also opens dumps
in both directions between the alias and GRF 2.10.3 and plays Kaggle's football environment end to
end on the alias, its videos and the private trace that it reads at the end of an episode included,
with the same matches as on GRF. Reports of code that behaves otherwise on the alias are welcome as
issues.

The alias activates through a file `cuju_gfootball.pth`, which Python runs at startup, and decides
at the first import of one of GRF's names or of the port. The decision concerns GRF itself. The
name `gfootball` belongs to an installed GRF, whether a regular or editable install, a folder on
`PYTHONPATH` or the installed distribution `gfootball`, so beside any of them the alias activates
nothing, the port keeps its own module names, and one `RuntimeWarning` names what it found.
`python -m cuju_gfootball` prints which case holds and exits with 1 when the alias is inactive. GRF
and cuju then run side by side in one script through `gfootball` and `cuju.gfootball`, as without
the alias, and uninstalling either one of GRF and cuju-gfootball settles the name at the next start
of Python. `pip uninstall cuju-gfootball` removes the alias and leaves cuju as it was.

### Rendering and GRF's interactive features

Every feature of GRF's Python works through these paths, rendering included. cuju's software
renderer draws GRF's frames for `render=True`, `env.render()`, `write_video` with `render`, the
representations `pixels` and `pixels_gray` and `PixelsStateWrapper`, byte for byte as GRF draws them
on its reference, Mesa 25.2.8's llvmpipe, once the extra `pixels` has installed GRF's media.

```sh
pip install 'cuju-engine[pixels]'
```

Code that renders in GRF therefore renders in `cuju.gfootball` as it stands, and the section on
[pixel observations and frames](#pixel-observations-and-frames) discusses its move to a batch. A
few features have points worth knowing.

| GRF | In cuju | Note |
| --- | --- | --- |
| the `keyboard` and `gamepad` players | GRF's own players, in `cuju.gfootball` with the extra `gfootball-play` | cuju's renderer draws into memory, so a program in which a person plays shows the frames itself, for example with OpenCV's `imshow` |
| `real_time` | GRF's schedule, in `cuju.gfootball` | the renders per step follow the wall clock, in GRF and in cuju alike, so the matches depend on the speed of the machine |
| `render_resolution_x`, `render_resolution_y` | GRF's options, in `cuju.gfootball` and in a batch | any height and a width that is a multiple of 4, with the frames tested at nine sizes |
| `create_remote_environment` | GRF's client in `cuju.gfootball.env.remote_football_env`, with the extra `gfootball-remote` | a client of GRF's online game server, whose availability is unknown |

A rendering GRF plays other matches than an unrendered one, as listed in
[Differences from GRF](differences-from-grf.md), and cuju's engine plays the rendering ones
whenever it renders. Every result trained on GRF's vector or SMM observations came from an
unrendered match, which a batch plays by default, and every result trained on GRF's pixels came
from a rendering match, which a batch with `pixels` plays. `write_video` with `render` off writes
GRF's two-dimensional view of the match, which GRF's Python draws with OpenCV from the extra
`gfootball-video`, and Kaggle's videos are of this kind.

## GRF's interface for one environment

Install the `spaces` extra with `pip install 'cuju-engine[spaces]'` before using
`cuju.env` in an environment without Gymnasium or GRF's gym. This also applies
when you use the single-environment interface to write full episode dumps.

`cuju.env.create_environment` takes GRF's arguments with GRF's defaults and returns one environment
with GRF's interface, the one of gym 0.21 that GRF 2.10.3 is written for. `reset()` returns the
observation, and `step(action)` returns `(observation, reward, done, info)`. It plays the same
matches as GRF's after the same `random.seed(...)`, since it draws its seeds from Python's `random`
as GRF does, and as `cuju.gfootball.env.create_environment` does. It runs GRF's episode logic in
cuju's engine rather than in GRF's Python, which makes it several times as fast, at the cost of
GRF's internals. An environment with a stacked SMM and checkpoint rewards takes one step in a few
lines.

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

Three keyword-only arguments come on top of GRF's, `device`, which is `'cpu'` by default,
`threads`, which is 1, and `api`. The CPU is the default because one environment occupies one warp
of a GPU and leaves the rest of the device idle, while on the CPU one environment already runs
several times as fast as a GRF process.

`api='gymnasium'` returns Gymnasium's interface instead, where `reset()` returns
`(observation, info)` and `step` five values. `import cuju` registers the ids
`GFootball-<scenario>-<representation>-v0`, with the representations `SMM`, `simple115` and
`simple115v2`, for every shipped scenario, with Gymnasium as soon as it is imported, and
`gymnasium.make('GFootball-11_vs_11_kaggle-simple115v2-v0')` gives that interface. gym gets GRF's
own environments under the same ids from `import cuju.gfootball`. gym's interface suits code
written for GRF, and Gymnasium's suits libraries that expect it.

`get_state`, `set_state`, `observation()`, which returns GRF's raw dictionaries of the agent's
players, and `close` work as GRF's do, and `env.batch` is the batch of one environment underneath,
for access to a batch method from GRF's interface. With `representation='raw'`, `observation_space`
is `None`, as GRF's raw observation has no space. `seed` returns `None` and `reward_range` is
unbounded, as gym's `Env` gives them to GRF's environment, and `disable_render` does nothing.
`render=True`, `write_video=True` and the `Pixels` ids raise `ValueError`, since GRF's rendering
takes place in `cuju.gfootball`, as [Frames and videos](rendering.md#grfs-rendering-in-cujugfootball)
describes, and in a `BatchEnvironment` with `representation='pixels'` or `render_frames=True`, as
[Rendering batches](rendering.md#rendering-batches) describes.

The environment is a Gymnasium `Env` whose `reset` and `step` return gym's values. Where a gym
older than 0.26 is installed, as GRF requires, the environment is also gym's `Env` and its spaces
are gym's, so code written for GRF checks `isinstance(env, gym.Env)` and
`isinstance(space, gym.spaces.Box)` as it did. With only Gymnasium installed, the spaces are
Gymnasium's. `api='gymnasium'` always gives Gymnasium's spaces. GRF's wrappers from
`cuju.gfootball.env.wrappers` run over this interface too.

This interface serves existing code, the check of a port and comparisons, and the batch serves
training. A loop over many such environments, one per process as GRF is usually scaled, combines
GRF's architecture with cuju's engine and leaves most of the batch's throughput unused.

### Dumps and replays

GRF writes a dump on a call to `env.write_dump(name)`, and so does cuju's environment. It writes at
most the last 100 steps of the episode as a GRF dump, `<logdir>/<name>_<time>.dump` with the date and
time in GRF's format, and returns the path without `.dump`, as GRF's does. Each name writes once an
episode, and a second call returns `None`. After the episode ends it raises `RuntimeError` until
`reset()`, where GRF raises `AttributeError`.

The options `write_full_episode_dumps` and `write_goal_dumps` write GRF's dumps as well, with GRF's
names, `<logdir>/episode_done_<time>.dump` for an episode and `score_<time>.dump` or
`lost_score_<time>.dump` for a goal, each with the time at which GRF starts the dump. The step that
finishes a dump lists it in `info['dumps']`, as GRF does, with one dictionary for each dump, such as
`{'dump': '/tmp/dumps/episode_done_20261027-142233123456.dump', 'name': 'episode_done'}`, and a goal
that ends an episode lists its goal dump first.

A dump holds every observation of the steps it covers, and keeping all of them during play would
cost memory at every step for a file written only on occasion. The environment therefore keeps the
actions of each step alone, and plays them again in a batch of one environment to write the dump on
request. The replay of a long episode takes a moment on the CPU, and the `time` of each step in the
dump is the time of that replay rather than of the original step. The match itself is the same, bit for bit.

A dump can also be checked without rendering. `cuju replay --trace_file=<dump>`
plays a dump's actions again from its configuration and compares every step with the dump, bit for
bit. It prints the first difference and exits 1, or the steps found equal and exits 0. It takes
dumps of cuju and of GRF that start at the start of an episode, and confirms that a dump written by
GRF on a given machine plays the same match in cuju. GRF's own `replay` and `ScriptHelpers().replay`
in `cuju.gfootball` render with cuju's renderer, as GRF's do with its own, and its `dump_to_video`
turns a dump into a rendered video.

### Custom players

GRF loads scripted players from `extra_players`, and so does cuju. A name other than `agent`, `bot`
and `lazy`, as in `extra_players=['<name>:left_players=1']`, loads
`cuju.gfootball.env.players.<name>`, else the module `<name>` itself, such as
`my_package.my_player`, and builds its class `Player(player_config, env_config)`, as GRF does. Its
`take_action(observations)` receives GRF's raw observation of each of its players and returns one
action each, and `reset()` follows every `env.reset()`. `cuju.gfootball.env.player_base.PlayerBase`
is GRF's base class, so a player written for GRF imports it from there, and the `replay` player
plays a dump's actions. In `cuju.env`, a missing module or class, or a wrong number of actions,
raises `ValueError`, and in `cuju.gfootball` such a player fails as it does in GRF.

Such a player runs in Python once per step, as in GRF, which suits an opponent in evaluation. For
training, an opponent plays as agents of a batch, or as the ported
[`bot` and `lazy` players](#scripted-players-and-control_all_players), which act for every
environment at once.

### Scenario configuration

`cuju.gfootball.env.config.Config` is GRF's, so `Config.ScenarioConfig()` builds the scenario of
the current episode with GRF's `scenario_builder`, from GRF's scenario files, which
`cuju.gfootball.scenarios` holds, and returns GRF's `ScenarioConfig` with `float32` positions, as
GRF's engine stores them. Code that reads a scenario's settings, such as the length of a half or
whether it ends on a goal, keeps working. A dump written after 150 steps and the start of the second
half of `11_vs_11_kaggle` each take a line.

```python
import cuju.env
from cuju.gfootball.env import config

env = cuju.env.create_environment('11_vs_11_easy_stochastic', logdir='/tmp/dumps')
env.reset()
for _ in range(150):
    env.step(0)
path = env.write_dump('shutdown')  # '/tmp/dumps/shutdown_<time>'

configuration = config.Config({'level': '11_vs_11_kaggle'})
configuration.ScenarioConfig().second_half  # 1500
```

GRF's `Config` takes a level that names a module of `cuju.gfootball.scenarios`, as GRF's takes one of
`gfootball.scenarios`, and ends the process for any other, as GRF's does. The levels that name a
bundle or the path of a scenario file belong to `cuju.env` and the batch.

## Batched training loop

The batch replaces a list of GRF environments, or a pool of GRF processes, with one object. The
mapping of names is mostly mechanical. A port centres on three changes, which the
[pitfalls of a port](#pitfalls-of-a-port) discuss in detail. Outputs are shared buffers, the episode resets
inside the step, and every output has an environment axis.

### Loop comparison

With GRF, one environment runs per process.

```python
import gfootball.env as football_env

env = football_env.create_environment(
    env_name='5_vs_5',
    representation='simple115v2',
    rewards='scoring,checkpoints',
    number_of_left_players_agent_controls=4,
)
observation = env.reset()
for _ in range(1000):
    action = policy(observation)  # [4] actions
    observation, reward, done, info = env.step(action)
    if done:
        observation = env.reset()
```

With cuju, 1,024 environments run in one process.

```python
import cuju
import torch

batch = cuju.BatchEnvironment(
    '5_vs_5_l4_r0',
    environment_count=1024,
    device='cpu',
    representation='simple115v2',
    rewards='scoring,checkpoints',
)
observation = batch.reset()
for _ in range(1000):
    actions = policy(observation)  # [1024, 4] actions, a tensor
    observation, reward, done, information = batch.step(actions)
    # No reset: environments whose episode ended have already started the next one.
    # information['final_observation'][done.bool()] holds their last observations.
```

The policy receives 1,024 observations at once, the batched input that a neural network processes
in one pass, and the `if done` branch disappears, since the batch resets its own environments. The
number of agents moves into the name, `_l4_r0`, the way a batch names its agents, as described in
[Scenarios](scenarios.md).

### Arguments of `create_environment`

Every argument of GRF's `create_environment` has a place in the batch. Most keep their names, and
the agent counts move into the scenario's name or `agent_counts`.

| GRF argument | `BatchEnvironment` | Notes |
| --- | --- | --- |
| `env_name` | the first argument | a shipped scenario by name, a bundle by path, or a scenario file |
| `stacked` | `stacked` | same |
| `representation` | `representation` | same names, the default is `simple115v2` where GRF's is `extracted`, and `pixels` and `pixels_gray` render, as in [Pixel observations and frames](#pixel-observations-and-frames) |
| `rewards` | `rewards` | same, `scoring` or `scoring,checkpoints` |
| `write_goal_dumps`, `write_full_episode_dumps`, `dump_frequency`, `logdir` | `cuju.dumps.DumpWriter` | written on request, outside the step, as in [Dumps from a batch](#dumps-from-a-batch) |
| `render`, `write_video` | `render_frames=True` for the frames, and `write_video` in `cuju.gfootball` | a batch returns frames, and GRF's videos come from GRF's Python |
| `extra_players` | `cuju.players.Bot`, `cuju.players.Lazy` | GRF's `bot` and `lazy` players, ported, as in [Scripted players](#scripted-players-and-control_all_players) |
| `number_of_left_players_agent_controls` | the `_l<left>` of the name, or `agent_counts` | same meaning, any count from 0 to the team size |
| `number_of_right_players_agent_controls` | the `_r<right>` of the name, or `agent_counts` | same meaning |
| `channel_dimensions` | `channel_dimensions` | same |
| `other_config_options` | the keyword arguments of the constructor | flattened into the constructor |

The one default that changes is the representation. The batch defaults to `simple115v2` because
most training on GRF reads a vector, and an SMM of thousands of environments takes gigabytes, as
measured in [Observations](observations.md). A port that names its representation is unaffected by
the default.

### `other_config_options`

GRF passes its lower-level options in a dictionary. The batch takes them as keyword arguments, with
the same names where they mean the same thing.

| GRF key | cuju | Notes |
| --- | --- | --- |
| `action_set` | `action_set` | same, `default`, `v2`, `full` |
| `physics_steps_per_frame` | `physics_steps_per_frame` | same |
| `game_engine_random_seed` | `game_engine_random_seed` | same, and a sequence sets one per environment |
| `episode_number` | `episode_number` | same |
| `reverse_team_processing` | `reset(reverse=...)` | the team-order flag, read at an environment's first reset |
| `level` | the first argument | the scenario |
| `players` | the agent counts, `cuju.players.Bot`, `cuju.players.Lazy`, and custom players through `create_environment` | the `keyboard` player plays in a GRF that renders |
| `real_time` | the speed of the hardware | a batch runs as fast as the hardware allows |
| `display_game_stats`, `custom_display_stats` | GRF's options in `cuju.gfootball` | GRF's Python draws them over its videos |
| `render_resolution_x`, `render_resolution_y` | `render_resolution_x`, `render_resolution_y` | same, the size of a rendering batch's frames, with a width that is a multiple of 4 |
| `video_format`, `video_quality_level`, `write_video` | GRF's options in `cuju.gfootball` | GRF's Python writes its videos |
| `dump_full_episodes`, `dump_scores`, `tracesdir` | `cuju.dumps.DumpWriter` | as in [Dumps from a batch](#dumps-from-a-batch) |

`reverse_team_processing` is the only option whose place changes in meaning. In GRF it is drawn from
the seed and set by hand mainly to replay a run, so the batch takes it at the reset where a replay
happens, as covered in [Episodes and seeds](episodes-and-seeds.md).

### Environment methods

The methods map one to one, with an environment axis added to everything they take and return.

| GRF | cuju | Notes |
| --- | --- | --- |
| `env.reset()` | `batch.reset()` | resets every environment, `reset(mask=...)` some, and `reset(seeds=..., reverse=...)` replays chosen seeds |
| `env.step(action)` | `batch.step(actions)` | `actions` is `[environment_count, agents]`, and it returns `(observation, reward, done, information)` for the whole batch |
| `env.observation()` | `batch.observation()` | the current observation |
| `env.get_state(to_pickle)` | `batch.get_state(index)` | GRF's format, as in [States](states.md) |
| `env.set_state(state)` | `batch.set_state(index, state)` | GRF's format, in both directions |
| `env.close()` | `batch.close()` | frees the buffers |
| `env.observation_space` | `batch.single_observation_space` | `batch.observation_space` for the batch |
| `env.action_space` | `batch.single_action_space` | `batch.action_space` for the batch |
| `env.render()`, `env.disable_render()` | `render_frames=True` and `batch.frame()` | a batch renders from its creation, every environment alike |
| `env.write_dump(name)` | `env.write_dump(name)` in `cuju.env`, `cuju.dumps.DumpWriter` for a batch | the last steps of an episode, or whole episodes and goals |
| `env.tracker_setup(start, end)` | `get_state` at the chosen steps, or GRF's tracker in `cuju.gfootball` | GRF's tracker compares two engines of one process in lockstep |
| `info['score_reward']` | `information['score_reward']` | same, `[environment_count]` |
| `len(obs['left_team'])`, `len(obs['right_team'])` | `batch.left_players`, `batch.right_players` | per environment, from a left agent's view |

States are per environment, so `get_state` and `set_state` take an index. They copy between the
device and the host, which makes them the one part of the mapping to keep out of the hot loop.

### Wrappers

GRF builds its observations by wrapping the environment. A batch computes them inside the step, on
the device, so each wrapper becomes an option. The values are the same, and the cost of a wrapper
in Python disappears.

| GRF wrapper | cuju | Notes |
| --- | --- | --- |
| `Simple115StateWrapper` | `representation='simple115'` or `'simple115v2'` | computed inside the batch |
| `SMMWrapper` | `representation='extracted'` | computed inside the batch |
| `PixelsStateWrapper` | `representation='pixels'` or `'pixels_gray'` | drawn by cuju's renderer, the same frame for every agent |
| `FrameStack` | `stacked=True` | four observations, the oldest first |
| `CheckpointRewardWrapper` | `rewards='scoring,checkpoints'` | same reward, same state key |
| `SingleAgentObservationWrapper`, `SingleAgentRewardWrapper` | `squeeze=True` | drops the agent axis for one agent, which GRF does by itself when one agent plays and cuju only when asked |
| `GetStateWrapper` | built in | `get_state` and `set_state` are methods of the batch |
| `PeriodicDumpWriter` | `cuju.dumps.DumpWriter` | as in [Dumps from a batch](#dumps-from-a-batch) |
| `MultiAgentToSingleAgent` | `cuju.wrappers.MultiAgentToSingleAgent` | for scenarios that set `control_all_players`, as in [Scripted players](#scripted-players-and-control_all_players) |

GRF's classes themselves are in `cuju.gfootball.env.wrappers` for code on GRF's interface, where
they wrap one environment at a time, as GRF's do. Over one environment `MultiAgentToSingleAgent`
gives the players that the policy does not move GRF's `builtin_ai` action, which an environment
plays with the action set `v2` or `full`, so the environment under it takes
`other_config_options={'action_set': 'v2'}`, whose first 19 actions are those of `default`.

A custom wrapper that transforms observations ports best as a function applied to the batch's
tensors after each step, where it runs once for all environments rather than once per environment.

### Gym names

The batch spells out a few names that Gym abbreviates. The objects are the same.

| Gym and GRF | cuju |
| --- | --- |
| `env` | `batch`, a `BatchEnvironment` |
| `num_envs` | `environment_count` |
| `obs` | `observation` |
| `info` | `information` |
| `final_obs` (Gymnasium's vector environments) | `information['final_observation']` |

### Shapes

The outputs of a batch carry an environment axis in front of GRF's shapes.

| Output | GRF, one agent | GRF, several agents | cuju |
| --- | --- | --- | --- |
| observation (`simple115v2`) | `[115]` | `[agents, 115]` | `[environment_count, agents, 115]` |
| reward | scalar | `[agents]` | `[environment_count, agents]` |
| done | bool | bool | `[environment_count]` of `uint8` |
| action | int | `[agents]` | `[environment_count, agents]` |

GRF drops the agent axis by itself when one agent plays, and cuju keeps it unless `squeeze=True` is
passed. cuju keeps it by default because a batch can mix scenarios with different numbers of
agents, and code whose agent axis comes and goes is a common source of silent broadcasting bugs.
With `squeeze=True` and one agent per environment, cuju's shapes become `[environment_count, 115]`,
`[environment_count]` and `[environment_count]`, GRF's single-agent shapes with an environment axis,
and the spaces become GRF's for one agent, such as `Discrete(19)` in place of `MultiDiscrete([19])`.

## Dumps from a batch

`cuju.dumps.DumpWriter` writes GRF's dump files for chosen environments of a batch. They are the
same pickles, with the same keys and values, that GRF's `write_full_episode_dumps` and
`write_goal_dumps` write, so GRF's tools read them, and GRF's `dump_to_video` turns them into
videos. A writer that follows two environments of a batch for one match writes their episodes and
goals.

```python
import cuju
import cuju.dumps
import torch

batch = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=64, device='cpu', with_raw=True)
writer = cuju.dumps.DumpWriter(batch, 'dumps', environments=[0, 1], write_goal_dumps=True)
batch.reset()
for _ in range(3001):
    actions = torch.randint(0, 19, (64, 4), dtype=torch.int32)
    batch.step(actions)
    writer.record(actions)  # after every step, with the same actions
writer.close()  # writes the episodes still running
print(writer.paths)
```

`record(actions)` follows every step. It reads the batch's buffers after the step and leaves every
match as it plays. It waits for the device and copies the rows of the followed environments alone, so on a GPU
it adds a wait at every step, and its cost grows with the environments it follows. A writer
therefore follows a few chosen environments.

The files are `episode_done_environment_<i>_episode_<n>.dump`, and for goals
`score_environment_<i>_episode_<n>_step_<frame>.dump` and `lost_score_...`. GRF names a dump by its
kind and the time to the microsecond, which serves one environment, and the environments of a batch
can start dumps in the same microsecond, so the writer of a batch names each dump by its
environment and episode instead. `dump_frequency` selects
episodes as GRF's `PeriodicDumpWriter` does. GRF also limits dumps by wall-clock time, to at most
one full episode every 10 seconds, a limit sized for GRF's speed. A batch plays many episodes a
second, so the writer writes every selected episode, and `dump_frequency` is the setting that bounds
the space taken on disk.

The steps hold GRF's observation only if the batch writes `raw`, through `with_raw=True` or
`representation='raw'`. GRF's `replay` needs none, and `dump_to_video` does. `with_raw=True`
therefore serves dumps meant for video, and a dump meant for replay needs no raw observation.

An episode whose difficulty differs from its scenario's names its level
`<scenario>_difficulty_<left>_<right>` and writes that GRF scenario file next to the dumps. GRF's
tools find it once it is copied into GRF's `gfootball/scenarios/`, since GRF looks scenarios up by
name. `cuju.dumps.read_dump(path)` reads a dump, cuju's, GRF's or one of `cuju.gfootball`, without
GRF or numpy, and runs no code from it, which makes it safe for dumps from other sources.

In GRF and in `cuju.gfootball`, `ScriptHelpers().replay(path, render=False)` replays a dump of
cuju bit for bit, with GRF's default `physics_steps_per_frame=10`, and `cuju replay` checks one
without GRF. A rendering replay plays the matches of render mode, so its video can part from an
unrendered match that the dump was written from, in GRF and in `cuju.gfootball` alike. A batch with
`render_frames=True` gives the frames of the match that it plays itself.

## Pixel observations and frames

Code that trains or evaluates on GRF's pixels moves to cuju along the same two stages as any other
code. With its imports pointed at `cuju.gfootball`, it runs as it stands and draws GRF's frames
through cuju's software renderer, so a policy trained on GRF's pixels sees in `cuju.gfootball` the
pictures it would see on GRF's reference. Its training loop then moves to a batch with
`representation='pixels'` or `'pixels_gray'`, which draws the same frames for many environments at
once, each on a worker of its own.

The frames are those of one reference, GRF on Mesa 25.2.8's llvmpipe, as
[Frames and videos](rendering.md#frames-equal-to-grfs) explains. A GRF that rendered with a GPU's driver
drew slightly different pixels, so a policy trained there meets in cuju the same pictures up to
small differences of shading and filtering, much as it would meet them on another GRF machine, and
a comparison with its published numbers is statistical, over many episodes. GRF 3d9e754 drawing
with Mesa 25.2.8's llvmpipe and LLVM 20.1.2 on an AMD processor of the Zen 3 generation drew cuju's
frames exactly.

### The batch for pixel code

A loop over GRF environments with pixels maps onto a batch with few changes.

```python
import gfootball.env as football_env

env = football_env.create_environment('11_vs_11_stochastic', representation='pixels', render=True)
observation = env.reset()  # [72, 96, 3]
```

With cuju, eight such environments run in one batch.

```python
import cuju

batch = cuju.BatchEnvironment(
    '11_vs_11_stochastic',
    environment_count=8,
    device='cpu',
    representation='pixels',
    squeeze=True,
)
observation = batch.reset()  # [8, 72, 96, 3]
```

The options and outputs of GRF's pixel code map as the table shows.

| GRF | cuju | Notes |
| --- | --- | --- |
| `representation='pixels'`, `render=True` | `representation='pixels'` | a batch with pixels renders by itself, so `render` has no counterpart |
| `representation='pixels_gray'` | `representation='pixels_gray'` | `[..., 72, 96, 1]`, as in GRF |
| `channel_dimensions` | `channel_dimensions` | same, the size of the shrunk frame |
| `stacked=True` with pixels | a stack kept by the training code | the batch raises `ConfigurationError` for `stacked=True` with pixels, and [Training on pixels](rendering.md#training-on-pixels) shows a stack of four |
| `observation['frame']` of `raw` | `information['frame']` with `render_frames=True`, or `batch.frame()` | `[environment_count, render_resolution_y, render_resolution_x, 3]`, rows from the top, red, green, blue |
| `env.render(mode='rgb_array')` | `information['frame']` with its channels reversed | GRF's `render` reverses the channels of the observation's frame, and the batch returns the observation's order |
| `render_resolution_x`, `render_resolution_y` | `render_resolution_x`, `render_resolution_y` | same, and `pixels` and `pixels_gray` shrink the whole frame, as GRF's wrapper does |
| `write_video=True`, `dump_to_video` | GRF's, in `cuju.gfootball`, or a video written from `information['frame']` | GRF's video also holds the frames of the internal steps, which a batch does not draw |

### Points to check in a pixel port

**Matches of render mode.** A batch with pixels plays the matches of a rendering GRF, as every GRF
with pixels did, so the same seed in a batch with `simple115v2` plays other matches, which part once
the referee acts out a stoppage. A port that compares pixel runs with vector runs compares the game,
not the matches.

**Overwritten frames.** The pixel observations and the frames are buffers that the next step
overwrites, as every output of a batch is, so a rollout buffer or a video writer copies them at
each step.

**The cost.** A frame costs a few hundred milliseconds of processor time, so a batch on a laptop
draws about ten frames a second in all and each environment holds about 200 MB.
[Frames and videos](rendering.md#speed) gives the measured figures, and a port plans its batch from them
rather than from the speed of the simulation.

**The GPU.** On the GPU the simulation runs on the device and the frames are drawn on the host and
copied to the device after each step, so a batch with pixels refuses `step_async` and
`torch.compile`, and a loop that used them steps with `step`.

## Scripted players and `control_all_players`

GRF ships two scripted players and a wrapper for scenarios in which one policy controls a whole
team. cuju ports all three to act on every environment of a batch at once.

**`cuju.players.Bot(batch, agent)`** is GRF's `bot` player, ported exactly in float64. `act()`
returns its actions, an index of the `full` set, for every environment at once. It needs
`action_set='full'`, as GRF's bot asserts, and raw arrays. Like GRF's, it keeps its memory across
episodes. It provides a scripted opponent besides the built-in AI, for variety in training and
evaluation.

**`cuju.players.Lazy(batch, agents)`** is GRF's `lazy`, which plays `idle`. A policy that has
learned to score beats a team that stands still, which makes `Lazy` a baseline opponent for any
policy.

**`cuju.wrappers.MultiAgentToSingleAgent(batch, left_players, right_players)`** is GRF's wrapper for
scenarios that set `control_all_players`. A policy plays the designated player of each side, and the
built-in AI plays every other agent. The batch needs every controllable player of the played sides
as agents, raw arrays and the `v2` or `full` set, for `builtin_ai`. A batch accepts such a scenario
when each side has either no agent or one per controllable player, as GRF's engine asserts.

`create_environment` uses all of them as GRF does, from `extra_players` such as
`['bot:right_players=1']` and from a scenario file's `control_all_players`, so code that names them
there runs unchanged. The two combine as in GRF. With `control_all_players`, the wrapper covers the
players of the agent, and the players of `extra_players`, such as `['lazy:right_players=11']`, play
the other slots with their own actions.

## Pitfalls of a port

Most pitfalls of a port raise no error, and their effects appear in the results of training, so
each is worth checking once in a ported loop.

**Storing observations.** Code that appends `obs` to a list stores the same tensor many times,
since each step overwrites it. `observation.clone()`, or a write into a preallocated buffer, keeps
each step.

**The done flag.** The reset happens inside `step`, and the observation of a done environment is
the next episode's. Bootstrapping code that reads the observation after the last step reads
`information['final_observation']` instead.

**Time limits.** As in GRF, the end of the match time is a normal end of the episode, signalled by
the done flag alone.

**The default representation.** Code that relied on GRF's default `extracted` gets `simple115v2`
until it names the representation.

**Padded agents.** In a mixed batch, rewards and losses averaged over agents are first masked with
`batch.agent_mask`, so that the zeros of padded slots stay out of the average.

**Controllable players of `5_vs_5`.** Four players per team are controllable in `5_vs_5`, since the
scenario marks its goalkeepers as fixed to the built-in AI, as in GRF.

**One agent.** GRF drops the agent axis by itself when one agent plays, and cuju keeps it.
`BatchEnvironment('academy_empty_goal_close')` gives `[environment_count, 1, 115]` and
`MultiDiscrete([19])`, and `squeeze=True` gives GRF's shapes and spaces.

**`done` as `uint8`.** PyTorch deprecates indexing with a `uint8` mask, so the index is written
`final_observation[done.bool()]` in place of `final_observation[done]`. With
`representation='raw'`, `final_observation` is a dictionary of tensors, each indexed in turn, as
`{key: value[done.bool()] for key, value in final_observation.items()}`.

**The number of environments.** `environment_count` defaults to 1,024, so a ported call without it
creates 1,024 environments.

**The same matches in every run.** The default `seed=0` plays the same matches every time, where an
unseeded GRF draws different ones in each run. A different `seed` per run gives different matches.

**gym and gymnasium.** The spaces of a batch come from gymnasium whenever it is installed, even when
gym is installed too, so a check such as `isinstance(space, gym.spaces.Box)` is written against
`gymnasium.spaces.Box`.

**GRF's PPO example.** `gfootball/examples/run_ppo2.py` runs OpenAI Baselines, whose dependencies
install on Python 3.7 or older, and cuju runs on Python 3.10 or later. The learner is therefore
ported to PyTorch or kept in a process of its own.

## Reproduction of GRF results

A result made with cuju compares directly with one made on GRF when the setup matches. Exactness
makes such comparisons possible, and a batch can still change the shape of training while every
match is GRF's. Five matters decide whether the setups match.

**The options.** GRF's scripts that reproduce the PPO results of its paper,
`gfootball/examples/repro_scoring_easy.sh` and `repro_checkpoint_easy.sh`, play
`11_vs_11_easy_stochastic` with one agent on the left, the stacked SMM, the default action set, and
`scoring` or `scoring,checkpoints`. cuju expresses the same setup as one batch, and the bare name
gives one left agent, as GRF's `create_environment` does.

```python
batch = cuju.BatchEnvironment(
    '11_vs_11_easy_stochastic',
    environment_count=16,
    device='cpu',
    representation='extracted',
    stacked=True,
    rewards='scoring,checkpoints',
    squeeze=True,
)
```

**Counting steps.** GRF counts environment steps. One `env.step` is one step of one match, however
many agents play in it, and the step counts of papers, such as the `num_timesteps` of Baselines'
PPO, add them up over the parallel environments. One `batch.step` is therefore `environment_count`
such steps, and the agents of each environment add nothing to the count.

**The size of an update.** PPO learns from `num_envs × nsteps` steps per update, 16 × 512 = 8,192
in GRF's scripts. A batch of 4,096 environments with the same `nsteps` collects 2,097,152 steps per
update, so the same number of steps gives 256 times fewer updates, and a learning curve plotted
against steps changes while every match is GRF's. A comparison of curves keeps `num_envs × nsteps`
and the minibatches.

**The evaluation.** Evaluation against the built-in AI uses fixed seeds, given to
`reset(seeds=...)`, and reports an interval with the win rate. The same seeds play the same matches
in GRF.

**The version.** Exactness holds against GRF 2.10.3, commit 3d9e754. Many published results were
made on older versions of GRF, and comparisons with those are statistical, over many episodes,
since cuju's match-by-match agreement is established against 2.10.3.

A larger update makes a different experiment. A batch makes it cheap to collect far more experience
per update than GRF's scripts did, since the extra environments cost little, and the larger update
can improve the policy reached within a given time. A reproduction of a curve keeps the update at
GRF's size and gains in wall-clock time alone. A search for the best policy within a time budget
changes the size of the update on purpose and reports the change.

## See also

- [Differences from GRF](differences-from-grf.md), every behaviour that differs beyond the names
- [Episodes and seeds](episodes-and-seeds.md), the automatic reset, seeds and `reverse_team_processing`
- [States](states.md), `get_state` and `set_state` in GRF's format
- [Scenarios](scenarios.md), the naming of agents and batches that mix scenarios
- [Custom scenarios](custom-scenarios.md), scenario files and `control_all_players`
- [Recipes](recipes.md), patterns for wiring a batch into training
