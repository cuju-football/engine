# API reference

The **application programming interface** of cuju comprises the Python package `cuju`, whose public
class is `cuju.BatchEnvironment`, the machine config that holds the settings of speed, the command
`cuju`, and a C API that both native libraries export. `cuju.BatchEnvironment` creates a batch,
steps and resets it, exposes its raw observations and states, and raises the exceptions of
`cuju.errors`. Submodules add the interfaces and tools of Google Research Football, or GRF, around
it.

One rule decides where an option lives. An option that changes results is an argument of the
constructor, and an option that changes only speed is a key of the machine config. A call such as
`cuju.BatchEnvironment('5_vs_5', environment_count=1024, seed=7)` therefore names everything that
its matches depend on, and two runs of the same call play the same matches on every machine. In the
shapes of tensors, `E` stands for `environment_count` and `A` for `batch.agents`.

## BatchEnvironment

`cuju.BatchEnvironment(scenario, environment_count=1024, device='auto', seed=0, left_agents=1, right_agents=0, difficulty=None, representation='simple115v2', rewards='scoring', action_set='default', auto_reset=True, stacked=False, channel_dimensions=(96, 72), raw=False, frames=False, frame_size=(1280, 720), config=None)`

The arguments fall into four groups. The scenario arguments select the matches, the episode
arguments set their number and their seeds, the output arguments set what the batch returns, and the
last two arguments serve rendering and the machine config. The column GRF names the option of GRF
that corresponds to an argument.

| Argument | Type | Default | GRF | Meaning |
| --- | --- | --- | --- | --- |
| `scenario` | `str` | required | `env_name` | A shipped scenario without agent counts, such as `5_vs_5` or `11_vs_11_kaggle`, or the path of a scenario file. `cuju.scenario_names()` lists the shipped ones. Described in [Scenarios](scenarios.md) and [Custom scenarios](custom-scenarios.md). |
| `environment_count` | `int` | `1024` | | The number of environments. |
| `device` | `str` or `torch.device` | `'auto'` | | `auto`, `cpu`, `cuda` or `cuda:N`. Both backends give the same outputs bit for bit, so the device decides only where the batch runs. `auto` takes the current CUDA device where PyTorch sees a GPU and the CUDA library is installed, else the CPU, and logs its choice once. |
| `seed` | `int` | `0` | | From `0` to `2**64 - 1`. Environment `index` plays as GRF after `random.seed(seed * 2**32 + index)`. Described in [Episodes and seeds](episodes-and-seeds.md). |
| `left_agents` | `int` | `1` | `number_of_left_players_agent_controls` | The agents of the left team, which the policy controls. |
| `right_agents` | `int` | `0` | `number_of_right_players_agent_controls` | The agents of the right team. The players that no agent controls play GRF's built-in AI. |
| `difficulty` | `(float or None, float or None)` or `None` | `None` | `left_team_difficulty`, `right_team_difficulty` | The `(left, right)` difficulty of the teams, each from 0 to 1 or `None` for the scenario's. |
| `representation` | `str` | `'simple115v2'` | `representation` | `simple115`, `simple115v2`, `extracted`, `raw`, `pixels` or `pixels_gray`. Described in [Observations](observations.md) and, for the pixels, [Frames and videos](rendering.md#rendering-batches). |
| `rewards` | `str` | `'scoring'` | `rewards` | `scoring` or `scoring,checkpoints`. Described in [Rewards and actions](rewards-and-actions.md). |
| `action_set` | `str` | `'default'` | `action_set` | `default` with 19 actions, `v2` with 20 or `full` with 33. |
| `auto_reset` | `bool` | `True` | | Starts the next episode inside the step that ends one. |
| `stacked` | `bool` | `False` | `stacked` | Stacks the last four observations along the last axis, the oldest first. |
| `channel_dimensions` | `(int, int)` | `(96, 72)` | `channel_dimensions` | The width and height of the SMM, to which `pixels` and `pixels_gray` shrink their frames as well. |
| `raw` | `bool` | `False` | | Also writes the raw observation, for `raw()` and `information['final_raw']`, whatever the representation. |
| `frames` | `bool` | `False` | `render` | Renders every environment as GRF does, for `frame()`. |
| `frame_size` | `(int, int)` or `int` | `(1280, 720)` | `render_resolution_x`, `render_resolution_y` | The `(width, height)` of the frames, the width a multiple of 4, each up to 16,384. A width alone keeps the height at 720, as in GRF. |
| `config` | `dict` or `None` | `None` | | Keys of the table `[run]` of the [machine config](#machine-config) that this batch overrides. |

The agent slots are the left agents first, then the right agents, and each agent sees the pitch from
its own end. A scenario name carries no agent counts, so `5_vs_5_l4_r0` is written as `'5_vs_5'`
with `left_agents=4`, and the constructor refuses a name with a suffix. Every environment of a batch
plays one scenario with one set of agent counts, so the agent axis has the same length everywhere.
The curriculum recipe in [Recipes](recipes.md#curriculum-across-scenarios) shows the way to train
across scenarios, with one batch for each.

### Methods

A program calls `reset` once, then `step` in a loop. The other methods serve a step split in two on
the CPU, the reset of ended environments without the automatic reset, the difficulty of the next
episodes, the state of one environment, and GRF's raw observation and frames.

| Method | Returns | Meaning |
| --- | --- | --- |
| `reset(seeds=None, mask=None, reverse=None)` | observation | Starts the next episode of every environment, or of those where `mask` is true. `seeds` takes one seed per environment, as a list, tuple, array or tensor, or one integer for all, each from `0` to `2**32 - 1` or `-1` to draw. `reverse` takes the team-order flag in the same forms, read at the first reset of an environment. A mask on the device of the batch needs no wait for the device, so a CUDA graph can capture it once every environment has had its first reset. |
| `step(actions)` | `(observation, reward, done, information)` | Runs one step. `actions` is `[E, A]` of integers, or `[E]` for one agent, and a contiguous `int32` tensor on the device of the batch is read in place. Releases the GIL while the library runs. |
| `step_async(actions)`, `step_wait()` | `None`, then as `step` | A step in two halves. On the CPU the step runs in a thread of the batch meanwhile, with a copy of the actions, so learning overlaps simulation. On the GPU `step_async` enqueues the step as `step` does. |
| `reset_done()` | `None` | With `auto_reset=False`, starts the next episode of every environment that ended. |
| `set_difficulty(values, mask=None)` | `None` | Sets the difficulty that each environment plays from its next reset. `values` is a `(left, right)` pair for the selected environments, each from 0 to 1 or `None` for the scenario's, or `[E, 2]` values with NaN for the scenario's. `mask` selects the environments, all by default. |
| `get_state(index, stacked=False)` | `bytes` | The state of one environment in the byte format of GRF. `stacked=True` adds the stack of four of GRF. |
| `set_state(index, state, stacked=False)` | `None` | Sets one environment to a state of cuju or of GRF, rewriting the returned tensors in place. The format is described in [States](states.md). |
| `raw()` | `dict` | GRF's raw observation after the last `reset` or `step`, a dictionary of `[E, A, ...]` tensors. It needs `raw=True` or the representation `raw`. |
| `frame()` | tensor | GRF's frame of every environment after the last `reset` or `step`, `[E, height, width, 3]` `uint8`. It needs `frames=True` or a `pixels` representation. |
| `close()` | `None` | Frees the batch at once. Repeated calls are harmless. |
| `__enter__()`, `__exit__()` | the batch, then `None` | The context manager, which closes the batch at the end of a `with` block, an exception included. |

A batch dropped without `close()` is freed with its last tensor, and `close()` frees it at once.
Explicit closing suits the GPU, where the memory of a batch is large and the release of its last
tensor can be hard to locate. Entering a closed batch raises `ClosedError`. A batch holds device
memory and threads, so it cannot be pickled, and each worker process creates its own.

### Step outputs

`step` returns the observation, the reward, the done flags and a dictionary of information.

| Output | Shape and type |
| --- | --- |
| observation | `[E, A, 115]` `float32` for the vector representations, `[E, A, height, width, 4]` `uint8` for `extracted`, a dictionary of `[E, A, ...]` tensors for `raw`. A stack multiplies the last axis by 4. |
| reward | `[E, A]` `float32` |
| done | `[E]` `bool` |
| `information['final_observation']` | as the observation, meaningful where `done` is true |
| `information['score_reward']` | `[E]` `int32`, the scoring reward of the left team |
| `information['final_raw']` | the raw observation of the step that ended each episode, present when the batch writes the raw observation |

The observation always keeps the agent axis, so a batch of one agent returns `[E, 1, 115]` and its
rewards `[E, 1]`. Code that wants the shapes of a single agent indexes the axis, as in
`observation[:, 0]`. Every returned tensor shares memory with the batch and is overwritten by the
next `reset`, `step` or `set_state`, so a caller clones what it keeps. Actions, masks and seeds in
host memory, page-locked or not, may be changed as soon as `step`, `step_async` or `reset` returns.
Frames come from `frame()` alone, and `information` never holds them.

### Attributes

A batch exposes its configuration and layout as attributes.

| Attribute | Type | Meaning |
| --- | --- | --- |
| `environment_count` | `int` | The number of environments. |
| `agents` | `int` | The size of the agent axis, `left_agents + right_agents`. |
| `left_agents`, `right_agents` | `int` | The agents of each team. |
| `action_count` | `int` | The actions of the action set, 19, 20 or 33. |
| `left_players`, `right_players` | `int` | The players of each team, as `len(observation['left_team'])` of GRF for a left agent. |
| `device` | `torch.device` | The device of the buffers, such as `cuda:0`. |
| `scenario_name` | `str` | The scenario, by its name without folder or suffix. |
| `difficulty` | `[E, 2]` `float32` | The difficulties that play, the scenario's where nothing overrides them. |
| `stacked` | `bool` | Whether the observation stacks the last four. |
| `memory_bytes` | `int` | The bytes of memory that the batch has requested so far, device memory on the GPU and host memory on the CPU. The driver reserves the local memory of the kernels at the first batch of a process on a GPU, about 5.6 KB for each thread that the GPU runs at a time, and this count leaves that reservation out. |
| `single_observation_space`, `single_action_space` | spaces | One environment, as the spaces of GRF, with `None` as the observation space of `raw`. |
| `observation_space`, `action_space` | spaces | The whole batch. |
| `run_config` | `dict` | The resolved machine settings, described under [the machine config](#machine-config). |

The spaces require `gymnasium`, installed with the extra `spaces`, or `gym`.

### Guarantees

`step` and `reset_done` are also the PyTorch operators `cuju::step` and `cuju::reset_done`, so a
function that steps a batch compiles under `torch.compile(fullgraph=True)`, and the steps of a GPU
batch can be captured in CUDA graphs, as [Backends](backends.md#compilation-with-torchcompile)
describes. A GPU batch enqueues its work and returns, so `step` never waits for the device.

A batch whose memory exceeds the free memory of its GPU at the moment of creation issues a
`UserWarning` that names both amounts. Under Windows and WSL2 the warning is the main sign that the
driver placed the batch in shared system memory, as described in [Performance](performance.md).
With `allocator = "torch"`, the memory that the cache of PyTorch holds unused counts as free. A GPU
that runs out of memory at creation raises `DeviceError`, whose message carries the bytes requested
and the bytes free.

### Actions

Actions are integers from 0 to `action_count - 1`, and 0 is the idle action in every action set. An
action indexes the action set, so the range has one meaning on every device and in every mode. The
CPU and a batch with `debug = true` refuse an action outside the range and raise `ArgumentError`
that names the first one, its agent and its environment, before any environment steps. A GPU batch
with `debug` off plays an action outside the range on the device as idle, since a check of every
action would make each step wait for the device. Code that passed `-1` for idle passes 0 instead.
Floating-point actions raise `TypeError`. The functions of `cuju.env` take GRF's indices as GRF
defines them, so GRF code that passes negative indices there plays the same matches as before. The
C API takes indices from `-action_count` to `action_count - 1`.

`UndefinedPathError` reports a match that reached a path that GRF leaves undefined. On the CPU the
library raises it from the step. On the GPU the device traps, which ends the CUDA context of the
process, so the next CUDA call reports the failure and the process restarts. A long run therefore
saves states with `get_state` at intervals, as described in
[Episodes and seeds](episodes-and-seeds.md) and [States](states.md). A GPU batch enqueues its work,
so a failure on the device usually surfaces at a later call that waits. With `debug = true` the
batch waits after every call and raises `DeviceError` at the call that caused the failure, which
locates a fault at the cost of the waits.

The errors of `step` fall into two kinds. A wrong type, a wrong shape, an action out of range in
host memory and a closed batch are errors of the calling code, raised before any environment steps,
so the batch keeps its state and the corrected call succeeds. A step before the first `reset` raises
`LibraryError`.

## Machine config

The **machine config** is a TOML file that holds the settings that change how fast a batch runs and
leave every output as it is. The settings depend on the machine, with its processors, its GPU and
its memory, and not on the experiment, so they live beside the machine and not in the code. The same
script then runs on a laptop and on a server with the thread count and the allocator that suit each,
the call of an experiment lists only what its results depend on, and a measurement of speed on one
machine does not leak into a program that another machine runs.

The file is the path in `CUJU_CONFIG`, else `~/.config/cuju/engine.toml` on Linux and macOS, with
`XDG_CONFIG_HOME` in place of `~/.config` where it is set, and `%APPDATA%\cuju\engine.toml` on
Windows. A missing file leaves every key at its default. The file holds the table `[run]` and
nothing else, and `[run]` holds only the keys below. Any other table, key or kind of value raises
`ConfigurationError` that names the file.

```toml
[run]
threads = 0            # 0 for every processor the process may use
debug = false          # wait for the device after each call and check every action
allocator = "cuda"     # or "torch"
pin_threads = false
numa = false
pin_memory = "auto"    # on when CUDA is present and the batch is on the CPU
render_threads = 0
render_workers = 0
log = false            # one line per batch created
```

| Key | Default | Meaning |
| --- | --- | --- |
| `threads` | `0` | The CPU threads, the caller included. `0` takes one per processor the process may run on, up to its cgroup CPU quota. |
| `debug` | `false` | Waits for the device after every `step`, `step_async`, `reset_done` and `reset` of a GPU batch, so that a failure on the device raises at the call that caused it, and checks every action against the action set. The CPU checks every action anyway. |
| `allocator` | `"cuda"` | The source of GPU buffers, `"cuda"`, or `"torch"` for the caching allocator of PyTorch. |
| `pin_threads` | `false` | Binds each CPU worker to a processor of its own. |
| `numa` | `false` | `true` forms one group of CPU workers per NUMA node, each with its own environments. A string such as `"0,0,1,1"` gives the node of each processor and stands in for what the system reports. |
| `pin_memory` | `"auto"` | Page-locks the output buffers of a CPU batch for asynchronous copies to the GPU. `"auto"` turns it on where PyTorch sees a CUDA GPU. |
| `render_threads` | `0` | The threads that draw each frame of the software renderer. `0` takes the processors left to each worker of a batch, and for `cuju.gfootball` the processors up to four. |
| `render_workers` | `0` | The renderers of a batch that draw at once. `0` takes one per environment up to the processors. |
| `log` | `false` | Logs one line per batch created, through the logger `cuju`. |

[Backends](backends.md) describes the effect of each key.

The argument `config` of the constructor overrides keys of `[run]` for one batch, as in
`cuju.BatchEnvironment('5_vs_5', device='cpu', config={'threads': 4})`, and wins over the file. A
key outside `[run]` raises `ConfigurationError`. The attribute `run_config` of a batch is a plain
dict that maps each key to `{'value': ..., 'source': ...}`, where the source is `'default'`,
`'file <path>'` or `'call'`, so a program can log which settings a batch ran with and where each
came from.

`cuju check` prints the path that it read and every value that differs from the default. A line such
as `machine config: /home/user/.config/cuju/engine.toml, read` is followed by
`differs from the default: threads = 8`. A config that fails to load prints the line with `refused`
and the reason. A `debug` that is on also gets a line of its own, since every call of a GPU batch
then waits for the device.

## Environment variables

The library variables select a rebuilt library without reinstalling the package, and the others
select the config, the media of the software renderer and the hooks of batches. None of them changes
an output, the frames of the software renderer included.

| Variable | Meaning |
| --- | --- |
| `CUJU_CONFIG` | the path of the machine config, in place of the default location |
| `CUJU_BATCH_HOOKS` | The names of batch creation hooks, separated by commas. Each name is an entry point in the group `cuju.batch_hooks` for a function that receives the new batch. The logger `cuju` warns once per process for a missing hook and warns for the affected batch if a hook fails. The batch continues after either warning. |
| `CUJU_GRF_MEDIA` | a folder of GRF's media for the software renderer, in place of the installed cuju-grf-media |
| `CUJU_CUDA_LIBRARY` | the path of the CUDA library to load |
| `CUJU_CPU_LIBRARY` | the path of the CPU library to load |
| `CUJU_RENDER_LIBRARY` | the path of the software renderer's library to load |

The bundles that the package builds from scenario files are cached in the user cache folder,
`$XDG_CACHE_HOME/cuju/bundles` or `~/.cache/cuju/bundles` on Linux, `~/Library/Caches/cuju/bundles`
on macOS and `%LOCALAPPDATA%\cuju\bundles` on Windows.

## Namespace

`import cuju` shows exactly these names.

| Name | Meaning |
| --- | --- |
| `BatchEnvironment` | the batch, described above |
| `scenario_names()` | the names of the shipped scenarios, such as `5_vs_5`, which a batch takes as `scenario` |
| `Error` | the base of every exception that cuju raises |
| `errors` | the module that holds every class of [Errors](#errors) |
| `engine_status` | the check of the installation that the family uses |
| `__version__` | the version of the package |

The core of the package depends on PyTorch alone, and `import cuju` loads PyTorch when the name
`BatchEnvironment` is first used. Each submodule is imported by name, as in `import cuju.dumps`, and imports
the packages it depends on at that point.

### Public submodules

| Module | Names |
| --- | --- |
| `cuju.scenarios` | `build_bundle`, `describe`, `load_scenario`, `cached_bundle`, `cache_directory`, `ScenarioError` and the builder of GRF, `Builder`, `ScenarioConfig`, `PlayerRole` and `Team`. Described in [Custom scenarios](custom-scenarios.md). |
| `cuju.dumps` | `DumpWriter(batch, logdir, environments=None, write_full_episode_dumps=True, write_goal_dumps=False, dump_frequency=1)` with `record(actions)`, `close()` and `paths`, which writes the matches of a `BatchEnvironment` as GRF dumps, and `read_dump(path)`. Described in [Migrating from GRF](migrating-from-grf.md#dumps-from-a-batch). |
| `cuju.env` | `create_environment(env_name, ..., *, device='cpu', threads=1, api='gym')` with the arguments and defaults of GRF, which returns one environment with the gym 0.21 interface of GRF, or the Gymnasium interface with `api='gymnasium'`, and `register_environments()` for the `GFootball-*-v0` ids. It needs Gymnasium, from the extra `spaces`, or GRF's gym. Described in [Migrating from GRF](migrating-from-grf.md). |
| `cuju.gfootball` | GRF's Python package `gfootball`, every module ported with its import lines changed, so that GRF code runs with its imports switched, internals included. Described in [Migrating from GRF](migrating-from-grf.md). |
| `cuju.gfootball_engine` | GRF's engine package `gfootball_engine` over cuju's engine, with the conversions and errors of GRF's compiled module, `SignatureError` and `TrackerError` among them. Described in [Differences from GRF](differences-from-grf.md#grfs-python-package). |

`cuju.gfootball` needs `absl-py` and `six`, which cuju installs, and the installed gym wherever GRF
runs on it, else a copy of gym 0.21's core that it carries. `cuju.gfootball_engine` and
`cuju.scenarios` run without PyTorch. Importing `cuju` registers the `GFootball-*-v0` ids of
`cuju.env` with Gymnasium as soon as Gymnasium is imported, and `import cuju.gfootball` registers
GRF's ids with the gym it uses, as importing `gfootball` registers them in GRF.

### Private names

The names that begin with an underscore, such as `cuju._batch.Batch`, and a few modules without one
belong to the family and are free to change between releases. `cuju._batch.Batch` is the batch with
every option of the engine, which `BatchEnvironment` maps onto, and `cuju check`, `cuju.env`, the
replays and the packages of the family use it directly. The step observers, the roles of the
players, `cuju.kaggle`, `cuju.raw`, `cuju.state`, `cuju.players`, `cuju.wrappers`, the action logs,
the digests and the replays are private as well. Programs outside the family use the public names
above.

## Errors

Every class lives in `cuju.errors`, is reachable as `cuju.Error` for the base, and derives from
`cuju.Error` and from a built-in type, so that `except ValueError` and similar clauses also catch
it. The double ancestry allows handlers at three levels of precision. `except cuju.Error` catches
every exception that cuju raises and no other, as suits a training script that logs and restarts. A
specific class, such as `StateError` around `set_state`, catches one failure. Code written against
the built-in types, such as code written for GRF, keeps catching the type it catches.

| Class | Also a | Raised for |
| --- | --- | --- |
| `Error` | `Exception` | the base of all |
| `ConfigurationError` | `ValueError` | an option refused at creation, including more agents than a team has players and a machine config that fails to load. `scenarios.ScenarioError` subclasses it |
| `ChoiceError` | `ConfigurationError`, `KeyError` | an unknown name for a choice, such as a `representation`, an `action_set` or a `device` |
| `ArgumentError` | `ValueError`, `IndexError` | a bad argument of a call, such as actions of the wrong shape, an action out of range, an environment index outside the batch, or `reset(seeds=, reverse=, mask=)` of the wrong length or range |
| `StateError` | `ValueError`, `RuntimeError` | a state that fails to load into its environment |
| `ClosedError` | `RuntimeError` | any call on a closed batch |
| `DeviceError` | `RuntimeError` | a GPU that PyTorch does not see, a CPU without AVX2, BMI2 or FMA, a CUDA failure, or out of memory, with the bytes requested and free |
| `UndefinedPathError` | `RuntimeError` | an environment that reached a path that GRF leaves undefined |
| `LibraryError` | `RuntimeError` | `step` before `reset`, `raw()` or `frame()` on a batch without that output, and every other failure the library reports, each with a message that names the cause |
| `MissingFileError` | `FileNotFoundError` | a missing scenario file, bundle or library, or missing GRF media for the software renderer |

Floating-point actions, seeds, masks and flags raise the built-in `TypeError`.

## Command line

The engine installs `cuju`, which also runs as `python -m cuju`. Bare `cuju` and `cuju help` show
the engine commands followed by the installed family commands.

| Command | Meaning |
| --- | --- |
| `cuju check [--device D] [--long] [--case NAME] [--list] [--json]` | The self-check of the package data and of the matches that each backend plays, with the path of the machine config and its non-default values. |
| `cuju replay FILE [--device D] [--json]` | Plays a GRF dump again without rendering and compares each step with the recorded step, running no code from the dump. `D` is `cpu` by default. |
| `cuju dump FILE [-o OUT] [--json]` | Writes a dump as text, without running its code. The text goes to standard output without `-o`. |
| `cuju help [NAME ...]` | The commands of the family, or the help of one, so that `cuju help arena team add` shows `cuju arena team add --help` when the arena is installed. |
| `cuju completion SHELL` | The completion script for `bash`, `zsh`, `fish` or `powershell`. |
| `cuju -V` | One line per distribution, the engine first, then the distributions that provide commands, then the umbrella distribution `cuju` when it is installed. |

Use `cuju check --help` or `cuju help check` for the options and examples of one command. A
misspelled command exits with status 2 and can suggest a similar installed or known family command.
If the arena or renderer is missing, asking for it exits with status 3 and shows its installation
command.

`cuju check` verifies the package data and compares matches with the shipped digests. The default
checks all available backends, `--device cpu` or `--device cuda` selects one, and `--list` shows the
available cases before `--case NAME` selects one. Listing reads the case descriptions without
loading PyTorch or starting a backend. In text, each row starts with the case name and uses two
spaces between columns. With `--json`, standard output is one object whose first key is `schema`,
with the value `engine.check/1` for the check or `engine.check.list/1` for the case list. Diagnostic
and progress lines go to standard error, and `--quiet` hides them while keeping errors visible. If
no backend can run, the check exits with status 3 unless package data already failed the check.

`cuju replay FILE` reads dumps that GRF and cuju wrote. With `--json` it prints `engine.replay/1`,
an object with `steps`, the number of equal steps, `ended`, whether the replay ended the episode,
and `difference`, the first difference as text or `null`. A difference keeps exit status 1. `cuju
dump FILE` prints the text of the dump, or with `-o OUT` writes it to `OUT` and prints that path.
With `--json` it prints `engine.dump/1`, an object with `files`, the list of the written paths, for
`-o`, and with `text`, the text of the dump, otherwise. A missing or unreadable dump gives an error
and a hint, with exit status 1. Input, usage and environment errors of every command use `error/1`
as their schema under `--json`.

```sh
cuju check --device cpu
cuju replay episode_done.dump --json
cuju dump episode_done.dump -o episode_done.txt
```

The commands use exit status 0 for success, 1 for a negative result, 2 for a usage error and 3 when
the environment cannot run the command. An unexpected error exits with status 70. Add `--verbose`
before or after the command to see its traceback. Ctrl+C exits with status 130, and a closed output
pipe exits quietly. Subcommand diagnostics use the name `cuju` followed by the command, including
when the launcher is `cuju.exe`, and paths in errors appear in single quotes with their original
backslashes.

Use `cuju completion bash`, `cuju completion zsh`, `cuju completion fish` or
`cuju completion powershell` to generate a completion script. Save and source the script in Bash,
Fish or PowerShell. In Zsh, run `autoload -Uz compinit` and `compinit` before sourcing the script,
or save it as `_cuju` in a directory on `fpath` before running `compinit`. Press Tab to complete
commands, options and choices from the packages installed when the script was generated, the names
of the shipped cases after `cuju check --case` and the devices after `cuju replay --device`. The
script is generated again after a package is installed or removed.

A package adds commands by declaring entry points in the group `cuju.commands`, with one
`main(arguments)` function per command. The module of the entry point may expose a `parser()`
function that builds its argparse tree without side effects or heavy imports, which family help and
completion read. Such packages install as modules `cuju.<name>`, and cuju imports one only when a
command or a script asks for it, so `import cuju` and the steps of a batch leave them unloaded.
Where `cuju.<name>` is missing, `import cuju.<name>` and `python -m cuju.<name>` end with one
message that suggests `pip install cuju-<name>`. When importing an optional module in a `try`
block, a program compares the `name` of the exception with the module name to tell a missing module
from a missing dependency of that module.

## C API

Both libraries export the same C functions, declared in `src/environment_api.h` and the other
`src/*_api.h` headers, which define their exact signatures. The Python package is a thin layer over
them. The C API serves programs that run outside Python, such as an engine in another language or a
training system with its own runtime. It covers the engine alone, and the additions of the Python
package, such as the tensors, the spaces and the interfaces of GRF, are then built by the caller. A
call sequence proceeds in four stages.

1. `cuju_environment_create(&configuration)` with a `cuju_configuration` whose `version` is
    `CUJU_CONFIGURATION_VERSION`. It returns a handle, or null on failure.
2. `cuju_environment_reset`, then `cuju_environment_step` in a loop, with
    `cuju_environment_reset_done`, `cuju_environment_get_state` and `cuju_environment_set_state` as
    needed.
3. `cuju_environment_buffer(handle, name)` returns the address of a named output buffer, which stays
    valid and in place for the life of the handle.
4. `cuju_environment_destroy(handle)`.

A function that fails returns a non-zero status or a null pointer, and `cuju_last_error()` returns
the reason. On the CUDA library, the calls that take a stream run on that stream. `step` returns
without waiting unless asked to synchronize, `reset_done` never waits, and `reset`, `get_state` and
`set_state` wait for the device. The `version` field of `cuju_configuration` names the layout of the
structure, and `cuju_environment_create` refuses a configuration whose version differs from the
`CUJU_CONFIGURATION_VERSION` of the library, which is 3 in cuju 0.4.0. The actions of the C API run
from `-action_count` to `action_count - 1`.

Further calls read and write what the batch holds. `cuju_environment_roles` writes the role of each
player of one environment. `cuju_state_layout`, `cuju_state_gather` and `cuju_state_padding` read
the state format without a batch. `cuju_memory_estimate(&configuration, &bytes)` writes the bytes
that `cuju_environment_create` would request without allocating them, `cuju_environment_memory_bytes`
returns the bytes that a batch has requested so far, `cuju_local_memory_bytes` writes the local
memory that a creation on the current CUDA device adds for the kernels, and
`cuju_environment_synchronize(handle, stream)` waits for the work of the batch on `stream`. A library
built with the CMake option `CUJU_NVTX` wraps the creation, the resets, the steps and the state calls
in NVTX ranges, as described in [Backends](backends.md). The renderer has calls of its own, in
`src/render/render_api.h` and in the render calls of `src/environment_api.h`, which
[Frames and videos](rendering.md#the-renderstate) describes.

## Migration from 0.3

Release 0.4.0 breaks the names of 0.3.0 cleanly, and no old name stays as an alias. The outputs of
the matches stay as they were, bit for bit, for the same scenario, agents, seed and actions. The
tables list every change, the old name on the left and the place of the same thing now on the right.

### Constructor

| 0.3 | 0.4 |
| --- | --- |
| `BatchEnvironment('5_vs_5_l4_r0', ...)`, a name with agent counts | `BatchEnvironment('5_vs_5', left_agents=4, right_agents=0, ...)`. The suffixed name raises `ConfigurationError`. |
| `assets` as the first argument | `scenario` |
| `assets` as a list of scenarios, `assignment` | One batch for each scenario, as in [Recipes](recipes.md#curriculum-across-scenarios). |
| `agent_counts` | `left_agents` and `right_agents` |
| `difficulty` as a list of pairs for the bundles | `difficulty` as one `(left, right)` pair, and `set_difficulty` for each environment. |
| `device='cuda'` as the default | `device='auto'` as the default |
| `with_raw` | `raw` |
| `render_frames` | `frames` |
| `render_resolution_x`, `render_resolution_y` | `frame_size=(width, height)` |
| `squeeze` | The agent axis stays in every output, and `observation[:, 0]` drops it. |
| `first_index` | Machines take their own `seed`, as in [Recipes](recipes.md#one-process-per-gpu). |
| `episode_number`, `physics_steps_per_frame`, `game_engine_random_seed`, `numpy_legacy`, `animations`, `library` | Kept on the private `Batch` of the family, and absent from the public class. `reset(seeds=...)` plays a chosen seed. |
| `threads`, `allocator`, `pin_threads`, `numa`, `pin_memory` | Keys of `[run]` in the [machine config](#machine-config), or `config=` for one batch. |
| `debug`, `check_actions` | The key `debug` of `[run]`, which also checks actions. |

### Methods and attributes

| 0.3 | 0.4 |
| --- | --- |
| `done` as `uint8` | `done` as `bool`, so `done.bool()` becomes `done`. |
| `information['frame']` | `batch.frame()` |
| `final_raw()` | `information['final_raw']` for a batch with `raw=True` or the representation `raw` |
| `batch.torch_device` | `batch.device`, a `torch.device`. The old string `batch.device` is gone. |
| `batch.threads`, `batch.players`, `batch.agent_mask` | The agent counts are integers, so every environment has every agent and no mask is needed. `batch.run_config['threads']` holds the setting. |
| `left_agents`, `right_agents` and `left_players`, `right_players` as `[E]` tensors | Integers, the same in every environment. |
| `squeeze`, `channel_dimensions`, `auto_reset`, `check_actions`, `debug` attributes | The arguments that the caller passed, and `run_config['debug']`. |
| `roles(index)`, `observation()` | Private to the family. |
| `add_observer`, `remove_observer`, `observers` | Private to the family. A wrapper around `step` serves a program that counts goals, as in [Recipes](recipes.md#episode-return-and-length). |
| `reset(seeds=None, reverse=None, mask=None)` | `reset(seeds=None, mask=None, reverse=None)`. Calls with keywords are unchanged. |
| `memory_bytes` | Unchanged, and it leaves out the local memory of the kernels. |

### Names of the package

| 0.3 | 0.4 |
| --- | --- |
| `cuju.bundled_scenarios()` | `cuju.scenario_names()` |
| `cuju.bundled_assets()` | Removed. |
| `cuju.library_path()` | Removed. `CUJU_CUDA_LIBRARY` and `CUJU_CPU_LIBRARY` select a library. |
| `cuju.memory_estimate()` | Removed. A small batch gives `memory_bytes`, as in [Recipes](recipes.md#sizing-a-batch). |
| `cuju.MultiDeviceBatch` | One process per GPU, as in [Recipes](recipes.md#one-process-per-gpu). |
| `cuju.ActionLog`, `cuju.Digest`, `cuju.digest`, `cuju.replay` | Private, in `cuju.action_log`. `cuju replay FILE` replays a dump. |
| `cuju.ActionRangeError`, `cuju.EnvironmentIndexError` | `ArgumentError` |
| `cuju.SignatureError`, `cuju.TrackerError` | `cuju.gfootball_engine.SignatureError` and `cuju.gfootball_engine.TrackerError` |
| every class of `cuju.errors` on the top level | `cuju.Error` and the module `cuju.errors` |
| `cuju.kaggle`, `cuju.raw`, `cuju.state`, `cuju.players`, `cuju.wrappers` | Private to the family. |
| `cuju.dumps.DumpWriter(batch, ...)` | Unchanged, and it takes a `BatchEnvironment`. |

### Behavior

| 0.3 | 0.4 |
| --- | --- |
| Actions from `-action_count` to `action_count - 1` | Actions from 0 to `action_count - 1`. The CPU and a `debug` batch raise `ArgumentError` for others. A GPU batch with `debug` off plays an action outside the range as idle, so code that passed `-1` for idle passes 0. `cuju.env` and `cuju.kaggle` still take GRF's negative indices and turn them into the same positive action. |
| A GPU batch that runs out of memory raises `LibraryError` | `DeviceError`, with the bytes requested and free in the message |
| `ActionRangeError` and `EnvironmentIndexError` | `ArgumentError`, which is a `ValueError` and an `IndexError` |
| `device='cuda'` without a GPU raises `DeviceError` | `device='auto'` runs on the CPU there, and `device='cuda'` still raises `DeviceError` |

### Environment variables and commands

| 0.3 | 0.4 |
| --- | --- |
| `CUJU_NUMA_NODES` | The key `numa` of `[run]`, as a string such as `"0,0,1,1"`. |
| `CUJU_RENDER_THREADS`, `CUJU_RENDER_WORKERS` | The keys `render_threads` and `render_workers` of `[run]`. |
| `CUJU_LOG` | The key `log` of `[run]`. |
| `CUJU_BUNDLE_CACHE` | Removed, since the bundle cache is always on, in the user cache folder. |
| `cuju replay --trace_file FILE`, `--fps` | `cuju replay FILE` |
| `cuju dump-to-txt --trace_file FILE --output OUT`, `--include_debug`, `--noinclude_debug` | `cuju dump FILE -o OUT` |

## See also

- [Concepts](concepts.md), for batches, bundles and environments
- [Recipes](recipes.md), for complete programs built on the package
- [Common topics](faq.md), for the behaviors that most often prompt questions
- [Migrating from GRF](migrating-from-grf.md), for the interface of GRF's own Python package
