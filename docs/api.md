# API reference

The package `cuju` has one public class, `cuju.BatchEnvironment`, a machine config for settings that change only speed, the command `cuju`, and a C API. One rule places every option. What changes results is an argument of the constructor, and what changes only speed is a key of the machine config. In shapes, `E` is `environment_count` and `A` is `batch.agents`.

## BatchEnvironment

```python
batch = cuju.BatchEnvironment(
    scenario,
    environment_count=1024,
    device='auto',
    seed=0,
    left_agents=1,
    right_agents=0,
    difficulty=None,
    representation='simple115v2',
    rewards='scoring',
    action_set='default',
    auto_reset=True,
    stacked=False,
    channel_dimensions=(96, 72),
    raw=False,
    frames=False,
    frame_size=(1280, 720),
    config=None,
)
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `scenario` | required | A shipped scenario without agent counts, such as `5_vs_5`, or the path of a scenario file. See [Scenarios](scenarios.md). |
| `environment_count` | `1024` | The number of environments. |
| `device` | `'auto'` | `auto`, `cpu`, `cuda`, `cuda:N` or a `torch.device`. Both backends give the same bits. `auto` takes the current CUDA device when PyTorch sees a GPU and the CUDA library is installed, else the CPU, and logs its choice once. |
| `seed` | `0` | From 0 to 2**64 - 1. Environment `index` plays as GRF after `random.seed(seed * 2**32 + index)`. |
| `left_agents`, `right_agents` | `1`, `0` | The agents of each team. Players without an agent play GRF's built-in AI. |
| `difficulty` | `None` | A `(left, right)` pair, each from 0 to 1 or `None` for the scenario's. |
| `representation` | `'simple115v2'` | `simple115`, `simple115v2`, `extracted`, `raw`, `pixels` or `pixels_gray`. See [Observations, rewards and actions](observations-rewards-actions.md). |
| `rewards` | `'scoring'` | `scoring` or `scoring,checkpoints`. |
| `action_set` | `'default'` | `default` with 19 actions, `v2` with 20 or `full` with 33. |
| `auto_reset` | `True` | Starts the next episode inside the step that ends one. |
| `stacked` | `False` | Stacks the last four observations on the last axis, the oldest first. |
| `channel_dimensions` | `(96, 72)` | The width and height of the SMM, and of `pixels` and `pixels_gray`. |
| `raw` | `False` | Also writes the raw observation, for `raw()` and `information['final_raw']`. |
| `frames` | `False` | Renders GRF's frame of every environment, for `frame()`. |
| `frame_size` | `(1280, 720)` | `(width, height)`, the width a multiple of 4. A width alone keeps the height at 720, as in GRF. |
| `config` | `None` | Keys of the `[run]` table of the machine config that this batch overrides. |

The agent slots are the left agents first, then the right agents, and each agent sees the pitch from its own end. A scenario name carries no agent counts, so `5_vs_5_l4_r0` is written as `'5_vs_5'` with `left_agents=4`, and a suffixed name raises `ConfigurationError`. One batch plays one scenario, so training across scenarios uses one batch for each.

### Methods

| Method | Returns | Meaning |
| --- | --- | --- |
| `reset(seeds=None, mask=None, reverse=None)` | observation | Starts the next episode of every environment, or of those where `mask` is true. `seeds` is one seed per environment or one for all, each from 0 to 2**32 - 1, or -1 to draw. `reverse` is the team-order flag, read at the first reset of an environment. A mask on the batch's device needs no wait, so a CUDA graph can capture it. |
| `step(actions)` | `(observation, reward, done, information)` | Runs one step. `actions` is `[E, A]` integers, or `[E]` for one agent. A contiguous `int32` tensor on the batch's device is read in place. |
| `step_async(actions)`, `step_wait()` | `None`, then as `step` | A step in two halves. On the CPU the step runs in a batch thread meanwhile, so learning overlaps simulation. |
| `reset_done()` | `None` | With `auto_reset=False`, starts the next episode of every environment that ended. |
| `set_difficulty(values, mask=None)` | `None` | Sets the difficulty from the next reset. `values` is a `(left, right)` pair or `[E, 2]` with NaN for the scenario's. |
| `get_state(index, stacked=False)` | `bytes` | The state of one environment in GRF's byte format. |
| `set_state(index, state, stacked=False)` | `None` | Loads a state of cuju or of GRF into one environment. The returned tensors are rewritten in place. |
| `get_states(stacked=False)` | `list[bytes]` | The state of every environment, byte for byte what `get_state` gives each, with one copy from the device. |
| `set_states(states, stacked=False)` | `None` | Loads one state into every environment, as a loop of `set_state` does, with one copy to the device. |
| `raw()` | `dict` | GRF's raw observation after the last call, `[E, A, ...]` tensors. Needs `raw=True` or the representation `raw`. |
| `frame()` | tensor | GRF's frames, `[E, height, width, 3]` `uint8`. Needs `frames=True` or a pixel representation. |
| `close()` | `None` | Frees the batch at once. A `with` block closes it too. |

A call on a closed batch raises `ClosedError`. A batch holds device memory and threads, so each worker process creates its own, and pickling one raises `TypeError`.

### Step outputs

| Output | Shape and type |
| --- | --- |
| observation | `[E, A, 115]` `float32` for the vector representations, `[E, A, height, width, 4]` `uint8` for `extracted`, a dictionary of `[E, A, ...]` tensors for `raw`. A stack multiplies the last axis by 4. |
| reward | `[E, A]` `float32` |
| done | `[E]` `bool` |
| `information['final_observation']` | as the observation, meaningful where `done` is true |
| `information['score_reward']` | `[E]` `int32`, the left team's scoring reward |
| `information['final_raw']` | the raw observation of the step that ended each episode, with the raw output on |

A batch of one agent returns `[E, 1, 115]`, so `observation[:, 0]` drops the agent axis. Every returned tensor shares memory with the batch and the next `reset`, `step` or `set_state` overwrites it, so clone what you keep. Host memory passed in may be reused as soon as the call returns.

### Attributes

| Attribute | Meaning |
| --- | --- |
| `environment_count`, `agents`, `left_agents`, `right_agents` | The sizes. `agents` is `left_agents + right_agents`. |
| `action_count` | The actions of the set, 19, 20 or 33. |
| `left_players`, `right_players` | The players of each team. |
| `device` | A `torch.device`. |
| `scenario_name`, `stacked` | The scenario name and the stack flag. |
| `difficulty` | `[E, 2]` `float32`, the difficulties that play. |
| `memory_bytes` | The bytes requested so far, on the device for a GPU batch and on the host for a CPU batch. The driver's local memory for kernels is not counted. |
| `single_observation_space`, `single_action_space`, `observation_space`, `action_space` | Gymnasium or gym spaces, installed with the extra `spaces`. The observation space of `raw` is `None`. |
| `run_config` | The resolved machine settings, each as `{'value': ..., 'source': ...}` with the source `'default'`, `'file <path>'` or `'call'`. |

### Compilation and memory

`step` and `reset_done` are the PyTorch operators `cuju::step` and `cuju::reset_done`, so they compile under `torch.compile(fullgraph=True)` and the steps of a GPU batch can be captured in CUDA graphs. A GPU step enqueues its work and never waits for the device. A batch larger than the free memory of its GPU warns with both amounts. On Windows and WSL2 that warning shows that the driver moved the batch to shared memory, which [Performance](performance.md) covers. A GPU that runs out of memory at creation raises `DeviceError` with the bytes requested and free.

### Actions

Actions are integers from 0 to `action_count - 1`, and 0 is idle in every set. The CPU and a batch with `debug = true` raise `ArgumentError` for an action outside the range, naming the first one, before any environment steps. A GPU batch with `debug` off plays an out-of-range action as idle. Floating-point actions raise `TypeError`. `cuju.env` takes GRF's negative indices and turns them into the same positive action.

A match that reaches a path GRF leaves undefined raises `UndefinedPathError` on the CPU. On the GPU the device traps, which ends the process's CUDA context, so a long run saves states at intervals with `get_state`. With `debug = true` a device failure raises `DeviceError` at the call that caused it.

## Machine config

The machine config is a TOML file for the settings that change speed and leave every output as it is. It lives beside the machine, so the same script runs on a laptop and a server with the settings that suit each.

The file is `CUJU_CONFIG`, else `~/.config/cuju/engine.toml` on Linux and macOS, with `XDG_CONFIG_HOME` honored, and `%APPDATA%\cuju\engine.toml` on Windows. A missing file leaves every key at its default. The file holds the table `[run]` and its keys only, and anything else raises `ConfigurationError` naming the file.

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

| Key | Meaning |
| --- | --- |
| `threads` | The CPU threads, the caller included. `0` takes one per processor the process may run on, up to its cgroup quota. |
| `debug` | Waits for the device after every `step`, `step_async`, `reset_done` and `reset` of a GPU batch, and checks every action. |
| `allocator` | The source of GPU buffers, `"cuda"` or `"torch"` for PyTorch's caching allocator. |
| `pin_threads` | Binds each CPU worker to a processor of its own. |
| `numa` | `true` forms one worker group per NUMA node. A string such as `"0,0,1,1"` gives the node of each processor. |
| `pin_memory` | Page-locks the output buffers of a CPU batch for asynchronous copies to the GPU. `"auto"` turns it on where PyTorch sees a CUDA GPU. |
| `render_threads` | The threads that draw each frame of the software renderer. `0` takes the processors left to each worker. |
| `render_workers` | The renderers of a batch that draw at once. `0` takes one per environment, up to the processors. |
| `log` | Logs one line per batch created, through the logger `cuju`. |

`config={'threads': 4}` in the constructor overrides keys for one batch and wins over the file. `cuju check` prints the path it read and every value that differs from the default.

## Environment variables

| Variable | Meaning |
| --- | --- |
| `CUJU_CONFIG` | The path of the machine config. |
| `CUJU_BATCH_HOOKS` | Comma-separated entry points of the group `cuju.batch_hooks`, each a function that receives every new batch. |
| `CUJU_GRF_MEDIA` | A folder of GRF's media for the software renderer, in place of `cuju-grf-media`. |
| `CUJU_CUDA_LIBRARY`, `CUJU_CPU_LIBRARY`, `CUJU_RENDER_LIBRARY` | The path of a rebuilt library to load. |

Bundles built from scenario files are cached in the user cache folder, `$XDG_CACHE_HOME/cuju/bundles` or `~/.cache/cuju/bundles` on Linux, `~/Library/Caches/cuju/bundles` on macOS and `%LOCALAPPDATA%\cuju\bundles` on Windows.

## Namespace

`import cuju` shows `BatchEnvironment`, `scenario_names()`, `Error`, `errors`, `types`, `engine_status` and `__version__`. Submodules are imported by name.

| Module | Holds |
| --- | --- |
| `cuju.scenarios` | `build_bundle`, `describe`, `load_scenario`, `cached_bundle`, `cache_directory`, `ScenarioError` and GRF's builder classes. See [Scenarios](scenarios.md). |
| `cuju.dumps` | `DumpWriter(batch, logdir, environments=None, write_full_episode_dumps=True, write_goal_dumps=False, dump_frequency=1)` with `record(actions)`, `close()` and `paths`, `read_dump(path)` and `iterate_dump(path)`, which read dumps under limits on depth and size. See [Migrating from GRF](migrating-from-grf.md). |
| `cuju.env` | `create_environment` with GRF's arguments plus `device`, `threads` and `api`, and `register_environments()`. See [Migrating from GRF](migrating-from-grf.md). |
| `cuju.agents` | `Controller`, `Spec`, `Observation`, `Actions`, `ControllerError`, `load(spec, base=None)`, `Match(batch, left=None, right=None, control=None, seed=0, check=True)` with `reset`, `step`, `step_async`, `step_wait`, `run(episodes=1, record=False)`, `set`, `episodes` and `close`, `Result`, `play(left, right, scenario, ...)`, `play_batch(left, right, scenario, environments=64, ...)`, `check(spec, scenario, steps=10, twice=False, process=False, command=None)`, `parts()`, `to_gymnasium(match)` and `to_pettingzoo(match)`. The standard controllers are in `cuju.agents.standard`, the Kaggle helpers `players_raw`, `observation`, `action_index`, `action_indices` and `accepts_configuration` in `cuju.agents.kaggle`, and the frames and messages of the protocol in `cuju.agents.protocol`. See [Controllers](agents.md) and [the protocol](protocol.md). |
| `cuju.types` | The `Literal` aliases `Representation`, `Rewards`, `ActionSet`, `Device`, `Side`, `Sides`, `Assignment`, `Goalkeeper` and the others that the signatures of `BatchEnvironment`, `cuju.agents` and `cuju.recording` use, so a type checker knows the strings that each option takes. |
| `cuju.recording` | `Recording`, `read(source)`, `load(paths, fields=None)`, `record(batch, environments, on_episode, steps=True, provenance=None, teams=None)` and its `Recorder`, `from_grf_dump(source, device='cpu')`, `from_kaggle(source)` and `RecordingError`. See [Recording](recording.md). |
| `cuju.gfootball` | GRF's package `gfootball`, so GRF code runs with its imports switched. |
| `cuju.gfootball_engine` | GRF's engine package over cuju's engine, with `SignatureError` and `TrackerError`. |

Names that start with an underscore, and the modules not listed here, belong to the family of packages and change between releases.

## Errors

Every class lives in `cuju.errors`, derives from `cuju.Error` and from a built-in type, so `except cuju.Error` catches everything cuju raises and `except ValueError` still works.

| Class | Also a | Raised for |
| --- | --- | --- |
| `Error` | `Exception` | the base |
| `ConfigurationError` | `ValueError` | an option refused at creation, such as more agents than players, or a machine config that fails to load |
| `ChoiceError` | `ConfigurationError`, `KeyError` | an unknown name for a representation, action set or device |
| `ArgumentError` | `ValueError`, `IndexError` | a bad argument, an action out of range, or an index outside the batch |
| `StateError` | `ValueError`, `RuntimeError` | a state that fails to load |
| `ClosedError` | `RuntimeError` | a call on a closed batch |
| `DeviceError` | `RuntimeError` | a missing GPU, a CPU without AVX2, BMI2 or FMA, a CUDA failure, or out of memory |
| `UndefinedPathError` | `RuntimeError` | a match on a path GRF leaves undefined |
| `LibraryError` | `RuntimeError` | `step` before `reset`, `raw()` or `frame()` without that output, any other library failure |
| `MissingFileError` | `FileNotFoundError` | a missing scenario, library or media file |

## Command line

`cuju` also runs as `python -m cuju`. Bare `cuju` and `cuju help` list the engine commands and those of installed family packages.

| Command | Meaning |
| --- | --- |
| `cuju check [--device D] [--long] [--case NAME] [--list] [--json]` | The self-check of the package data and of the matches each backend plays, with the config path and its non-default values. |
| `cuju replay FILE [--device D] [--json]` | Replays a GRF dump without rendering, compares each step with the recorded one, and runs no code from the dump. |
| `cuju dump FILE [-o OUT] [--json]` | Writes a dump as text to standard output, or to `OUT`. |
| `cuju recording show FILE`, `check FILE [--replay]`, `convert PATH... -o DIR` | Shows what a recording holds, validates it and replays it, and converts GRF dumps and Kaggle episodes. See [Recording](recording.md). |
| `cuju agents check PATH --scenario S`, `play A B --scenario S`, `parts`, `client` | Plays short matches with a controller and reports what failed and how long each call took, plays two controllers against each other, lists the parts of a tree and prints the file that runs a child. See [Controllers](agents.md). |
| `cuju help [NAME ...]`, `cuju completion SHELL`, `cuju -V` | Family help, a completion script for `bash`, `zsh`, `fish` or `powershell`, and the installed versions. |

```sh
cuju check --device cpu
cuju replay episode_done.dump --json
cuju dump episode_done.dump -o episode_done.txt
```

With `--json`, output is one object whose first key is `schema`, such as `engine.check/1`, `engine.check.list/1`, `engine.replay/1`, `engine.dump/1`, `recording.show/1`, `agents.check/1`, `agents.play/1` and `agents.parts/1`, and errors use `error/1`. Progress goes to standard error, and `--quiet` hides it. Exit status 0 is success, 1 a negative result such as a difference in a replay, 2 a usage error, 3 an environment that cannot run the command, 70 an unexpected error and 130 Ctrl+C. Add `--verbose` for a traceback.

Packages add commands through entry points in the group `cuju.commands`, one `main(arguments)` function each, and install as modules `cuju.<name>`.

## C API

Both libraries export the same C functions, declared in `src/environment_api.h` and the other `src/*_api.h` headers. A program outside Python creates a handle with `cuju_environment_create(&configuration)`, with `version` set to `CUJU_CONFIGURATION_VERSION`, which is 3. It calls `cuju_environment_reset`, then `cuju_environment_step` in a loop, with `reset_done`, `get_state` and `set_state` as needed, reads named outputs with `cuju_environment_buffer(handle, name)`, and ends with `cuju_environment_destroy`. A failing function returns a non-zero status or null, and `cuju_last_error()` gives the reason. On the CUDA library `step` returns without waiting unless asked to synchronize. The C API takes actions from `-action_count` to `action_count - 1`.

## Migrating from 0.3

0.4.0 breaks the names of 0.3.0 and keeps no aliases. Outputs stay bit for bit the same for the same scenario, agents, seed and actions.

| 0.3 | 0.4 |
| --- | --- |
| `BatchEnvironment('5_vs_5_l4_r0', ...)` | `BatchEnvironment('5_vs_5', left_agents=4, right_agents=0, ...)` |
| `assets`, `agent_counts` | `scenario`, `left_agents` and `right_agents` |
| `assets` as a list, `assignment` | One batch for each scenario. |
| `difficulty` as a list of pairs | One `(left, right)` pair, and `set_difficulty` per environment. |
| `device='cuda'` as default | `device='auto'` |
| `with_raw`, `render_frames` | `raw`, `frames` |
| `render_resolution_x`, `render_resolution_y` | `frame_size=(width, height)` |
| `squeeze` | The agent axis stays, and `observation[:, 0]` drops it. |
| `first_index` | Each machine takes its own `seed`. |
| `threads`, `allocator`, `pin_threads`, `numa`, `pin_memory`, `debug`, `check_actions` | Keys of `[run]` in the machine config, or `config=`. `debug` also checks actions. |
| `done` as `uint8` | `done` as `bool` |
| `information['frame']` | `batch.frame()` |
| `final_raw()` | `information['final_raw']` |
| `batch.torch_device` | `batch.device`, a `torch.device` |
| `agent_mask`, `[E]` agent counts | Integers, the same in every environment. |
| `roles(index)`, `observation()`, observers | Private. Wrap `step` to count goals. |
| actions from `-action_count` | Actions from 0. Code that passed `-1` for idle passes 0. |
| out of memory raises `LibraryError` | `DeviceError` |
| `device='cuda'` without a GPU raises | `device='auto'` runs on the CPU, and `'cuda'` still raises `DeviceError` |
| `cuju.bundled_scenarios()` | `cuju.scenario_names()` |
| `bundled_assets`, `library_path`, `memory_estimate` | Removed. A small batch gives `memory_bytes`, and `CUJU_CUDA_LIBRARY` selects a library. |
| `MultiDeviceBatch` | One process per GPU with `seed=base + rank`. |
| `ActionRangeError`, `EnvironmentIndexError` | `ArgumentError` |
| `cuju.SignatureError`, `cuju.TrackerError` | `cuju.gfootball_engine` |
| `cuju.kaggle`, `cuju.players`, `cuju._kaggle` and its `evaluate` | `cuju.agents`, with the `kaggle`, `bot` and `lazy` controllers, `play` and `play_batch`. |
| `cuju.raw`, `cuju.state`, `cuju.wrappers`, `cuju.action_log` | Renamed `_raw`, `_state`, `_wrappers` and `_action_log`, which are private to the family of packages. |
| `ActionLog`, `Digest`, `replay` | Private. `cuju replay FILE` replays a dump. |
| `CUJU_NUMA_NODES`, `CUJU_RENDER_THREADS`, `CUJU_RENDER_WORKERS`, `CUJU_LOG` | The keys `numa`, `render_threads`, `render_workers` and `log`. |
| `CUJU_BUNDLE_CACHE` | Removed. The bundle cache is always on. |
| `import cuju` registering the Gymnasium ids | `import cuju.env`, or `gymnasium.make('cuju.env:GFootball-...')`. Gymnasium before 1.0 also loads them at its own import. |
| `cuju_gfootball`, `python -m cuju_gfootball` | `cuju.gfootball_alias`, `python -m cuju.gfootball_alias` |
| `cuju_grf_media` | `cuju.grf_media` |
| `cuju replay --trace_file FILE`, `cuju dump-to-txt --trace_file FILE --output OUT` | `cuju replay FILE`, `cuju dump FILE -o OUT` |
