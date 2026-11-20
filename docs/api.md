# API reference

The **application programming interface** of cuju comprises the Python package `cuju` and a C API
that both native libraries export. Every operation on a batch is reached from
`cuju.BatchEnvironment`, which creates a batch, steps and resets it, exposes its raw observations
and states, and raises the exceptions of `cuju.errors`. `cuju.MultiDeviceBatch` spreads one batch
over several devices, and submodules add the interfaces and tools of Google Research Football, or
GRF, around it.

Each option of a batch has a type, a default and, where one exists, a counterpart among the options
of GRF. In the shapes of tensors, `E` stands for `environment_count` and `A` for `batch.agents`.

## Batch creation

`cuju.BatchEnvironment(assets, **options)`

The options fall into four groups. The scenario, episode and output options determine which matches
a batch plays and what it returns, so two runs are comparable when they agree on all three groups.
The run options determine how the work is done and leave every output unchanged, so they may differ
between machines.

### Scenario options

The scenario options select the matches, their agents and their difficulty.

| Parameter | Type | Default | GRF | Meaning |
| --- | --- | --- | --- | --- |
| `assets` | `str` or `list[str]` | required | `env_name` | A shipped scenario with agents, such as `5_vs_5_l4_r0`, a bare scenario name, which gives one left agent, the path of a bundle, or the path of a scenario file. A list holds several. Described in [Scenarios](scenarios.md). |
| `assignment` | integers or `None` | `None` | | The bundle index of each environment, `[E]`. `None` gives every environment the first bundle. |
| `agent_counts` | list of pairs or `None` | `None` | `number_of_left_players_agent_controls`, `number_of_right_players_agent_controls` | One `(left, right)` pair per bundle, each `None` for what the name gives. |
| `difficulty` | list of pairs or `None` | `None` | `left_team_difficulty`, `right_team_difficulty` | One `(left, right)` pair per bundle, each from 0 to 1 or `None` for the scenario's. |
| `physics_steps_per_frame` | `int` | `10` | `physics_steps_per_frame` | The ticks in one round of a step. |
| `animations` | `str` or `None` | `None` | | The path of `animations.bin`. `None` takes the shipped tables. |

### Episode options

The episode options set the number of environments, their seeds and the sequence of episodes.

| Parameter | Type | Default | GRF | Meaning |
| --- | --- | --- | --- | --- |
| `environment_count` | `int` | `1024` | | The number of environments. |
| `seed` | `int` | `0` | | Environment `index` plays as GRF after `random.seed(seed * 2**32 + first_index + index)`, from `0` to `2**64 - 1`. Described in [Episodes and seeds](episodes-and-seeds.md). |
| `first_index` | `int` | `0` | | The index of environment 0 in the seed rule. Every index stays below `2**32`. |
| `episode_number` | `int` | `0` | `episode_number` | The number of the episode before the first. Its parity selects the form. |
| `auto_reset` | `bool` | `True` | | Starts the next episode inside the step that ends one. |
| `game_engine_random_seed` | `int`, sequence or `None` | `None` | `game_engine_random_seed` | One seed for every episode. A sequence gives one per environment, `-1` to draw. |

### Output options

The output options set the representation, the rewards and the action set.

| Parameter | Type | Default | GRF | Meaning |
| --- | --- | --- | --- | --- |
| `representation` | `str` | `'simple115v2'` | `representation` | `simple115`, `simple115v2`, `extracted`, `raw`, `pixels` or `pixels_gray`. Described in [Observations](observations.md) and, for the pixels, [Frames and videos](rendering.md#rendering-batches). |
| `with_raw` | `bool` | `False` | | Also writes the `raw` arrays, for `raw()` and `final_raw()`. |
| `render_frames` | `bool` | `False` | `render` | Renders every environment as GRF does and adds GRF's frames to the information, for `frame()`. |
| `render_resolution_x` | `int` | `1280` | `render_resolution_x` | The width of a rendering batch's frames, a multiple of 4 up to 16,384. |
| `render_resolution_y` | `int` | `720` | `render_resolution_y` | The height of a rendering batch's frames, up to 16,384. It stays 720 when only the width changes, as in GRF. |
| `stacked` | `bool` | `False` | `stacked` | Stacks the last four observations along the last axis. |
| `channel_dimensions` | `(int, int)` | `(96, 72)` | `channel_dimensions` | The width and height of the SMM. |
| `squeeze` | `bool` | `False` | the single-agent wrappers | Drops the agent axis. Requires one agent in every environment. |
| `rewards` | `str` | `'scoring'` | `rewards` | `scoring` or `scoring,checkpoints`. Described in [Rewards and actions](rewards-and-actions.md). |
| `numpy_legacy` | `bool` | `True` | | Adds the checkpoint reward in the precision of numpy before version 2. |
| `action_set` | `str` | `'default'` | `action_set` | `default` with 19 actions, `v2` with 20 or `full` with 33. |

### Run options

The run options select the device and the execution resources and leave every output unchanged.
[Backends](backends.md) describes each of them.

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `device` | `str`, `int` or `torch.device` | `'cuda'` | `cuda`, `cuda:N`, `cpu` or `auto`. |
| `threads` | `int` | `0` | The CPU threads, the caller included. `0` takes one per processor the process may run on, up to its cgroup CPU quota. |
| `library` | `str` or `None` | `None` | The path of the library of `device`. `None` takes `library_path(device)`. |
| `allocator` | `str` | `'cuda'` | The source of GPU buffers, `cuda`, or `torch` for the caching allocator of PyTorch. |
| `pin_threads` | `bool` | `False` | Binds each CPU worker to a processor of its own. |
| `numa` | `bool` | `False` | Forms one group of CPU workers per NUMA node, each with its own environments. |
| `pin_memory` | `bool` | `False` | Page-locks the output buffers of a CPU batch for asynchronous copies to the GPU. |
| `check_actions` | `bool` | `False` | Checks actions in GPU memory against `[-action_count, action_count)` before each step, at the cost of a wait for the device. The CPU checks every action anyway. |
| `debug` | `bool` | `False` | Waits for the device after every `step`, `step_async`, `reset_done` and `reset` of a GPU batch, so that a failure on the device raises at the call that caused it, and checks the actions as `check_actions` does. The CPU ignores it. |

The two checking options leave the outputs of every step that runs unchanged. They differ from the
other run options in one respect, since an action out of range in GPU memory then raises an error
where it would otherwise play as `idle`.

The constructor raises `ChoiceError` for an unknown `representation`, `action_set`, `device` or
`allocator`, `ConfigurationError` for malformed or inconsistent options, `MissingFileError` for an
asset that names nothing or a missing library, `DeviceError` for a GPU that PyTorch does not see or
a CPU without AVX2, BMI2 or FMA, and `LibraryError` when the library refuses the batch. A GPU batch
that runs out of memory at creation raises `LibraryError` with the memory it requested, the memory
it already held and the free memory of the GPU. A GPU batch whose request exceeds the free memory of
the GPU at the moment of creation, yet is granted, issues a `UserWarning` that names both amounts.
For the first batch of the process on a GPU, the request includes the local memory that the creation
reserves for the kernels. With `allocator='torch'`, the memory that the cache of PyTorch holds
unused counts as free. Under Windows and WSL2 the warning is the main sign that the driver placed
the batch in shared system memory, as described in [Performance](performance.md).

`cuju.memory_estimate(assets, **options)` returns the bytes that a batch with the same arguments
would request, without creating it, and `batch.memory_bytes` returns the bytes that a batch has
requested so far. Both count device memory on the GPU and host memory on the CPU, the buffers,
worlds and tables of the batch, and on the GPU the work area of 12 KB per environment that the
kernels use for their large lists. The estimate is an `int` with the attribute `local_memory_bytes`,
the local memory that the driver reserves for the kernels when the process creates its first batch
on the GPU and then holds until the process ends, about 5.6 KB for each thread that the GPU runs at
a time. The library computes it from the local memory of each kernel, the current stack of the
device and the threads that the GPU runs at a time, and it is 0 once the process has created a batch
on that GPU, and always 0 on the CPU. Every later batch of the process on that GPU shares the
reservation. The estimate equals the `memory_bytes` of the batch right after its creation. It reads
the bundles and animation tables as a creation does, takes a fraction of a second, and raises the
errors that the constructor raises for the same arguments. A few later calls add a small buffer
each, on the GPU the first `reset` with seeds, a reverse flag or a host mask, and on the CPU the
first `step_async`.

```python
import cuju

per_environment = cuju.memory_estimate(
    '5_vs_5_l4_r0', environment_count=2, device='cpu'
) - cuju.memory_estimate('5_vs_5_l4_r0', environment_count=1, device='cpu')
```

A batch dropped without `close()` is freed with its last tensor, and `close()` frees it at once.
Explicit closing suits the GPU, where the memory of a batch is large and the release of its last
tensor can be hard to locate. A batch is also a context manager, so
`with cuju.BatchEnvironment(...) as batch:` closes it at the end of the block, an exception
included, and entering a closed batch raises `ClosedError`.

## Loop methods

A program calls `reset` once, then `step` in a loop. The other methods serve a step split in two on
the CPU, the reset of ended environments without the automatic reset, and the current observation
without a step.

| Method | Returns | Meaning |
| --- | --- | --- |
| `reset(seeds=None, reverse=None, mask=None)` | observation | Starts the next episode of every environment, or of those where `mask` is true. `seeds` takes one per environment, as a list, tuple, array or tensor, or one integer for all, each from `0` to `2**32 - 1` or `-1` to draw. `reverse` takes the team-order flag in the same forms, read at the first reset of an environment. |
| `step(actions)` | `(observation, reward, done, information)` | Runs one step. `actions` is `[E, A]` of integers, or `[E]` with `squeeze`, and a contiguous `int32` tensor on the device of the batch is read in place. Releases the GIL while the library runs. |
| `step_async(actions)`, `step_wait()` | `None`, then as `step` | A step in two halves. On the CPU it runs in a thread of the batch meanwhile, and on the GPU `step_async` enqueues the step as `step` does. |
| `reset_done()` | `None` | Starts the next episode of every environment that ended, for `auto_reset=False`. |
| `observation()` | observation | The current observation, as the last call returned it. |
| `close()` | `None` | Frees the batch at once. Repeated calls are harmless. |
| `__enter__()`, `__exit__()` | the batch, then `None` | The context manager, which closes the batch at the end of a `with` block, an exception included. |

`step` returns the observation, the reward, the done flags and a dictionary of information with two
entries, whose shapes and types depend on the representation.

| Output | Shape and type |
| --- | --- |
| observation | `[E, A, 115]` `float32` for the vector representations, `[E, A, height, width, 4]` `uint8` for `extracted`, a dictionary of tensors for `raw`. A stack multiplies the last axis by 4 |
| reward | `[E, A]` `float32` |
| done | `[E]` `uint8` |
| `information['final_observation']` | as the observation, meaningful where `done == 1` |
| `information['score_reward']` | `[E]` `int32`, the scoring reward of the left team |

`step` raises `ActionRangeError` for an action out of range in host memory, or in GPU memory with
`check_actions` or `debug`, `TypeError` for floating-point actions, `ArgumentError` for actions of
another shape, `UndefinedPathError` after an environment reached a path that GRF leaves undefined,
`DeviceError` when the device fails in a `debug` batch, `ClosedError` on a closed batch, and
`LibraryError`, a `RuntimeError`, before the first reset. `step_async` checks its actions in the
same way, and `step_async`, `reset` and `reset_done` raise `DeviceError` when the device fails in a
`debug` batch. Every returned tensor shares memory with the batch and is overwritten by the next
`reset`, `step` or `set_state`. Actions, masks and seeds in host memory, page-locked or not, may be
changed as soon as `step`, `step_async` or `reset` returns, since a GPU batch copies page-locked
ones before the device reads them.

`step` and `reset_done` are also the PyTorch operators `cuju::step` and `cuju::reset_done`, so a
function that steps a batch compiles under `torch.compile` with `fullgraph=True`.

The errors of `step` fall into two kinds. A wrong type, a wrong shape, an action out of range in
host memory and a closed batch are errors of the calling code, raised before any environment steps,
so the batch keeps its state and the corrected call succeeds. Actions already on the GPU go to the
step as they are by default, so that a step never waits for the device, and an action out of range
there plays as `idle`. With `check_actions=True`, or with `debug=True`, the batch checks those
actions on the device before the step and raises `ActionRangeError` naming the first action out of
range, its agent and its environment, before any environment steps. The check reads its result on
the host, so every step waits for the device, and a CUDA graph capture skips it. As with actions in
host memory, the slots of padded agents go unchecked.

`UndefinedPathError` reports a match that reached a path that GRF leaves undefined. On the CPU the
library raises it from the step. On the GPU the device traps, which ends the CUDA context of the
process, so the next CUDA call reports the failure and the process restarts. A long run therefore
saves states with `get_state` at intervals, as described in
[Episodes and seeds](episodes-and-seeds.md) and [States](states.md). A GPU batch enqueues its work
and returns, so a failure on the device usually surfaces at a later call that waits. With
`debug=True` the batch waits after every step and reset and raises `DeviceError` at the call that
caused the failure, which locates a fault at the cost of the waits.

## Step observers

A **step observer** is a callable that a batch calls after each of its resets and steps. A program
can use an observer to collect statistics, such as goal counts, with the same stepping loop.

| Method or attribute | Returns | Meaning |
| --- | --- | --- |
| `add_observer(observer)` | the observer | Calls `observer` after every reset and step from now on, and once when the batch closes. |
| `remove_observer(observer)` | `None` | Stops calling an observer. |
| `observers` | `tuple` | The observers, in the order of their calls. |

The batch calls `observer(batch, done, information)` with the batch itself and two values that say
what happened.

| After | `done` | `information` |
| --- | --- | --- |
| `step`, `step_wait` | the done flags that the step returns | the information that the step returns, with the rewards that the step returns under `reward` and the actions that the step played under `actions` |
| `reset`, `reset_done` | `None` | `{'reset': mask}`, where `mask` holds `[E]` booleans of the environments that started an episode, or is `None` when every environment did |
| `close` | `None` | `{'close': True}`, after which the batch drops its observers |

```python
goals = [0]


def count_goals(batch, done, information):
    if done is not None:
        goals[0] += int(information['score_reward'].abs().sum())


batch.add_observer(count_goals)
```

The batch calls its observers once the outputs are written, in the order they were added, so each
observer sees the tensors that the caller receives and may read states with `get_state`. After a
step, `information` is a dictionary of its own, which adds `reward` and `actions` to the entries of
the dictionary that the step returns, so the caller's dictionary stays as `step` documents it. An
observer is expected to leave the outputs, the information and the states as they are, and a batch
whose observers do so plays every match as it would without them, bit for bit, with the same random
numbers. A batch without observers checks one attribute per step and does nothing more for them.

The entry `actions` holds the actions that the step played, `[E, A]` `int32` on the device of the
batch, every agent slot of every environment as the caller gave it, so a negative index that counts
from the end of the action set reads as given. Actions of one agent passed as `[E]` read as
`[E, 1]`. When the caller passes a contiguous `int32` tensor on the device of the batch, which the
step reads in place, the observer receives that same tensor. Actions in any other form, such as a
list, another integer type or host memory for a GPU batch, reach the observer as the tensor that the
step converted them to, on the device of the batch. By contract an observer only reads the tensor
and leaves it as it is, as it does the outputs. The observer runs before `step` returns, so the
tensor then holds exactly what the step played, and an observer that needs the actions after its
call keeps a clone, since the caller may write its tensor for the next step. A step split in two
differs here, because the caller may change its actions as soon as `step_async` returns. A batch
with observers therefore copies the actions at `step_async`, and `step_wait` hands its observers
that copy, which holds the actions that the step started with. A batch without observers copies
nothing for them, so the entry costs nothing until an observer is added.

The actions go with every step, including a step on which some environments end and start their
next episode through the automatic reset, so an observer receives on that step both the actions
that ended the episode and the done flags. Resets play no actions, so the calls after `reset`,
a reset of chosen environments with `mask` included, and after `reset_done` carry no `actions`
entry. The parts of a `MultiDeviceBatch` hand their observers the actions of their own
environments, on their own devices.

An observer that raises an `Exception` is removed with one warning through the logger `cuju`, and
the batch continues with its other observers. An observer may remove itself or another observer
during any call, including the close call. An observer removed before its turn misses that call.
`add_observer` raises `ArgumentError` for a value that is not callable or an observer already added,
and `ClosedError` on a closed or closing batch.
`remove_observer` raises `ArgumentError` for an observer that an open batch lacks, and once `close`
begins it does nothing, since the batch drops every observer.

An observer may also close the batch. The reset or step that called it then raises `ClosedError`,
since its outputs would view the freed buffers, and the observers after it miss that call. A batch
dropped without `close` frees its memory once no tensor that it returned remains, and makes no close
call. An observer that keeps the batch in an attribute forms a cycle with it, which holds the
batch's memory until Python's garbage collector breaks the cycle, so an observer that needs the
batch between its calls keeps a `weakref.ref` to it, as the observers of the renderer do. Each call
hands the observer the batch itself.

Steps and resets that run without Python call no observer. Those are the calls that a CUDA graph
captured and replays, and the operators `cuju::step` and `cuju::reset_done` inside a function that
`torch.compile` compiled. The parts of a `MultiDeviceBatch` take observers of their own, each
called with its part and the done flags of that part, whose environments count from 0.

## Raw observations

Two methods return the `raw` arrays of GRF as tensors.

| Method | Needs | Returns |
| --- | --- | --- |
| `raw()` | `representation='raw'` or `with_raw=True` | The `raw` arrays of the last step, a dictionary of tensors. |
| `final_raw()` | the same | The `raw` arrays of the step that ended each episode. |

A batch that renders, with `pixels`, `pixels_gray` or `render_frames=True`, also returns GRF's
frames.

| Method | Needs | Returns |
| --- | --- | --- |
| `frame()` | `pixels`, `pixels_gray` or `render_frames=True` | GRF's frame of every environment after the last reset or step, `[environment_count, render_resolution_y, render_resolution_x, 3]` `uint8`. |

## States

Two methods save and load the state of one environment.

| Method | Meaning |
| --- | --- |
| `get_state(index, stacked=False)` | The state of one environment as `bytes`, in the format of GRF. `stacked=True` adds the stack of four of GRF. |
| `set_state(index, state, stacked=False)` | Sets one environment to a state of cuju or of GRF, rewriting the returned tensors in place. |

`set_state` raises `StateError` for bytes that fail to load, a state of another scenario, agents,
difficulties or end flags, naming the fields that differ, or a stack that differs from that of the
batch, and `EnvironmentIndexError` for an index outside the batch. The format is described in
[States](states.md).

## Roles

One method reports the current role of each player in an environment.

| Method | Returns |
| --- | --- |
| `roles(index)` | `{'left_team_roles': ..., 'right_team_roles': ...}`, each `[players]` `int32` on the CPU, one entry per player of the team |

A role is GRF's `e_PlayerRole` as a number, from 0 for the goalkeeper to 9 for the centre forward,
as listed in [Observations](observations.md#raw-observation). The players come in the order of the
raw observation of a left agent, so the two tensors equal the raw arrays of the same names without
their padding, and a batch of any representation reports them. Each call copies a few bytes of the
environment to the host, after waiting for the device as `get_state` does, and changes nothing in
the batch, so its outputs and its matches stay as they are, bit for bit.

The roles change during a match in one case. When a goalkeeper is sent off, GRF hands the keeper's
formation entry, the role of goalkeeper and the keeper's place in the formation, to the first
active player of the same team, who plays in goal from then on. GRF stores the reassignment in
the environment's team data, so the state from `get_state` contains the scenario's roles. The
`roles` method and the raw observation both report the reassignment. A state loaded with
`set_state` keeps the reassignment of the environment that loads it, so the same state shows the
replacement in goal in the environment where the keeper was sent off and the scenario's roles in
a fresh one, in GRF and in cuju alike, as [States](states.md#scope-of-a-state) describes.
`roles` raises `EnvironmentIndexError` for an index outside the batch.

## Attributes

A batch exposes its configuration and layout as attributes.

| Attribute | Type | Meaning |
| --- | --- | --- |
| `device` | `str` | `cuda` or `cpu`, the backend |
| `torch_device` | `torch.device` | the device of the buffers, such as `cuda:0` |
| `environment_count` | `int` | the number of environments |
| `threads` | `int` | the CPU threads, the caller included, 0 on the GPU |
| `agents` | `int` | the room for agents in each environment |
| `players` | `int` | the room for players in each team array of `raw` |
| `action_count` | `int` | the actions of the action set |
| `left_agents`, `right_agents` | `[E]` `int32` | the agents of each team of each environment |
| `agent_mask` | `[E, A]` `bool` | the agents that exist |
| `left_players`, `right_players` | `[E]` `int32` | the players of each team, from the view of a left agent |
| `difficulty` | `[E, 2]` `float32` | the difficulties that play, the scenario's where nothing overrides them |
| `squeeze`, `stacked`, `channel_dimensions`, `auto_reset`, `check_actions`, `debug` | | the options of the batch |
| `scenario_name` | `str` | the scenarios that the batch was created with, by their names without folders or suffixes, joined by commas, such as `5_vs_5_l4_r0` |
| `observers` | `tuple` | the step observers, described in [Step observers](#step-observers) |
| `memory_bytes` | `int` | the bytes of memory that the batch has requested so far, device memory on the GPU and host memory on the CPU |
| `single_observation_space`, `single_action_space` | spaces | one environment, as the spaces of GRF, with a `None` observation space for `raw` |
| `observation_space`, `action_space` | spaces | the whole batch |

The spaces require `gymnasium`, installed with the extra `spaces`, or `gym`.

## Several devices

`cuju.MultiDeviceBatch(assets, devices, environment_count=1024, split=None, output_device=None, assignment=None, game_engine_random_seed=None, **options)`
holds one batch per device and steps them together. Environment `i` plays as environment `i` of one
`BatchEnvironment`, whatever the split, through `first_index`. `devices` may mix GPUs and the CPU,
`split` sets the environments of each part, by default an even split, and `options` are those of
`BatchEnvironment` except `device` and `first_index`, which each part takes from `devices` and
`split`. The outputs of `reset` and `step` are gathered on `output_device`, by default the first
device, and `batch.parts` holds the parts, each with its own outputs. Its `reset` takes `seeds` and
`reverse`, and `get_state`, `set_state` and `roles` take the index of an environment over the whole
batch.
Step observers attach to the parts, one part at a time, as in `batch.parts[0].add_observer(f)`,
and `scenario_name` is that of the parts.
`close()` closes every part, after which every other call raises `ClosedError`. The layouts are
described in [Backends](backends.md).

## Module functions

The package also exports functions and classes for scenarios, digests and action logs.

| Name | Returns or does |
| --- | --- |
| `cuju.bundled_scenarios()` | the names of the shipped scenarios, such as `5_vs_5` |
| `cuju.bundled_assets()` | the names of the shipped bundles, such as `5_vs_5_l0_r0` |
| `cuju.library_path(device='cuda')` | the path of the library that a batch on `device` loads |
| `cuju.memory_estimate(assets, **options)` | the bytes that `BatchEnvironment(assets, **options)` would request at creation, without creating it, with the reservation of a first GPU batch for the kernels as `local_memory_bytes` |
| `cuju.__version__` | the version of the package |
| `cuju.Digest()` | a running SHA-256 of tensors, dictionaries, lists and bytes, with `update`, `copy` and `hexdigest` |
| `cuju.digest(assets, device='cpu', steps=100, seed=7, environment_count=4, **options)` | the self-check digest of a configuration |
| `cuju.ActionLog(assets, **options)` | a batch whose calls are journaled, with `reset`, `step`, `reset_done`, `set_state`, `save(path)` and `hexdigest()` |
| `cuju.replay(path, device=None, **options)` | plays an action log again and returns its digest, raising at the first call that differs |

## Exceptions

Every class lives in `cuju.errors`, is exported by `cuju`, and derives from `cuju.Error` and from a
built-in type, so that `except ValueError` and similar clauses also catch it.

The double ancestry allows handlers at three levels of precision. `except cuju.Error` catches every
exception that cuju raises and no other, as suits a training script that logs and restarts. A
specific class, such as `StateError` around `set_state`, catches one failure. Code written against
the built-in types, such as code written for GRF, keeps catching the type it catches.

Each class derives from `cuju.Error` and from the classes in its second column.

| Class | Also a | Raised for |
| --- | --- | --- |
| `Error` | `Exception` | the base of all |
| `ConfigurationError` | `ValueError` | an option refused at creation. `scenarios.ScenarioError` subclasses it |
| `ChoiceError` | `ConfigurationError`, `KeyError` | an unknown `representation`, `action_set`, `device` or `allocator` |
| `ArgumentError` | `ValueError` | an argument of a call, such as actions of the wrong shape, or `reset(seeds=, reverse=, mask=)` of the wrong length or range |
| `ActionRangeError` | `IndexError` | an action outside `[-action_count, action_count)` |
| `EnvironmentIndexError` | `IndexError`, `RuntimeError` | an environment index outside the batch |
| `StateError` | `ValueError`, `RuntimeError` | a state that fails to load into its environment |
| `ClosedError` | `RuntimeError` | any call on a closed batch |
| `DeviceError` | `RuntimeError` | a GPU that PyTorch does not see, a CPU without AVX2, BMI2 or FMA, a CUDA failure |
| `UndefinedPathError` | `RuntimeError` | an environment that reached a path that GRF leaves undefined |
| `SignatureError` | `TypeError` | arguments that a function of `cuju.gfootball_engine` does not take, with the message of GRF's `Boost.Python.ArgumentError` |
| `TrackerError` | `RuntimeError` | two engines under GRF's tracker whose states differ, where GRF ends the process |
| `LibraryError` | `RuntimeError` | every other failure the library reports |
| `MissingFileError` | `FileNotFoundError` | a missing scenario file, bundle or library, or missing GRF media for the software renderer |

## Modules

Each submodule is imported by name, as in `import cuju.dumps`, and imports the packages it depends
on, such as numpy for `cuju.kaggle`, at that point. The core of the package depends on PyTorch
alone. `cuju.env` needs Gymnasium, from the extra `spaces`, or GRF's gym. `cuju.gfootball` needs
`absl-py` and `six`, which cuju installs, and the installed gym wherever GRF runs on it, else a copy
of gym 0.21's core that it carries. `cuju.gfootball_engine` and `cuju.scenarios` run without
PyTorch. `import cuju` leaves Gymnasium and gym unloaded and registers the `GFootball-*-v0` ids of
`cuju.env` with Gymnasium as soon as it is imported, before or after cuju. `import cuju.gfootball`
registers GRF's ids of GRF's environment with the gym it uses, as importing `gfootball` registers
them in GRF.

| Module | Names |
| --- | --- |
| `cuju.scenarios` | `build_bundle(scenario, out, name=None, device='cpu', library=None, animations=None)`, `describe(scenario, episodes=...)`, `load_scenario(path)`, `cached_bundle(scenario, directory=None, device='cpu', library=None, animations=None)`, `cache_directory()`, `ScenarioError`, and the builder API of GRF, `Builder`, `ScenarioConfig`, `PlayerRole`, `Team`. Described in [Custom scenarios](custom-scenarios.md) |
| `cuju.env` | `create_environment(env_name, ..., *, device='cpu', threads=1, api='gym')` with the arguments and defaults of GRF, the gym 0.21 interface of GRF by default and the Gymnasium interface with `api='gymnasium'`, and `register_environments()` for the `GFootball-*-v0` ids. Described in [Migrating from GRF](migrating-from-grf.md) |
| `cuju.dumps` | `DumpWriter(batch, logdir, environments=None, write_full_episode_dumps=True, write_goal_dumps=False, dump_frequency=1)` with `record(actions)`, `close()` and `paths`, and `read_dump(path)` |
| `cuju.players` | `Bot(batch, agent)` and `Lazy(batch, agents)`, the `bot` and `lazy` players of GRF, each with `act()`, and `Bot.reset(mask=None)` |
| `cuju.raw` | `grf_observation(raw, environment, agent, own_players, other_players)`, `grf_observations(batch, environment, agents=None, final=False)` and `team_sizes(batch, environment, agent)`, the raw dictionaries of GRF in the types of GRF |
| `cuju.wrappers` | `MultiAgentToSingleAgent(batch, left_players, right_players, action_set=None, stacked=False, squeeze=False)`, the wrapper of GRF for scenarios that set `control_all_players` |
| `cuju.kaggle` | `players_raw(batch, environment, agent, final=False)`, `observation(batch, environment, agent, step=0, final=False, raw=None)`, `KaggleAgent(function, batch, agent, configuration=None, agents=None)` and `evaluate(left, right=None, episodes=1, scenario='11_vs_11_kaggle', seed=0, device='cpu', threads=0, configuration=None)`, the observations and agents of Kaggle, and matches of Kaggle agents, each the first episode of its environment, as Kaggle plays them. `AgentRunner(function, configuration=None)` with `act(obs, count)`, `accepts_configuration(function)`, `action_index(answer, count)` and `action_indices(answer, count, players)` run a Kaggle agent function and check its answers |
| `cuju.gfootball` | GRF's Python package `gfootball`, every module ported verbatim with its import lines changed, `env` with `create_environment` and `create_remote_environment`, `config`, `football_env`, `football_env_core`, `football_action_set`, `observation_preprocessing`, `observation_processor`, `scenario_builder`, `wrappers`, `players`, `scenarios`, `play_game`, `replay`, `dump_to_txt`, `dump_to_video`, `eval_server` and `examples`, so that GRF code runs with its imports switched. Described in [Migrating from GRF](migrating-from-grf.md) |
| `cuju.gfootball_engine` | GRF's engine package `gfootball_engine` over cuju's engine, `GameEnv`, `ScenarioConfig`, `GameConfig`, `FormationEntry`, `Vector3`, `Position`, `SharedInfo`, `PlayerInfo`, `ControllerInfo`, the vectors `FormationEntryVec`, `PlayerInfoVec`, `ControllerInfoVec`, `FloatVec`, `IntVec` and `StringVector`, and the enumerations `GameState`, `e_PlayerRole`, `e_GameMode`, `e_BackendAction`, `e_Team` and `e_RenderingMode`, with the conversions and errors of GRF's compiled module. Described in [Differences from GRF](differences-from-grf.md#grfs-python-package) |
| `cuju_gfootball` | the package cuju-gfootball, from `pip install cuju-gfootball`, which imports `gfootball` and its submodules as `cuju.gfootball` and `gfootball_engine` and `_gameplayfootball` as `cuju.gfootball_engine`, with `active()`, whether GRF's names reach the port, `status()`, a sentence that says so or names the install of GRF that keeps them, `__version__`, and the command `python -m cuju_gfootball`, which prints the status and exits with 1 where the alias is inactive. Described in [Migrating from GRF](migrating-from-grf.md#grfs-names-through-cuju-gfootball) |
| `cuju.check` | `cuju check`, with `[--long] [--device all\|cpu\|cuda] [--case NAME] [--list] [--json]` |
| `cuju.state` | `layout()`, every field of the match in memory by path, `gather(state, fields, frame='rest')`, the bytes of chosen fields of a state, `read(state, fields, frame='rest')`, the same as numpy arrays, `Field` and `FRAMES`. Described in [States](states.md#reading-fields) |

`cuju.kaggle.AgentRunner` calls a Kaggle agent function as Kaggle does, with the configuration
second when `accepts_configuration(function)` finds two positional parameters, and its
`act(obs, count)` returns one action index per controlled player, checked against an action set of
`count` actions. One controlled player may answer in Kaggle's single form, such as `[12]`, `12` or
an `Action` member, and several answer with one action each. `action_index(answer, count)` and
`action_indices(answer, count, players)` check an answer alone and raise `ValueError`, `TypeError`
or `IndexError` for one outside the set. With a list of agents, `observation` builds Kaggle's
multi-agent form, and `raw` takes one host copy of `batch.raw()` that several observations share.

```python
from cuju import kaggle

runner = kaggle.AgentRunner(agent)
host = {name: tensor.cpu() for name, tensor in batch.raw().items()}
obs = kaggle.observation(batch, 0, [0, 1, 2], step=0, raw=host)
actions = runner.act(obs, batch.action_count)
```

## Command line

The engine installs `cuju`, which also runs as `python -m cuju`. Bare `cuju` and `cuju help`
show the engine commands followed by the installed family commands. Use `cuju check --help` or
`cuju help check` for the options and examples of one command. For a command in another package,
the same form works with more words. For example, `cuju help arena team add` shows the help from
`cuju arena team add --help` when the arena is installed.

Run `cuju --version` or `cuju -V` to see one line per distribution. The engine comes first, then
the distributions that provide commands, then the umbrella distribution `cuju` when it is
installed. A misspelled command exits with status 2 and can suggest a similar installed or
known family command.
If the arena or renderer is missing, asking for it exits with status 3 and shows its installation
command.

Use `cuju check` to verify the package data and compare matches with the shipped digests. The
default checks all available backends. Select a backend with `--device cpu` or `--device cuda`,
and use `--list` to see the available cases before you select one with `--case NAME`.
With `--json`, standard output is one object. Its first key is `schema`, with the value
`engine.check/1` for the check or `engine.check.list/1` for the case list. A usage error or a check
that cannot run on the requested backend uses `error/1`. Diagnostic and progress lines go to
standard error. Use `--quiet` to hide them while keeping errors visible. The case list runs
without collecting machine diagnostics. On a terminal, diagnostic lines wrap to the available
width and case progress updates on one line. If no backend can run, the check exits with status
3 unless package data already failed the check.

`cuju replay --trace_file FILE` plays a dump again and compares each step with the recorded step.
`cuju dump-to-txt --trace_file FILE --output FILE` writes a dump as text and prints the output path.
Both accept their existing GRF flags, which their help pages describe. A missing or unreadable
dump gives an error and a hint, with exit status 1.

Subcommand diagnostics use the name `cuju` followed by the command, including when the launcher is
`cuju.exe`. Paths in errors and hints appear in single quotes with their original backslashes.
The CLI uses Python string notation when a path contains a quote or control character. If the
reader closes a pipe connected to standard output, the command exits quietly with status 0.

Use `--json` to read a replay result under `engine.replay/1`. Its fields give the number of
equal steps, whether the replay ended the episode, and the first difference as text or `null`.
A difference keeps exit status 1. The text exporter uses `engine.dump_to_txt/1` and puts the
written path in `files`. Input, usage and environment errors from either command use
`error/1`.

The commands use exit status 0 for success, 1 for a negative result, 2 for a usage error and 3
when the environment cannot run the command. An unexpected error exits with status 70. Add
`--verbose` before or after the command to see its traceback. Ctrl+C exits with status 130,
and a closed output pipe exits quietly.

Use `cuju completion bash`, `cuju completion zsh`, `cuju completion fish` or
`cuju completion powershell` to generate a completion script. Save and source the script in Bash,
Fish or PowerShell. In Zsh, run `autoload -Uz compinit` and `compinit` before sourcing the
script. You can also save it as `_cuju` in a directory on `fpath` before running `compinit`.
Press Tab to complete commands, options and choices from the packages installed when you
generated the script. Generate it again after installing or removing a package.

A package can add commands by declaring entry points in the group `cuju.commands`, with one
`main(arguments)` function per command. The module of the entry point may expose a `parser()`
function that builds its argparse tree without side effects or heavy imports. Family help and
completion read that tree. Help shows the distribution summary when the package provides no
parser hook.

Such packages install as modules `cuju.<name>`, and cuju imports one only when a command or a
script asks for it, so `import cuju` and the steps of a batch leave them unloaded and read no entry
points. An editable install of such a package keeps its module in its own folder `cuju`, which
`import cuju` joins to the package, as `pkgutil.extend_path` does. Where `cuju.<name>` is missing,
`import cuju.<name>` and `python -m cuju.<name>` end with one message that suggests
`pip install cuju-<name>`.

The message comes from a finder that `import cuju` adds last to `sys.meta_path`. For a missing
optional module, `importlib.util.find_spec` raises the same `ModuleNotFoundError` as an import.
When importing an optional module in a `try` block, compare the exception's `name` with the
module name to distinguish a missing module from a missing dependency of that module.

```sh
cuju help
cuju check --device cpu
```

## Environment variables

The library variables select a rebuilt library without reinstalling the package, and the others
control logging, caching, NUMA placement, the software renderer and the hooks of batches. All of them leave the outputs
unchanged, the frames of the software renderer included.

| Variable | Meaning |
| --- | --- |
| `CUJU_CUDA_LIBRARY` | the path of the CUDA library to load |
| `CUJU_CPU_LIBRARY` | the path of the CPU library to load |
| `CUJU_LOG` | `1` logs one line per batch created, through the logger `cuju` |
| `CUJU_BUNDLE_CACHE` | `1` or a directory, to cache the bundles built from scenario files |
| `CUJU_NUMA_NODES` | the NUMA node of each processor, for `numa=True` |
| `CUJU_RENDER_THREADS` | the threads that draw each frame of the software renderer, by default the processors left to each worker of a batch, and for `cuju.gfootball` the processors up to four |
| `CUJU_RENDER_WORKERS` | the renderers of a batch that draw at once, by default one per environment up to the processors |
| `CUJU_GRF_MEDIA` | a folder of GRF's media for the software renderer, in place of the installed cuju-grf-media |
| `CUJU_RENDER_LIBRARY` | the path of the software renderer's library to load |
| `CUJU_BATCH_HOOKS` | The names of batch creation hooks, separated by commas. Each name is an entry point in the group `cuju.batch_hooks` for a function that receives the new batch. The logger `cuju` warns once per process for a missing hook and warns for the affected batch if a hook fails. The batch continues after either warning. |

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
`set_state` wait for the device. The headers state the contract of each call. The `version` field of
`cuju_configuration` names the layout of the structure, and `cuju_environment_create` refuses a
configuration whose version differs from the `CUJU_CONFIGURATION_VERSION` of the library, which is
3 in cuju 0.3.0.

`cuju_environment_roles(handle, index, roles, capacity)` writes the role of each player of one
environment, the left team's then the right team's, as the method `roles` returns them, and
returns their count, or -1 when the index lies outside the batch or `capacity` holds too few.

Three calls read the state format without a batch. `cuju_state_layout(&size)` returns the layout
of the match in memory as text, one field per line with its path, offset, size and type.
`cuju_state_gather(bytes, size, frame, offsets, sizes, count, out)` loads a match, the bytes of a
state after its pickle length and pickle, and copies the chosen fields into `out`, in the rest frame
for `frame` 0 or the frame of the pitch for 1. `cuju_state_padding(bytes, size, mask)` marks the
bytes of a match that GRF leaves uninitialized, so two states compare without them.

Render mode and the state of a render have calls of their own, which a renderer uses.
`cuju_environment_set_render(handle, mask, render_mode, referee_animations)` turns on GRF's render
mode and referee animations for the later resets of the chosen environments, so that they play the
matches of a rendering GRF. `cuju_environment_render_states(handle, environments, count, states,
size, processed)` fills a `RenderState` for each chosen environment, the structure of
`include/cuju/render/render_state.h` that holds what GRF's graphics read at a render, and the count
of state saves and loads of each, after which GRF draws the goal nets afresh. On the CUDA library
the extraction runs on the device and one copy brings the states to the host. Each environment
carries its `RenderState` from call to call, as GRF's widgets carry the shown score and the cards,
and starts it afresh at each reset, so a renderer calls it after every step it draws.
`cuju_engine_render_state_version()` and `cuju_engine_render_state_size()` name the version and the size of the
structure, and the call refuses a `size` that differs. The engine of `cuju.gfootball` has the
counterparts `cuju_engine_set_render`, `cuju_engine_prepare_render`, which applies what a render
of GRF changes in the match, `cuju_engine_render_state`, `cuju_engine_render_start`, the state that
starts a match's graphics, and `cuju_engine_internal_render_state`, the state of the match that a
rendering GRF engine builds at `start_game`. A renderer that draws every render of a batch, as the
representation `pixels` does, turns on `cuju_environment_enable_render_events(handle, capacity)`,
after which the batch records each environment's renders in render mode on its device, the start of
each match and every render with its RenderState, and `cuju_environment_render_events(handle,
index, events, capacity, &count)` moves an environment's events to the host in their order, each
`cuju_environment_render_event_size()` bytes. The software renderer has a C API of its own, in
`src/render/render_api.h`, whose calls follow GRF's graphics calls, as
[Frames and videos](rendering.md#the-renderstate) describes.

Four further calls serve memory planning and debugging.
`cuju_memory_estimate(&configuration, &bytes)` writes the bytes that `cuju_environment_create`
would request for the same configuration, without allocating them, and
`cuju_environment_memory_bytes(handle)` returns the bytes that a batch has requested so far.
`cuju_local_memory_bytes(&bytes)` writes the local memory that a creation on the current CUDA
device adds for the kernels, which is 0 once a batch exists there and on the CPU.
`cuju_environment_synchronize(handle, stream)` waits for the work of the batch on `stream` and fails
as the device failed, and it returns at once on the CPU. A library built with the CMake option
`CUJU_NVTX` wraps the creation, the resets, the steps and the state calls in NVTX ranges, as
described in [Backends](backends.md).

## See also

- [Concepts](concepts.md), for batches, bundles and environments
- [Recipes](recipes.md), for complete programs built on the package
- [Common topics](faq.md), for the behaviours that most often prompt questions
