# Differences from GRF

The **differences from Google Research Football**, or GRF, are the points at which cuju behaves
otherwise than GRF 2.10.3 while it reproduces GRF bit for bit within the scope of its
[exactness](exactness.md) claim. They lie in the interface of a batch, in the inputs that cuju
validates, in a bound of its engine, in the parts of GRF it ports and in the frames that it
renders. Behaviour outside these differences follows GRF, and issues reporting any further
difference are welcome.

Three computations are known to give other bits than GRF. Two of them arise in no match that a
batch accepts, and the third concerns GRF's loader of states, where cuju continues the saved match
exactly. The interface is that of a batch of environments returning PyTorch tensors, with an
automatic reset inside the step, with GRF's single-environment interface available as
`cuju.env.create_environment`, and with GRF's own Python package, ported verbatim, in
`cuju.gfootball`. cuju validates scenarios and states with named errors where GRF crashes or hangs,
keeps a bounded history of positions, and renders GRF's frames with a software renderer of its own
in `cuju.gfootball` and in batches that ask for frames. Batches with rendering disabled play the
matches of unrendered GRF.

## Output differences

The known differences in output concern three computations, each with a bounded reach, which the
table compares.

| Computation | cuju | GRF | Reach |
| --- | --- | --- | --- |
| the time a player of the built-in AI needs to reach the ball, an unsigned 32-bit count of milliseconds, in three tests of the AI | converts the time straight to `float` | converts the time to `float` through `int` | from 2^31 milliseconds, while the engine draws the time from the ball's predictions a few seconds ahead, so no match reaches it |
| the reciprocal inside the two `Normalize` helpers of the vector class, and any other float operation, for a NaN input | gives the GPU's NaN bit pattern on the GPU, and the host's on the CPU as GRF does | gives the host's NaN bit pattern | no output, since the engine passes no NaN there |
| the frame in which each of the AI's mental images stores its ball predictions, after `set_state` | in a batch and in `cuju.env`, sets the frame of every image from the state and continues the saved match exactly, and in `cuju.gfootball` loads as GRF does | keeps the frame outside the state, and when the environment holds fewer images than the state, adds the missing ones with their predictions mirrored, so the AI reads wrong predictions until new images replace the added ones, about every 100 milliseconds, and the match drifts | the steps after a `set_state` whose state has `reverse_team_processing` on, a non-deterministic scenario with an odd engine seed, into a GRF environment that holds fewer images than the state, such as a freshly reset one |

Every other output equals GRF's, bit for bit, within the scope of the [exactness](exactness.md)
claim. The third row is the reason a comparison of the two engines from a state loads GRF's state
into cuju, as described in [States](states.md#format).

### Variation within GRF

Two behaviours vary between runs or machines of GRF itself and affect comparisons.

GRF's state bytes vary from run to run. GRF writes the padding between the fields of its engine's
structures as it finds it in memory, so two identical GRF runs give states that differ in a few
hundred padding bytes. cuju writes zeros there, so its states of one match are identical byte for
byte, and states of the two engines are compared by their fields or by loading them.

GRF's `bot` player varies with the processor. It computes norms and dot products with numpy, whose
BLAS may fuse a multiplication and an addition on a processor with AVX-512, where GRF's own bot can
then decide differently in the last bit. cuju's `bot` computes as numpy does without fusing,
as on an AVX2 processor, so it gives the same decisions on every machine.

## Interface

cuju is a batch of environments, and the differences of its interface follow from that. The table
gives each with its reason.

| GRF | cuju | Reason |
| --- | --- | --- |
| one environment per `create_environment` | `cuju.BatchEnvironment` with `environment_count` environments | to run thousands of matches per call, with `cuju.env.create_environment` giving GRF's interface for one |
| observations as numpy arrays | PyTorch tensors that view the buffers of the batch, overwritten by each call | copying would cost more than the simulation on the GPU |
| one observation per agent, the agent axis dropped for one agent | `[environment_count, agents, ...]`, the agent axis always kept, which `observation[:, 0]` drops for one agent | one fixed shape for every number of agents |
| default `representation='extracted'` | default `simple115v2` | a vector occupies 460 bytes per agent, and an SMM of thousands of environments occupies gigabytes |
| `raw` values in numpy's types | `float32`, `int32`, `bool` and `uint8` tensors, holding GRF's values | one fixed type per array on every device |
| `info` with `score_reward` | `information` with `score_reward` and `final_observation` | the automatic reset hands over the last observation of an episode in the step that ends it |
| a step after the end of an episode raises | the step that ends an episode starts the next one with `auto_reset=True`, the default, and with `auto_reset=False` the environment stays at its end, with a reward of 0, until `reset_done` or `reset` | stopping a batch for the few environments that ended would idle the rest |
| `create_environment` resets in its constructor, so a step before `reset` plays the first episode | a batch steps after its first `reset`, or after `set_state` of every environment, and raises `LibraryError` before | the batch allocates its worlds without playing |
| actions are always checked | actions in host memory are checked as GRF checks them, and actions already on the GPU play as `idle` when out of range unless `debug` is on in the machine config, which checks them and raises `ArgumentError` | a check on the GPU makes every step wait for the device, so it is a setting |
| the pickle inside a state is written with Python's default protocol, wrapper keys first | protocol 2, the episode logic first | the batch writes the pickle itself, and both sides read each other's states |
| dumps are limited by wall-clock time, a full episode at most every 10 seconds | `DumpWriter` writes every selected episode and goal | a batch plays many episodes per second of wall-clock time |
| a dump is named by its kind and the time to the microsecond, `<kind>_<date>-<time>.dump` | the same names from the dump options of `cuju.env.create_environment`, and names by environment, episode and goal step from `DumpWriter` over a batch | the environments of a batch can start dumps in the same microsecond, where their names would collide |

`cuju.env.create_environment` follows GRF's interface for one environment, with three differences.
A step after the end of an episode raises `RuntimeError`, where GRF raises `AssertionError`. An
action given as one of GRF's `CoreAction`s belongs to the environment's action set, and one from a
larger set raises `ValueError` with the name of the set to create. With no agent on any side,
`simple115` is an empty array of shape `(0, 115)`, where GRF's is `(0,)`, so the observation keeps
its last axis.

## Input validation

cuju validates scenarios when it builds them and states when it loads them. For inputs on which GRF
crashes, hangs or silently changes its behaviour, it raises a named error, as the tables list.

The batch and `cuju.env.create_environment` build each scenario into a bundle, which holds the two
parities of the episode number, and they check the scenario and the states against it.

| Input | GRF | cuju |
| --- | --- | --- |
| a scenario with a team of more than 11 players | crashes | `ScenarioError` |
| a scenario whose team's first player is not a goalkeeper | hangs in open play | `ScenarioError` |
| a scenario that draws from `random` or NumPy's global generator | runs, and shifts the seeds of every later episode | `ScenarioError` |
| a scenario that reads `controllable_left_players`, `controllable_right_players` or `dynamic_player_selection` while it builds | freezes the value of the first read, before the agents are set | `ScenarioError` |
| a scenario that depends on more than the parity of the episode number, or whose team sizes change with it | runs | `ScenarioError` |
| a deterministic scenario that turns `reverse_team_processing` on | runs | `ScenarioError` |
| a scenario with `control_all_players` and agents that would switch players | fails an assertion in its engine | `LibraryError` with GRF's condition |
| a state whose scenario differs from the environment's outside the constant part, such as in its formation entries | stops only when the constant part differs, in the ball, the team sizes, the agent counts, the difficulties, the flags and the duration | `StateError` for any difference but the seed, naming up to three fields |
| with `rewards='scoring,checkpoints'`, a state without checkpoint counts | `KeyError` | `StateError` |
| a stacked state loaded into an unstacked environment, or an unstacked state into a stacked one | ignores the stack of the first, and raises `KeyError` for the second | `StateError` for both |

`cuju.gfootball` runs GRF's Python as it stands, so every scenario that GRF's Python builds plays
there. The engine underneath, `cuju.gfootball_engine`, checks only the calls on which GRF's engine
ends the process, crashes or hangs.

| Call of `cuju.gfootball_engine` | GRF | cuju |
| --- | --- | --- |
| an index outside 0 to 2 of a `Vector3` or a `Position` | logs a fatal error and ends the process | `IndexError` |
| `get_info`, `step` or `get_state` before the first `reset` | crashes | `LibraryError` |
| `start_game` a second time | fails an assertion | `LibraryError` |
| `sticky_action_state` of an action other than a direction or a sticky button | logs a fatal error | `ArgumentError` |
| `perform_action` or `sticky_action_state` of a player whose keyboard lies outside the 22 of the engine | reads past its keyboards | `ArgumentError` |
| `set_state` of a state of another scenario, compared over GRF's constant part | logs a fatal error and crashes | `StateError`, naming up to three fields |
| `render`, or a step with the rendering on, in a match that its reset built while `game_config.render` was off, as after `disable_render` and a reset | crashes, redrawing a name tag that never took its text | `StateError` at that render |
| a scenario of more than 11 players a team, or whose team's first player is not a goalkeeper | crashes, or hangs in open play | `ScenarioError` |
| a scenario with more agents on a side than players | writes past the controllers of `get_info` | `ScenarioError` |
| a match longer than the position history, as [below](#position-history) | plays | `ScenarioError` |
| `physics_steps_per_frame` below 1 | never brings the match into play, so `reset` hangs | `ArgumentError` |
| a write of `reverse_team_processing` to `GameEnv.config` during a match | processes the teams in one order and mirrors them in the other | `ScenarioError` at the next call |
| a change of a team's size in `GameEnv.config` during a match | reads formation entries that no player has | `ScenarioError` at the next call |
| two engines under `tracker_setup` whose states differ | ends the process | `TrackerError` in both threads |

## Engine

The engine of cuju bounds one quantity that GRF lets grow, and it equals one reference build of
GRF. At paths that GRF leaves undefined, it stops with a named error.

### Position history

GRF keeps one position per player for each second between kick-offs, in a list that grows without
bound. A batch keeps 512 seconds of that history, which holds every shipped scenario, and accepts at
creation every match of up to 51,199 ticks of play, `game_duration` times `physics_steps_per_frame`,
which is 5,119 steps at the default of 10 ticks a step. A longer match raises an error at creation,
and `build_bundle` warns at build time and names the largest `physics_steps_per_frame` that fits.
The bound keeps the state of every environment at a fixed size, which a batch of thousands of
environments on a GPU requires.

### Reference build

cuju equals GRF as GCC builds it for x86-64 Linux with glibc 2.39, on a processor with FMA. A GRF
built for another platform or C library can play other matches, for the reasons given in
[Exactness](exactness.md#reference-build).

### Undefined paths

GRF leaves a few paths of its engine undefined, such as a set piece without a taker, and at such a
path it has no result for cuju to equal. cuju stops there, on the CPU with `cuju.errors.UndefinedPathError`
naming the environment and GRF's reason, and on the GPU with a trap of the device. Every shipped
scenario avoids them, as far as is known. The behaviour on each device, and the checkpoints that
protect a long run, are described in [Episodes and seeds](episodes-and-seeds.md#undefined-paths).

## GRF's Python package

`cuju.gfootball` is GRF's Python package, its modules ported verbatim with only their import lines
changed, over `cuju.gfootball_engine`, a module with the names, types and behaviour of GRF's
compiled `gfootball_engine`. Code written for GRF therefore runs on it with its imports changed, its
reach into `env.unwrapped._env` and the private attributes of GRF's classes included. Six details
of that boundary call for a description.

The pickle inside a state names GRF's modules. GRF's Python, running under `cuju.gfootball`,
pickles its classes under the names `cuju.gfootball...`, and the engine translates them to
`gfootball...` when it saves and back when it loads, so the states of `cuju.gfootball`, of GRF and
of `cuju.env` load into one another. The translation rewrites module names only, without
unpickling, and keeps the pickle's frames. `set_state` of `cuju.gfootball` then unpickles with
Python's `pickle`, as GRF does, so a state from an untrusted source runs code there, which the
restricted unpickler of the batch and of `cuju.env` prevents, as described in
[States](states.md#safety).

Dumps pickle GRF's classes too, and GRF's Python writes them straight to files, outside the engine.
A dump therefore names the module under which each class reports itself. On its own,
`cuju.gfootball` names `cuju.gfootball`, and GRF's readers in `cuju.gfootball`,
`ScriptHelpers().load_dump`, `dump_to_txt` and the `replay` player, read those dumps, while GRF
cannot import the name. With the package cuju-gfootball active, the classes report GRF's modules,
`gfootball.env.football_action_set` and `_gameplayfootball`, so the dumps of `cuju.gfootball` open
in GRF and GRF's dumps open in `cuju.gfootball`, as
[Migrating from GRF](migrating-from-grf.md#grfs-names-through-cuju-gfootball) describes. The dumps
of GRF, of `cuju.env` and of a batch name `gfootball`, as GRF's tools expect.
`cuju.dumps.read_dump`, `cuju replay` and `cuju dump` read all of them, and so does the
`replay` player of `cuju.env`.

The values of `GameEnv.config` that GRF's engine reads during a match reach cuju's engine at the
next call, as they reach GRF's. A write of the offside rule, the second half, the ball's starting
position, the engine seed or the end flags takes effect at once, and the team difficulties and the
magnet keep the values of the last reset in play while the state saves the written ones, as in
GRF, whose teams copy them at the reset. The cache of `dynamic_player_selection` and of the
controllable counts follows GRF's, cleared by every save and load of a state. Two writes, of
`reverse_team_processing` and of a team's size, raise instead, as the table of
[input validation](#input-validation) lists.

`cuju.gfootball` runs on the installed gym wherever GRF runs on it, and registers its ids with
that gym, as `import gfootball` does. GRF requires gym 0.21 and runs on every gym up to 0.24, and
on gym 0.25 with numpy 1. There the port's environments are the installed gym's `Env`, `gym.make`
gives them, and they print gym's warnings and fail GRF's own tests where GRF does. Gym 0.22 and
0.23 refuse the ids of the form `gfootball:GFootball-...` that GRF's `gym_test` and
`wrappers_test` make, and gym 0.24's step checker refuses the rewards of GRF's multi-agent
wrappers, in GRF and in `cuju.gfootball` alike.

Two gyms run `cuju.gfootball` where GRF fails. Gym 0.25 with numpy 2 checks each step with
`np.bool8`, which numpy 2 removed, so every step of GRF fails, and gym 0.26 takes gymnasium's
interface, so GRF's `reset` fails. With these gyms, and without any gym, `cuju.gfootball` serves a
copy of gym 0.21's core that cuju vendors, its environments are instances of that `Env`, so
`isinstance(env, gym.Env)` against the installed gym is false, and its ids reach
`cuju.gfootball._gym.make`, where GRF registers them with the installed gym that then fails them.

Rendering runs on cuju's software renderer. A `GameEnv` whose `game_config.render` is set at
`start_game`, and therefore `render=True` and the representations `pixels` and `pixels_gray`, draws
GRF's frames byte for byte, as [Frames and videos](rendering.md#grfs-rendering-in-cujugfootball) describes,
and raises `RendererMissingError`, naming `pip install 'cuju-engine[pixels]'`, where the renderer's
library or GRF's media is missing. `write_video=True` with `render` off needs no renderer, since
GRF's Python then draws its two-dimensional view of the match with OpenCV, and it writes GRF's
videos in `cuju.gfootball`, Kaggle's among them. The points where frames and rendering differ from
GRF's are listed in [GRF's frames](#grfs-frames). The `animations` flag of `GameEnv.reset`, which a
rendering GRF sets at every reset, already plays as in GRF. With it, the referee lets every
stoppage play out in full, 2000 milliseconds before a set piece, 500 after a goal and 10000 more
for a card, where without it the clock skips that wait. Internal tests of matches with the flag on
find cuju's engine equal to GRF's, bit for bit.

An engine that renders plays the matches of a rendering GRF. GRF's graphics observe the nodes of
every player's and official's skeleton, and each change of a node recomputes the cached position
and rotation of the node and of its root at once, while the reset of a player's position for a set
piece moves the root without that recomputation. Until the next change of the root, the joints of
the team processed first and of the officials therefore derive from the root where it stood, and a
throw-in taker of that team can place the ball from there for one tick. A save or a load of a state
marks every cache to compute afresh, so saves change a rendering match, as in GRF. The engine of
`cuju.gfootball` keeps these caches whenever its `game_config.render` is on and a renderer opened
at `start_game`, the engine of GRF's render switch included, and a batch keeps them in its
environments with render mode on. Internal tests find the matches of the clean reference build of
GRF, rendered, equal to cuju's in every state, observation, reward and done flag, on the
switch at the first reset and on the switch during an episode.

The render switch of GRF's `render()` saves the state of the engine, starts a rendering engine,
resets it and loads the state into it. GRF's loader then mirrors the ball predictions of the AI's
mental images once more than the rest of the match when `reverse_team_processing` is on, for each
image the fresh engine did not hold, so the AI reads mirrored predictions until newer images
replace them. `cuju.gfootball` loads states as GRF does, so its switched matches equal GRF's, and
the batch and `cuju.env` continue the saved match exactly, as the
[output differences](#output-differences) describe.

GRF's tracker runs as GRF's does. `tracker_setup(start, end)` plays two engines of one process in
lockstep, each waiting for the other at the end of every step in play, and compares their states
field by field, a float by its value and a radian by its angle, without the scenario part and the
fields that GRF leaves out, the camera, the keyboards' mirror, the colour of the players, the
linesmen and each gamer's keyboard. A difference raises `TrackerError` in both threads, where GRF
ends the process. An engine whose partner never steps waits for it without end, as in GRF.

## GRF's frames

cuju's software renderer draws the frames of one reference, GRF 3d9e754 on the llvmpipe driver of
Mesa 25.2.8 with LLVM 20.1.2, on an x86-64 processor with AVX2 and FMA of AMD's Zen 3 generation,
and it draws them on every machine, as [Frames and videos](rendering.md#frames-equal-to-grfs) explains.
GRF's own frames depend on the driver that runs its shaders, so a GRF that draws with a GPU's
driver, another version of Mesa or llvmpipe on another processor gives frames of its own, and cuju
equals the reference's. The table lists every known point at which the frames of cuju, or the
rendering around them, differ from GRF's.

| Situation | GRF | cuju |
| --- | --- | --- |
| the driver | draws with the driver of the machine, a GPU's or Mesa's, each with pixels of its own | draws the reference's frames on every machine, with no driver |
| llvmpipe on a processor other than AMD's Zen 3 | encodes the mipmaps of its sRGB textures with the processor's approximate reciprocal square root, whose last bits differ between AMD and Intel processors | draws the reference's mipmaps, from a table of the Zen 3 result for every input |
| `GameEnv.render()` of the engine module called while `game_config.render` is off, which GRF's Python never does | prepares the match without its graphics, so the players, the ball, the camera and the nets stay where the last render with the rendering on left them, the captions keep their text and the radar moves, while the render still drops the cached skeleton positions of render mode that it touches, which later ticks compute afresh | draws the frame as with the rendering on, and keeps the cached skeleton positions |
| `real_time` | runs the ticks of a step one at a time and renders after some of them, on a schedule of the wall clock, so the renders per step, and in render mode the match, depend on the speed of the machine | runs the same schedule with the timings of cuju's renderer, so its renders per step, and the matches with them, differ from GRF's and from run to run |
| the width of a frame | draws any `render_resolution_x`, and for a width that is not a multiple of 4 hands its Python a picture whose rows slide into each other, since OpenGL pads each row read back to whole words | draws any width that is a multiple of 4 and any height, up to 16,384 each, with frames tested at nine sizes, and raises `ArgumentError` at `start_game`, or `ConfigurationError` at the creation of a batch, for another width |
| rendering environments in a process | one, as GRF's Python asserts, whose engine the next rendering environment takes over with its resolution and history | the same in `cuju.gfootball`, which runs GRF's Python, and any number in a batch, each drawing at the batch's `render_resolution_x` and `render_resolution_y` |
| `stacked=True` with `pixels` | stacks the last four frames along the channels | stacks them in `cuju.gfootball`, while a batch raises `ConfigurationError` and the training code keeps the stack |
| the frames of the internal steps of a stoppage | in the video of `write_video` | in the video of `write_video` in `cuju.gfootball`, while a batch returns the frame before each observation and draws no pixel of the others |
| `render=True`, `write_video=True` and the `Pixels` ids in `cuju.env.create_environment` | render | raise `ValueError`, since GRF's rendering takes place in `cuju.gfootball` and in a rendering batch |

Every frame outside this table equals the reference's, byte for byte, within what testing covers,
as [Frames and videos](rendering.md#what-testing-covers) lists. A frame of cuju that differs from the
reference's is a bug, and an issue that reports it is welcome.

## Rendering, players and wrappers

GRF is both a training environment and a game with rendering, video and human input.
`cuju.gfootball` holds every module of GRF's package, and each interactive feature works as in GRF,
as the table shows.

| GRF | cuju |
| --- | --- |
| `render`, the representations `pixels` and `pixels_gray`, and `write_video` with `render` | GRF's, in `cuju.gfootball`, drawn by cuju's software renderer, and the representations `pixels` and `pixels_gray` and `frames` of a batch, as described in [Frames and videos](rendering.md) |
| the tracker, `tracker_setup` | GRF's, in `cuju.gfootball`, as described above |
| keyboard and gamepad control and the bundled `ppo2_cnn` player | GRF's modules in `cuju.gfootball.env.players`, which need pygame and TensorFlow as in GRF |
| the `replay` player, which plays a dump's actions | GRF's, in `cuju.gfootball.env.players.replay` |
| the `bot` and `lazy` players | GRF's in `cuju.gfootball.env.players`, and `extra_players` of `cuju.env` |
| the observation wrappers | GRF's classes in `cuju.gfootball.env.wrappers`, and representations and options of the batch, `stacked` and `channel_dimensions` |
| `CheckpointRewardWrapper` | GRF's class in `cuju.gfootball.env.wrappers`, and `rewards='scoring,checkpoints'` for a batch |
| `PixelsStateWrapper` | GRF's class in `cuju.gfootball.env.wrappers`, and `representation='pixels'` or `'pixels_gray'` for a batch |

cuju's renderer draws into memory, and the frames reach GRF's Python, its videos and its wrappers,
so a program that shows a match on screen while a person plays displays the frames itself, for
example with OpenCV's `imshow`. Training on GRF's vector and SMM observations runs without
rendering, the game that a batch plays by default, so a policy trained in such a batch meets in an
unrendered GRF the game that it trained on, and a policy trained on pixels meets in a rendering GRF
the frames and the matches that it trained on. Every option of GRF has a mapping in [Migrating from
GRF](migrating-from-grf.md).

## See also

- [Exactness](exactness.md)
- [Migrating from GRF](migrating-from-grf.md)
- [Observations](observations.md)
- [Episodes and seeds](episodes-and-seeds.md)
- [States](states.md)
- [Common topics](faq.md)
