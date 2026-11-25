# Exactness

cuju is exact when every output equals, bit for bit, the output of Google Research Football, or GRF, for the same seed and actions. This page states the claim, lists what a user can check, and lists the known differences from GRF.

The testing against GRF is internal. Issues are welcome, and a difference from GRF is the most valued report. See [Reporting a difference](#reporting-a-difference).

## The claim

> For the same seed and actions, every state, observation, reward and done flag of cuju equals, bit for bit, what GRF 2.10.3, commit 3d9e754, produces without rendering, and in render mode what it produces with rendering, as built by GCC for x86-64 Linux with glibc 2.39 on a processor with FMA. The two backends give the same bits, whatever the number of threads.

Every state means the whole state of the match as GRF serializes it, with every player, the ball, the officials, the animations in progress, the memory of the built-in AI and the clocks. Equal states at every step are a stricter test than equal observations.

The claim is empirical. It is stated narrowly, with the version, the build and the outputs named, so that a single counterexample refutes it.

## Reference build

GRF calls the math library for sines, cosines, arc tangents and powers. The last bits of those functions differ between C libraries, and one different bit in one `atan2f` can change which animation a player picks and so the whole match. The claim therefore names one build, the toolchain of Ubuntu 24.04.

cuju carries ports of the math functions of glibc 2.39, compiled into both backends in place of the system library. Its results are the same on every machine, on Linux, Windows and macOS, on x86-64 and Arm, with GCC, Clang, MSVC and Apple Clang. A GRF built against another glibc can play other matches, and then cuju differs from that build while it equals the reference.

`examples/compare_with_grf.py` shows which case a GRF installation falls in.

## What is covered

| Part | Covered |
| --- | --- |
| States, observations, rewards and done flags | every one, at every step |
| Backends | CPU and CUDA, any number of threads, any batch size, any neighbors in the batch |
| GPU architectures of the wheel | compute capabilities 7.5, 8.0, 8.6 and 8.9, from one source |
| Rendering GRF | cuju in render mode plays the matches of a rendering GRF |
| Frames | byte for byte, against one reference renderer, see [Frames](#frames) |
| Options that extend GRF, such as `difficulty` | equal to the scenario file GRF would build with the same values |

A rendering GRF plays other matches than one that does not. Its referee acts out each stoppage and its graphics keep cached skeleton positions that a set piece leaves behind for a tick. cuju plays the unrendered matches by default and the rendered ones in `cuju.gfootball` and in the environments of a batch that render.

Options that only read, such as `raw` and `frames`, leave every result unchanged. A batch plays each episode as one GRF run, so `reset` with chosen seeds gives GRF's episodes.

The kernels of this release were proven bit-exact on a GeForce RTX 3050 Laptop GPU, compute capability 8.6. The other architectures come from the same source with the same flags, and `cuju check` verifies any GPU in about a minute.

The checkpoint reward of `rewards='scoring,checkpoints'` takes the precision of the numpy beneath GRF, since numpy 2 changed how a Python float combines with a `float32`. Each precision equals its GRF bit for bit. See [Observations, rewards and actions](observations-rewards-actions.md).

## Frames

The frames of the software renderer have a claim of their own.

> For the same match, every frame of cuju's software renderer, and every `pixels` and `pixels_gray` observation made from it, equals byte for byte what GRF 3d9e754 draws on the llvmpipe driver of Mesa 25.2.8 with LLVM 20.1.2, on an x86-64 processor of AMD's Zen 3 generation with AVX2 and FMA, whatever the number of threads and workers and on both backends.

GRF's frames depend on the driver that draws them, so the claim names the software rasterizer GRF uses on a machine without a GPU driver. cuju draws that reference on every machine, with or without a GPU. Frames are tested internally at nine sizes from 1920 by 1080 down to 96 by 54, and through rare moments such as red cards and every caption. A frame that differs from the reference, on any machine or at any size, is a welcome report. See [Observations, rewards and actions](observations-rewards-actions.md) for rendering.

## Why the results repeat

Floating-point arithmetic is deterministic when every operation is the same operation in the same order. The code keeps to that rule in four ways.

- Expressions follow GRF's source, including its evaluation order and its choice of `float` or `double`.
- Multiplications and additions fuse into one FMA instruction only where GRF's code does, with `-ffp-contract=off` on the host and `--fmad=false` for nvcc. The build stops at flags such as `-ffast-math`, and on the GPU it selects IEEE division and square roots.
- The math functions are those of glibc 2.39, ported for host and device.
- Where GRF depends on x86 behavior, such as converting an out of range `float` to `int` or the sign of zero, the code reproduces it on both devices, Arm included.

Each environment keeps its own world and random generator, so a match plays the same bits whatever else its batch holds.

## What a user can check

A wheel on a GPU it covers needs only the self-check. A source build, other compiler flags or a GPU outside the wheel need the long cases and the tests as well.

### Self-check

`cuju check` ships in the wheel and needs only PyTorch. It prints the versions, the processor, each library with its SHA-256 and every GPU with the code it runs. It checks the shipped bundles and animations against `data/SHA256SUMS`, plays the digest cases on the CPU and on every visible GPU, and prints `PASSED` or `FAILED`. The exit status is 0 when everything that ran matched and 1 otherwise.

```sh
cuju check                # every short case, the CPU and every GPU
cuju check --device cuda  # one backend, which must then be present
cuju check --long         # also whole matches, several minutes on the CPU
cuju check --list         # the diagnostics and the cases, without playing
```

A failing digest means that the compiler, the flags or the GPU changed the results, and that build differs from the release. A GPU outside the wheel passes `--long` before use.

### Public tests

After a source build, `pytest tests/` runs the same digests and more. A digest is a SHA-256 over every output of a short run, so one changed bit changes it. The suite also checks the shapes, the independence from the number of threads, the agreement of the two backends and a state round trip. Skipped tests, such as the CUDA cases on a machine without a GPU, count as unchecked.

The frames have their own tests, which need the extra `pixels`.

```sh
pytest tests/render/test_render_frames.py
CUJU_LONG=1 pytest tests/render/test_render_frames.py  # whole episodes, several minutes
```

### Comparison with GRF

`examples/compare_with_grf.py` plays cuju next to a GRF installation, by seed and from a state that GRF saves midway. It compares bit for bit the observation of every agent in the representation that `--representation` selects, the rewards, the done flags and the match part of the state at the interval of `--state-interval`. `--agent submission.py:agent` plays a given policy, or a Kaggle submission, in both engines. The script prints the first difference and exits with 1, or exits with 0 when everything is equal.

```sh
python examples/compare_with_grf.py --scenario 5_vs_5 --left 4 --right 0 --steps 3000
```

For a trained policy, compare its win rate and goal difference against the built-in AI over the same seeds given to `reset`. When the engines agree, these agree as well.

### Recorded episodes

`cuju.dumps.DumpWriter` writes the episodes of chosen environments as GRF dumps. `cuju replay FILE` plays a dump again and compares every step with it bit for bit. It prints the first step that differs and exits with 1, or exits with 0 when the replay equals the dump. A dump made on the GPU replays on the CPU, and the command runs no code from the dump.

```python
import cuju
import torch
from cuju import dumps

batch = cuju.BatchEnvironment('academy_empty_goal_close', environment_count=8, device='cuda')
writer = dumps.DumpWriter(batch, 'dumps', environments=[0])
batch.reset()
actions = torch.randint(0, 19, (8, 1), dtype=torch.int32, device='cuda')
for _ in range(400):
    batch.step(actions)
    writer.record(actions)
print(writer.paths[0])
```

See [Migrating from GRF](migrating-from-grf.md) for dumps in GRF's own tools.

## Known differences from GRF

Outside this table, cuju equals GRF within the claim above. Each row says what a user does about it.

### Outputs

| Difference | Reach | What to do |
| --- | --- | --- |
| The time a built-in AI player needs to reach the ball is an unsigned 32-bit count of milliseconds that cuju converts to `float` directly and GRF converts through `int`. | Starts at 2^31 milliseconds. The engine draws the time from predictions a few seconds ahead, so no match reaches it. | Nothing. |
| A NaN input to the two `Normalize` helpers of the vector class gives the GPU's NaN bit pattern on the GPU. | The engine passes no NaN there, so no output changes. | Nothing. |
| After `set_state`, GRF keeps the frame of each mental image of the AI outside the state. When its environment holds fewer images than the state, it adds the missing ones with mirrored predictions, so its AI reads wrong predictions for about 100 milliseconds and the match drifts. cuju sets the frames from the state and continues the saved match exactly. | A state with `reverse_team_processing` on, a non-deterministic scenario with an odd engine seed, loaded into a freshly reset GRF environment. `cuju.gfootball` loads as GRF does. | Compare from a state by loading GRF's state into cuju, as `examples/compare_with_grf.py` does. |
| The checkpoint reward depends on the numpy of the GRF under comparison. | Only `rewards='scoring,checkpoints'`. | Install the numpy of the GRF you compare with, or use `cuju.env`, which takes the precision from the installed numpy. |

### Variation within GRF

| Behavior of GRF | cuju | What to do |
| --- | --- | --- |
| State bytes vary from run to run, since GRF writes the padding between fields as it finds it in memory. | Writes zeros, so the states of one match are identical byte for byte. | Compare states by their fields or by loading them. |
| The `bot` player varies with the processor, since numpy's BLAS may fuse operations on AVX-512. | Computes as numpy does without fusing, as on an AVX2 processor. | Nothing. Decisions are the same on every machine. |
| A GRF that renders plays other matches than one that does not. | Plays both, depending on the render mode. | Compare a rendering GRF with a batch or `cuju.gfootball` that renders. |

### Limits and validation

| Situation | GRF | cuju | What to do |
| --- | --- | --- | --- |
| A match longer than the history of positions | The list of positions grows without bound. | A batch keeps 512 seconds, which holds every shipped scenario and any match of up to 51,199 ticks, or 5,119 steps at the default of 10 ticks a step. A longer match raises an error at creation. | Shorten `game_duration` or lower `physics_steps_per_frame`. |
| A path of the engine that GRF leaves undefined, such as a set piece without a taker | No result to equal. | The CPU raises `cuju.errors.UndefinedPathError` naming the environment and GRF's reason, and the GPU traps. | Report it. Shipped scenarios avoid these paths as far as is known. |
| A scenario with a team of more than 11 players | Crashes. | `ScenarioError` | Use at most 11 players. |
| A scenario whose first player of a team is not a goalkeeper | Hangs in open play. | `ScenarioError` | Make the first player a goalkeeper. |
| A scenario that draws from `random` or NumPy's global generator | Runs and shifts the seeds of every later episode. | `ScenarioError` | Draw from the scenario's own generator. |
| A scenario that reads the controllable players or `dynamic_player_selection` while it builds | Freezes the first value read. | `ScenarioError` | Set the agents through the constructor. |
| A scenario that depends on more than the parity of the episode number | Runs. | `ScenarioError` | Make it depend on the parity only. |
| A state whose scenario differs from the environment's outside the constant part | Stops only when the constant part differs. | `StateError` for any difference but the seed, naming up to three fields. | Load the state into an environment of the same scenario. |
| A state without checkpoint counts under `rewards='scoring,checkpoints'`, or a stacked state in an unstacked environment | `KeyError` or silent ignoring | `StateError` | Load it into an environment with the same options. |

See [Scenarios](scenarios.md) for the rules of custom scenarios.

### Interface

The interface of a batch differs from GRF's by design. It returns PyTorch tensors, always keeps the agent axis, defaults to `simple115v2`, resets inside the step, and writes dumps by environment. [Migrating from GRF](migrating-from-grf.md) lists each point with its GRF equivalent, and `cuju.env.create_environment` gives GRF's interface for one environment.

### Frames and rendering

| Situation | GRF | cuju | What to do |
| --- | --- | --- | --- |
| The driver | Draws with the driver of the machine, each with pixels of its own. | Draws the reference's frames on every machine. | Compare with a GRF on llvmpipe of Mesa 25.2.8. |
| `real_time` | Renders on a schedule of the wall clock, so the match depends on the speed of the machine. | Runs the same schedule with the timings of cuju's renderer, so matches differ from GRF's and from run to run. | Leave `real_time` off when comparing. |
| The width of a frame | Any width, but a width that is not a multiple of 4 gives rows that slide into each other. | Any width that is a multiple of 4 and any height up to 16,384, and an error for another width. | Use a width that is a multiple of 4. |
| `stacked=True` with `pixels` in a batch | Stacks the last four frames. | `ConfigurationError`, and `cuju.gfootball` stacks as GRF does. | Stack in the training code. |
| `render=True` and `write_video=True` in `cuju.env.create_environment` | Render. | `ValueError` | Render in `cuju.gfootball` or in a rendering batch. |
| Frames of the internal steps of a stoppage | In the video of `write_video`. | In `cuju.gfootball`, and a batch returns the frame before each observation. | Use `cuju.gfootball` for videos of every tick. |

## Reporting a difference

A difference usually appears thousands of steps into a run, and its cause is found most directly from one step before it. A report is most useful with these items.

- The whole output of `cuju check`, which names the versions, the device, the libraries and their SHA-256, and whether the build reproduces the digests.
- The wheel, or the source build with its compiler and flags.
- The call that created the batch, with every argument.
- The batch `seed`, the index of the environment, and any `seeds` passed to `reset`.
- The state from `get_state` at the last step where the two agree.
- The actions from there on, the step where the two first differ, and what differs.
- The GRF build under comparison, with its version, operating system, glibc and GCC.

A frame report adds the scenario, the seed and the step, the resolution, `render_threads` and `render_workers` of the machine config, the processor, and the frame or the output of `pytest tests/render/test_render_frames.py`. [API reference](api.md) describes the machine config.

A difference that reproduces is a bug, and its fix brings cuju to GRF's result. Open an issue in the repository of cuju. See [Getting started](getting-started.md) to install and run the first check, and [Performance](performance.md) for the speed of each backend.
