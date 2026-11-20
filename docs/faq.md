# Common topics

The **common topics** of cuju are the behaviours of a batch and its surroundings that most often
prompt questions. They concern the outputs of a batch, which are reused buffers, padded in mixed
batches and reset inside the step, the agreement of cuju with Google Research Football, or GRF, and
the transfer of policies to GRF and Kaggle, the errors that hardware and installation raise, the
scale of a run over threads, processes and GPUs, and the interoperation of cuju with other
libraries.

Most surprises trace back to three properties. Each call overwrites the tensors that the previous
call returned, the step that ends an episode already starts the next one, and a bare scenario name
gives one agent, as in GRF.

## Outputs of a batch

### Overwritten observations

Every call overwrites the tensors it returned before. `reset` and `step` return views of the
buffers of the batch, so a list of returned observations holds the same tensor many times, all
showing the last step. Storing `observation.clone()`, or copying into a buffer allocated once as in
[Recipes](recipes.md), keeps each step.

### Agents of a bare scenario name

`BatchEnvironment('5_vs_5')` gives one agent, as `create_environment('5_vs_5')` of GRF does, with
that agent on the left and the built-in AI controlling every other player. Names such as
`5_vs_5_l4_r0` or `5_vs_5_l4_r4`, or the `agent_counts` option, set the agents explicitly.

### Truncation

cuju follows GRF in reporting the end of the match time through `done`, the same flag that reports
every other end of an episode. The time left is part of the state, `steps_left` in `raw`, and an
end at the time limit is one at which it reached zero, so training code that treats truncation
apart from termination reads `steps_left` from a batch with `with_raw=True`, as described in
[Episodes and seeds](episodes-and-seeds.md). The Gymnasium interface of `cuju.env` follows GRF as
well and reports every end through `terminated`, with `truncated` always false.

### Pitch coordinates

Raw `x` exceeds 1 and raw `y` exceeds 0.42 because the pitch is slightly larger than the
documentation of GRF states. Its goal lines lie at raw `x = ±1.0110` and its touchlines at
`y = ±0.4306`, the quotients 55 m / 54.4 and 36 m / 83.6. [Observations](observations.md) gives the
units, and the view of a right agent key by key.

### Padded agents and players

In a batch that mixes bundles, padded slots hold zeros. The observation and reward of a padded agent
are zero, and the batch ignores its actions. A padded player of a raw team array is all zeros, with
`*_active` false and the goalkeeper role `0`. `batch.agent_mask` and the one-line player mask in
[Observations](observations.md) select the real entries.

### Observation after the end of an episode

The step that ends an episode starts the next one and returns its first observation. The last
observation of the ended episode is in `information['final_observation']`, where `done == 1`.

## Agreement with GRF

### Differences from a local GRF

A difference from a local GRF may indicate a bug in cuju, and reports of such differences are
welcome as issues. Agreement requires that both run the same scenario, options, seed and actions, on
a GRF built as the reference is, with GCC for x86-64 Linux with glibc 2.39. A GRF built on another
platform may play other matches, and a GRF that renders plays the matches of render mode, which
cuju plays whenever it renders, in `cuju.gfootball` and in batches with pixels or frames. A
difference that persists under these conditions is reported in an issue with the details that
[Exactness](exactness.md) lists.

### Frames that differ from a local GRF's

GRF's frames depend on the driver that draws them, a GPU's driver or a version of Mesa, so a local
GRF can draw pixels that differ from cuju's while the match is the same. cuju draws the frames of
one reference, GRF on Mesa 25.2.8's llvmpipe with LLVM 20.1.2 on an AMD Zen 3 processor, on every
machine, as [Frames and videos](rendering.md#frames-equal-to-grfs) explains. A frame of cuju that differs
from that reference's is a bug and is welcome as an issue, while a difference from GRF on another
driver is the driver's own picture.

### Transfer of policies to GRF

A policy trained with cuju transfers to GRF unchanged. The matches are those of GRF, the
observations equal those of GRF for the same state, and the actions have the same meaning, so the
policy plays the same way in GRF, on a GRF built as the reference is. Exactness serves this purpose,
since results then compare directly with work done on GRF.

A GRF built another way is the same game, with the same rules, physics, built-in AI, observations
and actions. Its individual matches may still diverge after a difference in the last bit of a math
function, so comparisons with such a build are made over many episodes rather than match by match.

### Kaggle submissions

The football competition on Kaggle ran its own build of GRF, so a policy trained with cuju plays the
same game there, with different individual matches. How closely the GRF version of the competition
matches 2.10.3 is an open question as of version 0.1.0. Two further facts concern submissions.

A Kaggle agent receives the `raw` observation as lists, in `obs['players_raw'][0]`, and
`cuju.kaggle.players_raw` builds the same lists from a batch. The wrappers of GRF, applied to them
through `Simple115StateWrapper.convert_observation` and `observation_preprocessing.generate_smm`,
give exactly the `simple115v2` and `extracted` observations of cuju.

The raw arrays of cuju are `float32`, while the lists that Kaggle sends become `float64` arrays in
numpy, so the same feature code gives different last bits in training and in the submission.
Casting with `.double()` in training starts from the values that the submission sees.

In `11_vs_11_kaggle` the second half starts after step 1,500 with a kick-off by the other team. The
teams keep their ends, so every observation keeps its orientation through half time.

## Installation and hardware

### Diagnostics

`cuju check` reports the versions, processor, libraries and GPUs of an installation and
checks that the build plays the matches of GRF. Its output belongs in every issue report.

### Illegal instruction and AVX2 errors

On x86-64 the CPU library requires AVX2, BMI2 and FMA, which the Intel Core processors since
Haswell, released in 2013, and the AMD processors since Excavator, released in 2015, provide. On a
processor that lacks them, creating a CPU batch with the packaged library raises `DeviceError`
naming the missing extensions. The check reads all three on Linux and macOS, and AVX2 alone on
Windows, whose system reports that one. Some virtual machines hide these extensions, and a CPU
model that passes them through resolves the error, as described in [Installation](installation.md).
On Arm, Apple silicon included, the library uses the base Armv8-A and runs on every 64-bit
processor.

### Platforms

Release 0.1.0 has wheels for Linux on x86-64 and on Arm, Windows, macOS on Apple silicon and on
Intel processors, and WSL2, which takes the wheel for Linux. The CPU backend runs on all of them,
and the CUDA backend on Linux on x86-64, Windows and WSL2. Every platform passes the same self-check
with the same digests, so each gives the same bits, as listed in
[Installation](installation.md#platforms).

### Missing renderer or media

`RendererMissingError` in `cuju.gfootball`, or `MissingFileError` from a batch with pixels, names
`pip install 'cuju-engine[pixels]'`, which installs GRF's media, the package cuju-grf-media. The renderer's
library itself is part of cuju's wheel, and a source build makes it with the CMake option
`CUJU_RENDER`, on by default, as [Installation](installation.md) describes.

### Missing kernel image

The wheels for Linux on x86-64 and for Windows carry machine code for compute capabilities 7.5,
8.0, 8.6 and 8.9, all compiled from the same source. The kernels of this release were proven
bit-exact on the RTX 3050 Laptop GPU, compute capability 8.6, and `cuju check` verifies any GPU in
about a minute. Other GPUs run cuju from a source build, with `CMAKE_CUDA_ARCHITECTURES` set to the
compute capability of the GPU, after which `pytest tests/` checks the build.

### Slow first call

On the GPU, CUDA loads and prepares the kernel at the first launch, and timings start from the
second step.

## Scale

### Environments and threads

On the GPU, throughput rises until the device is full, so a batch holds as many environments as fit
in memory, typically thousands to tens of thousands. On the CPU, about 16 environments per thread
keep every core busy, and `threads=0` uses every processor available to the process.
[Performance](performance.md) gives the measured throughput.

### Multiprocessing and Ray

cuju works under multiprocessing and Ray with each batch created inside the worker that uses it. A
batch holds device memory and threads that stay with their process, so pickling one raises
`TypeError`, and a CUDA batch stays in the process that created it, as its CUDA context does. A CPU
batch created before a fork works in the child, and creating it after the fork saves copying its
buffers. One batch per worker process and GPU is the arrangement with the fewest interactions.

### Slowdown with a second GPU process

On a small GPU under Windows or WSL2, the driver moves memory beyond the capacity of the device into
shared system memory instead of failing the allocation, and every kernel that touches that memory
slows down, by a factor of about ten on a 4 GB laptop GPU. One batch per GPU, or evaluation on the
CPU beside training on the GPU, avoids the spill, as described in [Performance](performance.md).

### Comparison with many GRF processes

Running many GRF processes is the usual way of scaling GRF. Each process runs one match at a few
hundred steps per second, holds its own memory and Python interpreter, and sends its observations
through inter-process communication. A batch runs thousands of matches in one process, on one GPU
or on every core, with the outputs already in tensors. [Performance](performance.md) compares the
two on the same machine.

### Several GPUs

Several GPUs are used with one batch per GPU, usually one per process, with
`device=f'cuda:{local_rank}'` and `first_index=local_rank * environment_count`, so that the ranks
together play the environments of one large batch. `batch.torch_device` is the device of its
tensors. Within one process, `cuju.MultiDeviceBatch` steps one batch per device together.
[Backends](backends.md) describes both layouts. cuju 0.1.0 was checked on machines with a single
GPU, and reports from machines with several GPUs are welcome as issues.

## Interoperability

### JAX and other frameworks

The package returns PyTorch tensors, which DLPack hands to JAX without a copy through
`jax.dlpack.from_dlpack`. The C API underneath runs without Python, as described in the
[API reference](api.md).

### Gymnasium

`cuju.env.create_environment` reproduces the `create_environment` of GRF, with the gym 0.21
interface of GRF by default and the Gymnasium interface with `api='gymnasium'`. The ids
`GFootball-*-v0` work with `gymnasium.make`. [Migrating from GRF](migrating-from-grf.md) shows both.

### Play against the built-in AI

A side without agents, as in `5_vs_5_l4_r0`, is the scenario of GRF against the built-in AI and the
standard opponent for evaluation. Action 19, `builtin_ai`, of the `v2` and `full` sets instead hands
the player of one agent to the built-in AI for a step, which produces other matches.
[Recipes](recipes.md) compares the two.

### Rendering and video

cuju draws GRF's frames with a software renderer of its own, which needs no GPU and no display, in
`cuju.gfootball`, where `render=True`, `write_video=True`, `pixels` and `pixels_gray` run as in
GRF, and in a batch with `representation='pixels'`, `'pixels_gray'` or `render_frames=True`, after
`pip install 'cuju-engine[pixels]'`. The frames equal GRF's, byte for byte, and a frame takes a few hundred
milliseconds of processor time, as [Frames and videos](rendering.md) describes.

GRF's videos of a match come from `write_video=True` in `cuju.gfootball`, with every frame that
GRF's video holds, or from the frames of a batch with `render_frames=True`, written with any video
library. `cuju.dumps.DumpWriter` writes a match as a GRF dump, and the `dump_to_video` of GRF, in
GRF or in `cuju.gfootball`, turns the dump into a video, which plays the dump again in render mode
and so plays another match than the unrendered one that the dump holds.

## Project

### Affiliation

cuju is an independent reimplementation of Google Research Football, a project of Google Research.
It is written and maintained by the cuju authors, apart from Google.

### Name

Cuju is the ancient Chinese ball game that FIFA recognises as the earliest known form of football.
The README of the project describes its history.

## See also

- [Concepts](concepts.md), for batches, bundles and environments
- [Differences from GRF](differences-from-grf.md), for the interface, input checks and engine bounds in which cuju differs from GRF
- [Rewards and actions](rewards-and-actions.md), for the action sets and `builtin_ai`
