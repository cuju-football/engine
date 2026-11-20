# Cuju Engine

**cuju** is a bit-exact, batched reimplementation of Google Research Football for CUDA and the
CPU. It runs thousands of matches at once on one NVIDIA GPU or across the cores of a CPU. Given the
same seed and actions, its states, observations, rewards and done flags match Google Research
Football bit for bit. Results compare directly with work on the original benchmark, and a policy
trained on cuju plays the same way there.

Install the engine from PyPI. The distribution is `cuju-engine`, and Python code imports it
with `import cuju`.

```sh
pip install cuju-engine
cuju check --device cpu
```

From 0.3.0 on, `pip install cuju` installs the engine, arena and browser viewer through the
[Cuju umbrella distribution](https://github.com/cuju-football/cuju). The
[installation guide](https://github.com/cuju-football/engine/blob/main/docs/installation.md#upgrading-from-cuju-021) explains how to upgrade
from 0.2.1 and use `cuju-repair` if an upgrade damaged the engine.

Google Research Football, or GRF, is one of the richest benchmarks of multi-agent reinforcement
learning, introduced by Kurach et al. in 2020. Its physics, animations, referee and built-in AI
support full matches from five-a-side to eleven-a-side, as well as drills that teach one skill at a
time. GRF runs one match per process at a few hundred steps per second, so on-policy training, which
consumes fresh experience by the billion, spends much of its time waiting for the simulator.

cuju suits the cases where the simulator is the bottleneck, in on-policy and multi-agent training,
self-play, large evaluations and hyperparameter sweeps, and every result that has to stand next to
published GRF numbers. Its software renderer draws the same frames as the GRF reference build, byte
for byte, for pixel observations and rendered videos on the CPU, with no GPU needed. GRF's own
tools replay the dumps cuju writes.

cuju is an independent project, neither affiliated with nor endorsed by Google. Google Research
Football is a project of Google Research, and Google is a trademark of Google LLC.

|                     |                                                                       |
| ------------------- | --------------------------------------------------------------------- |
| **Developer**       | André Koraleski                                                       |
| **Initial release** | 0.1.0, 2026-10                                                        |
| **Written in**      | C++, CUDA, Python                                                     |
| **Platforms**       | Linux on x86-64 and Arm, Windows, macOS, WSL2                         |
| **Backends**        | NVIDIA GPUs of compute capability 7.5 or higher, x86-64 CPUs with AVX2, BMI2 and FMA, 64-bit Arm CPUs |
| **License**         | Apache-2.0                                                            |
| **Reference**       | Google Research Football 2.10.3, commit 3d9e754                       |
| **Repository**      | [github.com/cuju-football/engine](https://github.com/cuju-football/engine) |

[![PyPI](https://img.shields.io/pypi/v/cuju-engine)](https://pypi.org/project/cuju-engine)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/cuju-football/engine/blob/main/LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cuju-football/engine/blob/main/examples/cuju_quickstart.ipynb)

## Quick start

The `cuju-engine` distribution installs from PyPI and brings PyTorch along when it is missing.
PyPI's default PyTorch is a CUDA build on Linux and a CPU-only build on Windows and macOS.
To use a specific build, such as another CUDA version or a CUDA build for Windows, install it
from the index at pytorch.org before installing the engine.

Run `cuju` or `cuju help` to see the engine commands and the commands of installed family
packages. Run `cuju --version` to list their installed versions. For the self-check in a script,
use `cuju check --json`, or add `--list` to get the cases without playing them. See the
[command line guide](https://github.com/cuju-football/engine/blob/main/docs/api.md#command-line) for shell completion and exit statuses. To replay
a recorded episode, pass its dump file to `cuju replay --trace_file episode_done.dump`.
The [dump guide](https://github.com/cuju-football/engine/blob/main/docs/migrating-from-grf.md#dumps-from-a-batch) explains how to write one.

```python
import cuju
import torch

batch = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=64, device='auto')
device, count = batch.torch_device, batch.environment_count
observation = batch.reset()  # [count, 4, 115]
goals = torch.zeros((), dtype=torch.int64, device=device)

for _ in range(1000):
    actions = torch.randint(0, 19, (count, 4), dtype=torch.int32, device=device)
    observation, reward, done, information = batch.step(actions)
    # reward [count, 4], done [count], and ended episodes restart inside the step
    goals += information['score_reward'].abs().sum()

print(int(goals), 'goals in the first third of', count, 'matches')
```

It prints a line such as `54 goals in the first third of 64 matches`. On a GPU, `environment_count`
rises to thousands. Every call returns the same tensors and overwrites them, so whatever is kept
must be cloned. The [documentation](https://github.com/cuju-football/engine/blob/main/docs/index.md) starts from here, and the
[quickstart notebook](https://github.com/cuju-football/engine/blob/main/examples/cuju_quickstart.ipynb) runs on Colab's free GPU.

## Features

- **One source, two backends.** The same C++ runs on the GPU, one warp per match, and on the CPU, a
    thread pool that spreads matches over the cores. They give the same bits.
- **The whole game inside the batch.** The built-in AI, the referee, the animations and the
    observations all run on the device, and nothing crosses back to Python during a step.
- **All 19 GRF scenarios**, 11-a-side and the academies included, with any number of agents on
    either side, mixed in one batch, and GRF's own scenario files without GRF.
- **PyTorch tensors without copies.** Outputs share the batch's buffers, and a CUDA graph or
    `torch.compile` can capture a step together with the policy that drives it.
- **Resets under the caller's control.** A mask resets some environments, and each can take the
    seed of its next episode.
- **GRF's states in both directions**, stacked or not, in GRF's own format.
- **GRF's interfaces too**, from its import paths to its dumps and scripted players, so GRF code
    runs unchanged.

## Performance

| Engine    | Machine         | Configuration                   | Environment steps per second |
| --------- | --------------- | ------------------------------- | ---------------------------- |
| GRF       | AMD EPYC 7763   | one process                     | 207                          |
| GRF       | AMD EPYC 7763   | 31 processes, one per processor | 5,819                        |
| cuju, CPU | AMD EPYC 7763   | one environment on one thread   | 2,018                        |
| cuju, CPU | AMD EPYC 7763   | 31 threads                      | 38,497                       |
| cuju, GPU | NVIDIA RTX 4090 | 16,384 environments             | 32,582                       |

All numbers are for `5_vs_5` with four agents on each side and `simple115v2`, medians of interleaved
runs of the release wheel on one cloud machine, which gives the process 31 processors. GRF runs on
the CPU, so the GPU row compares with the CPU rows of the same machine. It is the throughput of a
batch that keeps its observations, actions and policy on the device, which leaves the processors
free for other work.
[Performance](https://github.com/cuju-football/engine/blob/main/docs/performance.md) has the method, more scenarios and batch sizes, and the commands
that reproduce every row.

## Exactness

The reference is GRF 2.10.3 at commit 3d9e754, without rendering, as built by GCC for x86-64 Linux
with glibc 2.39 on a processor with FMA. Both backends give the same bits, whatever the number of
threads. When cuju renders, it plays the matches of a rendering GRF, and its frames equal those that
GRF draws on Mesa 25.2.8's llvmpipe, on every machine.

cuju is tested against GRF internally. Anyone can run its self-check. `cuju check`, in the
wheel, and `pytest tests/`, in the repository, need no GRF and tell whether a build reproduces the
release bit for bit, and `examples/compare_with_grf.py` plays cuju next to a separately installed
GRF. Issues are welcome, and a difference from GRF is the most valued report.
[Exactness](https://github.com/cuju-football/engine/blob/main/docs/exactness.md) explains the claim, its scope and how to check it.

## Coming from GRF

There are three ways in. `cuju.gfootball` is GRF's own Python package over cuju's engine, so
switching the imports to it runs GRF code unchanged, internals included. `cuju.env.create_environment`
is cuju's faster version of GRF's function, with a choice of device, and `cuju.BatchEnvironment` is
the batch, where the speed is. A port usually switches the imports first, confirms what it already
knows, and then moves only the training loop to a batch. The table maps the batch.

| GRF                                                                                   | cuju                                                                       |
| ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `gfootball.env.create_environment('5_vs_5', number_of_left_players_agent_controls=4)` | `cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=...)`             |
| `env.step(action)` with `[4]` actions                                                 | `batch.step(actions)` with `[environment_count, 4]` actions                |
| `if done: env.reset()`                                                                | the step that ends an episode starts the next by itself                    |
| the last observation of an episode                                                    | `information['final_observation'][done.bool()]`, per array for `raw`       |
| `env.get_state()`, `env.set_state(state)`                                             | `batch.get_state(index)`, `batch.set_state(index, state)`, in GRF's format |
| `representation`, `rewards`, `stacked`, `channel_dimensions`, `action_set`            | the same names                                                             |
| code written for GRF                                                                  | the same code with `gfootball` replaced by `cuju.gfootball`                |

`done` is `uint8`, and PyTorch deprecates indexing with a `uint8` mask, so indexing goes through
`done.bool()`. With `representation='raw'` the observations are dictionaries of tensors, each
indexed on its own. [Migrating from GRF](https://github.com/cuju-football/engine/blob/main/docs/migrating-from-grf.md) discusses the three ways in and
maps every option, method and wrapper.

## Platforms

<!-- Keep this section and the Platforms row of the table at the top in step. -->

PyPI holds a wheel for each platform, and every platform passes the same self-check against the
same digests, so each plays GRF's matches bit for bit.

| Platform                       | Backends                  |
| ------------------------------ | ------------------------- |
| Linux on x86-64, WSL2 included | CPU and CUDA              |
| Linux on Arm, aarch64          | CPU                       |
| Windows on x86-64              | CPU and CUDA              |
| macOS on Apple silicon         | CPU                       |
| macOS on Intel processors      | CPU, with PyTorch 2.2     |

The CPU backend needs AVX2, BMI2 and FMA on x86-64 and runs on every 64-bit Arm processor. The CUDA
backend needs an NVIDIA GPU of compute capability 7.5 or higher, and the wheels carry its machine
code for the architectures checked for exactness, 7.5, 8.0, 8.6 and 8.9, and other architectures build from
source. On macOS on Intel processors, PyTorch ends with 2.2, and `torch.compile` traces a step
from PyTorch 2.4 on. [Installation](https://github.com/cuju-football/engine/blob/main/docs/installation.md) gives the requirements of each platform.

GRF's frames, the representations `pixels` and `pixels_gray` and GRF's rendered videos come from
cuju's software renderer once `pip install 'cuju-engine[pixels]'` has added GRF's media, in
`cuju.gfootball` and in batches. GRF's tools replay cuju's dumps and can make videos from them.
[Differences from GRF](https://github.com/cuju-football/engine/blob/main/docs/differences-from-grf.md) describes every known difference.

## Documentation

| Page                                                                                         | Covers                                                                 |
| -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| [Concepts](https://github.com/cuju-football/engine/blob/main/docs/concepts.md)                                                                 | environments, batches, bundles, agents, and why the API looks this way |
| [Installation](https://github.com/cuju-football/engine/blob/main/docs/installation.md)                                                         | requirements, PyPI, source builds, Docker, WSL2, troubleshooting       |
| [Migrating from GRF](https://github.com/cuju-football/engine/blob/main/docs/migrating-from-grf.md)                                             | every GRF name and its counterpart                                     |
| [Scenarios](https://github.com/cuju-football/engine/blob/main/docs/scenarios.md) and [Custom scenarios](https://github.com/cuju-football/engine/blob/main/docs/custom-scenarios.md)              | the scenarios that play, and custom scenario files                     |
| [Observations](https://github.com/cuju-football/engine/blob/main/docs/observations.md), [Rewards and actions](https://github.com/cuju-football/engine/blob/main/docs/rewards-and-actions.md)     | what agents see, receive and do                                        |
| [Episodes and seeds](https://github.com/cuju-football/engine/blob/main/docs/episodes-and-seeds.md), [States](https://github.com/cuju-football/engine/blob/main/docs/states.md)                   | resets, reproducibility, saving and moving matches                     |
| [Backends](https://github.com/cuju-football/engine/blob/main/docs/backends.md), [Performance](https://github.com/cuju-football/engine/blob/main/docs/performance.md), [Recipes](https://github.com/cuju-football/engine/blob/main/docs/recipes.md) | getting the most out of each device                                    |
| [Frames and videos](https://github.com/cuju-football/engine/blob/main/docs/rendering.md)                                                       | GRF's frames, pixel observations and videos from the software renderer |
| [Exactness](https://github.com/cuju-football/engine/blob/main/docs/exactness.md), [Differences from GRF](https://github.com/cuju-football/engine/blob/main/docs/differences-from-grf.md)         | what equals GRF, how to check it, and where cuju differs               |
| [Common topics](https://github.com/cuju-football/engine/blob/main/docs/faq.md), [API reference](https://github.com/cuju-football/engine/blob/main/docs/api.md)                                   | quick answers and every public name                                    |

## The Cuju family

For submissions, formats, ratings and tournaments, see [cuju-arena](https://github.com/cuju-football/arena).

The browser viewer and app are available from [cuju-render](https://github.com/cuju-football/render).

The [umbrella distribution](https://pypi.org/project/cuju/) installs the whole family.

## Name

Cuju, written 蹴鞠 and pronounced *cùjū*, literally kick ball, is the football of ancient China, and
the oldest form of the game that FIFA recognises. Records place it in the city of Linzi in the
Warring States period, more than two thousand years ago. Under the Han dynasty it served as military
training, and a work in twenty-five chapters on its rules appears among the books of the imperial
library. It reached its height under the Song, when clubs of players kept its craft, and it later
travelled to Japan, where it became kemari. The project takes the name because cuju is the earliest
football whose rules people wrote down, and the project is, at heart, an exercise in keeping a
game's rules exactly. The name is a tribute, and cuju, the project, is unconnected with any
institution that preserves the game. The name also shares its first syllable with CUDA.

## Citing

Research that uses cuju is asked to cite both cuju and GRF.

```bibtex
@software{cuju2026,
    author  = {Koraleski, Andr{\'e}},
    title   = {cuju: a bit-exact, batched reimplementation of {Google Research Football}, for {CUDA}
             and {CPU}},
    year    = {2026},
    version = {0.3.0},
    url     = {https://github.com/cuju-football/engine}
}

@inproceedings{kurach2020google,
    title     = {{Google Research Football}: A Novel Reinforcement Learning Environment},
    author    = {Kurach, Karol and Raichuk, Anton and Sta{\'n}czyk, Piotr and Zaj{\k{a}}c, Micha{\l}
               and Bachem, Olivier and Espeholt, Lasse and Riquelme, Carlos and Vincent, Damien
               and Michalski, Marcin and Bousquet, Olivier and Gelly, Sylvain},
    booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
    volume    = {34},
    number    = {04},
    pages     = {4501--4510},
    year      = {2020},
    doi       = {10.1609/aaai.v34i04.5878}
}
```

[CITATION.cff](https://github.com/cuju-football/engine/blob/main/CITATION.cff) holds the same entry for GitHub's citation button.

## Acknowledgments

cuju exists because of the authors of Google Research Football, who built and opened a remarkable
environment, because of Gameplay Football by Bastiaan Konings Schuiling, from which GRF's engine
descends, and because of the GNU C Library, whose math functions it ports.

## License

cuju is licensed under Apache-2.0. It reimplements GRF, licensed under Apache-2.0 by Google LLC,
whose engine descends from Gameplay Football by Bastiaan Konings Schuiling, and [NOTICE](https://github.com/cuju-football/engine/blob/main/NOTICE)
lists every credit. Four math functions in `include/cuju/libm/`, namely sin, cos, acos and asinf,
are ports of the GNU C Library under the LGPL 2.1 or later, and both libraries compile them in.

- Using cuju, from open or proprietary code, carries no obligation. Obligations start with
    distribution.
- Redistributing cuju's libraries, as a wheel or an image, means keeping LICENSE, NOTICE and
    LICENSES/ with them and offering the matching source, which every PyPI release carries as its
    sdist.
- Compiling the C++ headers into another program brings the LGPL into that program.
- The logo in `docs/assets/` is licensed under CC BY 4.0, so it may be reused with credit, without
    any right to the name or the logo as a mark of endorsement.
