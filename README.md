# Cuju Engine

**cuju** is a bit-exact, batched reimplementation of Google Research Football (GRF) for CUDA and the CPU. It runs thousands of matches at once on one NVIDIA GPU or across the cores of a CPU. With the same seed and actions, its states, observations, rewards and done flags equal GRF's bit for bit, so results compare directly with work on the original benchmark and a policy trained on cuju plays the same way there.

cuju suits on-policy and multi-agent training, self-play, large evaluations and sweeps, where the simulator is the bottleneck. Its software renderer draws GRF's frames byte for byte on the CPU, and GRF's own tools replay the dumps cuju writes. cuju is an independent project, neither affiliated with nor endorsed by Google.

|                |                                                                         |
| -------------- | ----------------------------------------------------------------------- |
| **Developer**  | André Koraleski                                                         |
| **Written in** | C++, CUDA, Python                                                       |
| **Platforms**  | Linux on x86-64 and Arm, Windows, macOS, WSL2                           |
| **Backends**   | NVIDIA GPUs of compute capability 7.5 or higher, CPUs with AVX2, BMI2 and FMA, 64-bit Arm CPUs |
| **License**    | Apache-2.0                                                              |
| **Reference**  | Google Research Football 2.10.3, commit 3d9e754                         |

[![PyPI](https://img.shields.io/pypi/v/cuju-engine)](https://pypi.org/project/cuju-engine)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cuju-football/engine/blob/main/examples/cuju_quickstart.ipynb)

## Install

```sh
pip install cuju-engine
cuju check --device cpu
```

The distribution is `cuju-engine`, and code imports `cuju`. PyTorch comes along when missing. For a specific PyTorch build, such as another CUDA version, install it from pytorch.org first. [Getting started](https://github.com/cuju-football/engine/blob/main/docs/getting-started.md) covers each platform and GPU.

## A first loop

```python
import cuju
import torch

batch = cuju.BatchEnvironment('5_vs_5', environment_count=64, left_agents=4, device='auto')
observation = batch.reset()  # [64, 4, 115]
goals = torch.zeros((), dtype=torch.int64, device=batch.device)

for _ in range(1000):
    actions = torch.randint(0, batch.action_count, (64, 4), dtype=torch.int32, device=batch.device)
    observation, reward, done, information = batch.step(actions)
    goals += information['score_reward'].abs().sum()  # ended episodes restart inside the step

print(int(goals), 'goals')
```

Every call returns the same tensors and overwrites them, so clone what you keep. On a GPU, `environment_count` rises to thousands. The [quickstart notebook](https://github.com/cuju-football/engine/blob/main/examples/cuju_quickstart.ipynb) runs on Colab's free GPU.

## Speed

| Engine    | Machine         | Configuration                 | Environment steps per second |
| --------- | --------------- | ----------------------------- | ---------------------------- |
| GRF       | AMD EPYC 7763   | one process                   | 207                          |
| GRF       | AMD EPYC 7763   | 31 processes                  | 5,819                        |
| cuju, CPU | AMD EPYC 7763   | 31 threads                    | 38,497                       |
| cuju, GPU | NVIDIA RTX 4090 | 16,384 environments           | 32,582                       |

The numbers are for `5_vs_5` with four agents a side and `simple115v2`. [Performance](https://github.com/cuju-football/engine/blob/main/docs/performance.md) has the method and the commands that reproduce them.

## Exactness

The reference is GRF 2.10.3 at commit 3d9e754 as built by GCC for x86-64 Linux with glibc 2.39. Both backends give the same bits whatever the thread count, and the renderer's frames equal GRF's on Mesa llvmpipe on every machine. Testing against GRF is internal, and `cuju check` and `pytest tests/` tell whether a build reproduces the release. Issues are welcome, and a difference from GRF is the most valued report. [Exactness](https://github.com/cuju-football/engine/blob/main/docs/exactness.md) has the details.

## Coming from GRF

`cuju.gfootball` is GRF's own Python package over cuju's engine, so switching the imports runs GRF code unchanged. `cuju.env.create_environment` is a faster `create_environment`, and `cuju.BatchEnvironment` is the batch, where the speed is.

| GRF | cuju |
| --- | --- |
| `create_environment('5_vs_5', number_of_left_players_agent_controls=4)` | `BatchEnvironment('5_vs_5', left_agents=4, environment_count=...)` |
| `if done: env.reset()` | the step that ends an episode starts the next |
| `env.get_state()`, `env.set_state(state)` | `batch.get_state(index)`, `batch.set_state(index, state)` |
| `representation`, `rewards`, `stacked`, `action_set` | the same names |

## Documentation

| Page | Covers |
| --- | --- |
| [Getting started](https://github.com/cuju-football/engine/blob/main/docs/getting-started.md) | installation, the training loop, seeds, states, the machine config |
| [API reference](https://github.com/cuju-football/engine/blob/main/docs/api.md) | every public name, the command line, errors |
| [Observations, rewards and actions](https://github.com/cuju-football/engine/blob/main/docs/observations-rewards-actions.md) | what agents see, receive and do, frames and videos |
| [Scenarios](https://github.com/cuju-football/engine/blob/main/docs/scenarios.md) | the shipped and custom scenarios |
| [Migrating from GRF](https://github.com/cuju-football/engine/blob/main/docs/migrating-from-grf.md) | GRF's interfaces, option mapping, dumps |
| [Exactness](https://github.com/cuju-football/engine/blob/main/docs/exactness.md) | what equals GRF, how to check, differences |
| [Performance](https://github.com/cuju-football/engine/blob/main/docs/performance.md) | backends, devices, threads, memory |

The Cuju family also has [cuju-arena](https://github.com/cuju-football/arena) for submissions and tournaments, [cuju-render](https://github.com/cuju-football/render) for the browser viewer, and an [umbrella distribution](https://github.com/cuju-football/cuju) that installs all of them.

## Name

Cuju, written 蹴鞠 and pronounced *cùjū*, is the football of ancient China, and the earliest form of the game whose rules people wrote down. The name is a tribute, and the project is unconnected with any institution that preserves the game.

## Citing

Research that uses cuju is asked to cite both cuju and GRF.

```bibtex
@software{cuju2026,
    author  = {Koraleski, Andr{\'e}},
    title   = {cuju: a bit-exact, batched reimplementation of {Google Research Football}, for {CUDA}
             and {CPU}},
    year    = {2026},
    version = {0.4.0},
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

## License

cuju is licensed under Apache-2.0. It reimplements GRF, licensed under Apache-2.0 by Google LLC, and [NOTICE](https://github.com/cuju-football/engine/blob/main/NOTICE) lists every credit. Four math functions in `include/cuju/libm/`, sin, cos, acos and asinf, are ports of the GNU C Library under the LGPL 2.1 or later.

- Using cuju from open or proprietary code carries no obligation, which starts with distribution.
- Redistributing cuju's libraries means keeping LICENSE, NOTICE and LICENSES/ with them and offering the matching source, which every PyPI release carries as its sdist.
- Compiling the C++ headers into another program brings the LGPL into that program.
- The logo in `docs/assets/` is licensed under CC BY 4.0.
