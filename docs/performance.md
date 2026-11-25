# Performance

cuju steps `5_vs_5` with four agents a side 9.75 times as fast as one Google Research Football process on one core of an AMD EPYC 7763, and 6.62 times as fast on all 31 cores. On an NVIDIA RTX 4090, 16,384 environments reach 32,582 steps per second, with observations, actions and policy kept in device memory.

## Method

Throughput is environment steps per second over the whole batch. With four agents a side, one environment step moves four agents and counts once.

Each figure is the median of interleaved runs, so heat and background load affect all configurations alike. The percentage beside a figure is the spread over the rounds, `(max - min) / median`. Each run discards 100 steps and times the next 1,000, episode resets included. Random actions come from a fixed generator. GRF runs on the same machine, unrendered, with the same scenario, agents and representation, one process per environment. The `11_vs_11_stochastic` rows use one agent, on the left. The machine is a cloud instance with an RTX 4090 and an EPYC 7763 whose container quota gives 31 processors. cuju is the 0.1.0 wheel, and GRF is commit 3d9e754 with gym 0.25.2 and numpy 1.26.4. The snippet under the CPU table times a cuju row.

## CPU throughput

One core, with cuju and GRF pinned to the same processor, over three rounds:

| Scenario | GRF, one process | cuju, one environment | Ratio |
| --- | --- | --- | --- |
| `5_vs_5`, 4 agents a side | 207 (4.9%) | 2,018 (1.5%) | 9.75x |
| `11_vs_11_stochastic`, 1 agent | 163 (1.7%) | 914 (1.9%) | 5.61x |

All 31 processors, with 31 threads of 16 environments each against 31 GRF processes started together, over three rounds:

| Scenario | GRF, 31 processes | cuju, 31 threads | Ratio |
| --- | --- | --- | --- |
| `5_vs_5`, 4 agents a side | 5,819 (0.2%) | 38,497 (0.7%) | 6.62x |
| `11_vs_11_stochastic`, 1 agent | 4,391 (0.4%) | 25,088 (0.2%) | 5.71x |

Each GRF process keeps about 91% of its single-core rate when 31 run together, and each cuju thread about 62%. The `5_vs_5` rows time a batch as follows, with `environment_count` 1 on one core under `taskset -c 0` and 496 on all 31, and one GRF process stepped the same way for the baseline:

```python
import time

import torch

import cuju

batch = cuju.BatchEnvironment(
    '5_vs_5', environment_count=496, device='cpu', left_agents=4, right_agents=4
)
batch.reset()
generator = torch.Generator().manual_seed(0)


def step():
    shape = (batch.environment_count, batch.agents)
    batch.step(torch.randint(0, batch.action_count, shape, dtype=torch.int32, generator=generator))


for _ in range(100):
    step()
start = time.perf_counter()
for _ in range(1000):
    step()
print(
    1000 * batch.environment_count / (time.perf_counter() - start), 'environment steps per second'
)
```

The 11-a-side rows use `'11_vs_11_stochastic'` with `left_agents=1` and `right_agents=0`. On a laptop with a Ryzen 5 7535HS of 12 threads under WSL2, over one round:

| Scenario | cuju, one environment | cuju, 12 threads |
| --- | --- | --- |
| `5_vs_5`, 4 agents a side | 859 | 6,146 |
| `11_vs_11_stochastic`, 1 agent | 439 | 3,724 |

## GPU throughput

The RTX 4090 figures use the compute capability 8.9 code of the wheel over two rounds that agree within 0.6%:

| Scenario | 1,024 environments | 4,096 | 16,384 |
| --- | --- | --- | --- |
| `5_vs_5`, 4 agents a side | 9,920 (0.6%) | 22,661 (0.2%) | 32,582 (0.0%) |
| `11_vs_11_stochastic`, 1 agent | 8,679 (0.0%) | 17,885 (0.2%) | 24,221 (0.0%) |

The same snippet with `device='cuda'`, `environment_count=16384` and a `torch.cuda.synchronize()` before each reading of the clock gives the 16,384 column. Draw the actions on the device with `device='cuda'` in `torch.randint` and in the generator.

Google Colab GPUs, one round each with no spread, on Xeon hosts of 12 processors, or eight for the T4:

| GPU (compute capability) | Scenario | 1,024 environments | 4,096 | 16,384 |
| --- | --- | --- | --- | --- |
| A100-SXM4-40GB (8.0) | `5_vs_5`, 4 agents a side | 5,462 | 8,940 | 10,334 |
| | `11_vs_11_stochastic`, 1 agent | 4,350 | 6,342 | 6,981 |
| L4 (8.9) | `5_vs_5`, 4 agents a side | 6,048 | 9,772 | 11,270 |
| | `11_vs_11_stochastic`, 1 agent | 4,503 | 6,627 | 7,316 |
| T4 (7.5) | `5_vs_5`, 4 agents a side | 2,761 | 3,601 | 3,859 |
| | `11_vs_11_stochastic`, 1 agent | 1,954 | 2,369 | 2,335 |

With one warp per environment, throughput favors high clock rates over many slower cores, so the A100 at 1.41 GHz stays near the L4 and well behind the 4090. The T4 is saturated at 4,096 environments of `11_vs_11_stochastic`.

The RTX 3050 Laptop GPU of the laptop above, 4 GB with compute capability 8.6, over one round. Its memory holds 4,096 environments of either scenario, while 16,384 need about 6 GiB of buffers.

At 1,024 and 4,096 environments it steps `5_vs_5` 3,332 and 3,982 times per second and `11_vs_11_stochastic` 2,384 and 2,718.

The 3050 figures come from the kernels of 0.2, which ran 1.7% to 5.8% faster on this GPU and needed 51,312 bytes of local memory per thread against 6,768 today.

## Choosing a device

GRF runs on the CPU, so compare GPU figures with the CPU of the same machine. At 16,384 environments the 4090 steps `5_vs_5` about 15% slower than 31 processors and `11_vs_11_stochastic` about 3% slower, and on the laptop 12 threads beat the GPU. The GPU earns its place through what it keeps in device memory. Observations, actions and policy stay there, so a training loop copies nothing between batch and network, and the CPU stays free for the optimizer, logging and evaluation.

Use `'cuda'` for training at scale, `'cpu'` for development, evaluation and small batches, and `'auto'` for code that runs with and without a GPU. Both backends give identical outputs, so the choice never changes results, and one program can hold batches on both.

## Batch size

GPU throughput rises with the number of environments until the device is full, then levels off. Start at 4,096 environments and double until steps per second stop rising or memory runs out. On the CPU, about 16 environments per thread keep every core busy.

## CPU threads and placement

The key `threads` of the machine config sets the size of the pool. The default `threads = 0` uses one thread per processor the process may run on, within the cgroup quota. Outputs are identical for every thread count.

```toml
[run]
threads = 12
pin_threads = true
numa = true
```

`pin_threads` binds each pool thread to a processor of its own. It lowers the variance of the step time on a dedicated machine and raises it on a shared one. `numa` forms one group of threads per NUMA node, which helps on servers with two or more sockets and leaves single-node machines unchanged. The [API reference](api.md) lists every key and its default. The argument `config={'threads': 12}` of the constructor sets a key for one batch.

### Async stepping

`step_async` starts a CPU step in a thread of the batch and returns, and `step_wait` returns what `step` would. The usual use is a CPU batch that feeds a learner on the GPU. The gain is bounded by the shorter of the two durations, so it is largest when learner and step take similar time.

```python
for _ in range(steps):
    batch.step_async(actions)
    learner_update()
    observation, reward, done, information = batch.step_wait()
```

With `pin_memory = "auto"`, the default, CPU outputs are page-locked when PyTorch sees a CUDA GPU, which makes the `non_blocking` copy asynchronous. Set it to `false` on a machine with little memory. On the GPU, `step_async` behaves as `step`.

## Several GPUs

Run one process per GPU, each with a batch on its own device and a seed of its own. A launcher such as `torchrun` sets `LOCAL_RANK`.

```python
import os

rank = int(os.environ['LOCAL_RANK'])
batch = cuju.BatchEnvironment(
    '5_vs_5', environment_count=4096, device=f'cuda:{rank}', left_agents=4, seed=1000 + rank
)
```

Only gradients cross between GPUs. Keep the number of ranks, the batch size and the base seed to reproduce a run.

## Memory

A batch of `5_vs_5` with four agents and `simple115v2` takes about 0.38 MiB per environment, so 16,384 environments take about 6 GiB. The attribute `batch.memory_bytes` gives the bytes the batch has requested. Each batch also takes about 31 MiB regardless of size. The table was measured on the 3050 with `allocator = "torch"`, and leaves out the kernel reservation, the PyTorch context and the model.

| Scenario | Representation | MiB per environment | 1,024 environments | 4,096 |
| --- | --- | --- | --- | --- |
| `5_vs_5`, 4 agents a side | `simple115v2` | 0.38 | 419 MiB | 1,583 MiB |
| | `extracted` | 0.80 | 845 MiB | 3,288 MiB |
| | `raw` | 0.38 | 417 MiB | 1,575 MiB |
| `11_vs_11_stochastic`, 1 agent | `simple115v2` | 0.37 | 413 MiB | 1,557 MiB |
| | `extracted` | 0.42 | 468 MiB | 1,771 MiB |
| | `raw` | 0.37 | 413 MiB | 1,559 MiB |

An observation of `simple115v2` holds 460 bytes per agent, and the `extracted` map at 96 by 72 cells holds 27,648 bytes, 60 times as much. Stacking multiplies either by four. [Observations, rewards and actions](observations-rewards-actions.md) gives each layout.

The first GPU batch of a process also makes the driver reserve local memory for the kernels, about 5.6 KB per thread the GPU can run at once beyond its 1 KB stack. That is about 1.05 GiB on a 4090, which runs 196,608 threads, and 135 MiB on a 3050, which runs 24,576. The reservation depends on the GPU alone. Later batches in the process share it, and each process pays it once. By default cuju allocates outside PyTorch, so `torch.cuda.memory_allocated` leaves the batch out and `torch.cuda.mem_get_info` counts everything. With `allocator = "torch"` PyTorch counts the batch as well.

### Shared-memory spill on Windows and WSL2

A GPU batch compares its request with the free memory of its GPU at creation. Under a Linux driver a request that does not fit raises `DeviceError` with the bytes requested and free. Under Windows and WSL2 the driver may place the batch in shared system memory, where the GPU reads it over PCIe. The batch then runs and every kernel is many times slower. Two training processes on a 4 GB 3050 under WSL2 each ran about ten times slower than one alone.

On these systems cuju issues a `UserWarning` naming both amounts in MiB, which is the main sign of a spill. Resize before training, with fewer environments or a smaller model. The Task Manager shows a spill as shared GPU memory above zero. With `log = true` in `[run]`, each batch logs the device memory it took.

## torch.compile and CUDA graphs

On the GPU a step is one kernel launch on the current PyTorch stream. When actions are in device memory, a CUDA graph captures it together with the policy, and each replay performs one step and computes the next actions.

```python
actions = torch.zeros((4096, 4), dtype=torch.int32, device='cuda')
graph = torch.cuda.CUDAGraph()
with torch.no_grad(), torch.cuda.graph(graph):  # after warm-up steps on a side stream
    observation, reward, done, information = batch.step(actions)
    actions.copy_(policy(observation).argmax(-1))
for _ in range(1000):
    graph.replay()
```

A replay produces the same bits as eager stepping. Capture covers `step` with device actions, `reset_done`, and `reset(mask=...)` with a device mask, while `reset()`, `get_state` and `set_state` run between replays.

Wrapping the step and the policy in `torch.compile(mode='reduce-overhead', fullgraph=True)` gives the same single graph.
Call `torch.compile` first and `reset` the batch after it, so the batch marks its buffers as static inputs. Compiling needs PyTorch 2.4 or later. On Windows, PyTorch ships without Triton, so compiling a GPU step raises `TritonMissing` and the eager step gives the same results.

Graphs pay off with a small policy or a few hundred environments. With thousands of environments the step kernel dominates and a graph saves little.

## Cost of rendering frames

A batch that renders frames with `pixels`, `pixels_gray` or `frames=True` spends almost all its time drawing, on the host's processors whatever the backend. On one core of a Ryzen 5 7535HS at about 2.3 GHz, an eleven a side frame of 1280 by 720 pixels takes about 0.56 seconds, 0.30 with two threads, 0.16 with four and 0.10 with eight or twelve. GRF's own frames on llvmpipe take 0.72 seconds on one core and 0.22 with four threads, against 13 milliseconds for a step without rendering. Figures scale with the clock, which ranges from 2.3 to 4.5 GHz on this laptop.

A batch of eight environments draws about nine to ten frames a second in all, and a batch of 32 about ten. Each environment draws on a worker of its own and holds about 200 MB. The keys `render_workers` and `render_threads` set the split. The first match of a process takes about three seconds to prepare. At ten frames a second, a million pixel observations take more than a day, so `representation='extracted'` gives a picture at the speed of the simulation.

The throughput tables above apply to batches that do not render. The first steps after a reset run faster than later ones, so time hundreds of steps and exclude the first GPU call, which loads the kernel.

See [Getting started](getting-started.md) for installation and a training loop, and [Exactness](exactness.md) for the equality of CPU and GPU outputs.
