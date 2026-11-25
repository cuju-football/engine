# Performance

The **performance** of cuju is its throughput in environment steps per second, measured against
Google Research Football, or GRF, on the same machine. On one processor of an AMD EPYC 7763, a
single environment of `5_vs_5` with four agents a side runs 9.75 times as many steps per second as
one GRF process, and on 31 processors the ratio is 6.62. On an NVIDIA RTX 4090, 16,384 environments
of the same scenario reach 32,582 steps per second, about 15% fewer than the 31 processors of the
same machine, with the observations, the actions and the policy kept in device memory.

GPU throughput rises with the number of environments until the device is full, and device memory,
at about 0.38 MiB per environment plus a fixed reservation for the kernels, bounds the size of a
batch. On small GPUs under Windows and WSL2, memory beyond the capacity of the device spills into
shared system memory, which slowed two concurrent training processes on a 4 GB laptop GPU by a
factor of about ten. The attribute `batch.memory_bytes` gives the memory that a batch has requested,
and a GPU batch that requests more than the free memory of its device warns at creation on Windows
and WSL2, where the driver spills such a batch into shared memory, and raises `DeviceError` with the
bytes requested and free under a Linux driver.

## Method

Throughput is counted in environment steps per second over the whole batch. With four agents on a
side, one environment step moves four agents, and it counts once.

Each figure is the median of interleaved runs. The configurations alternate within each round, so
that heat and background load affect them alike, and the percentage beside each figure is the spread
over the rounds, `(max - min) / median`. Each run discards its first 100 steps and times the next
1,000, episode resets included, as they occur in training. Random actions come from a fixed
generator, so every run plays the same matches. GRF runs on the same machine, unrendered, with the
same scenario, agents and representation, one process per environment. The script
`examples/benchmark.py` reproduces every row.

The machine is a cloud instance with an NVIDIA RTX 4090 and an AMD EPYC 7763 of the Zen 3
generation, whose container quota gives the process 31 processors. cuju is the release wheel,
`cuju 0.1.0`, and GRF is commit 3d9e754 built with its own flags, with gym 0.25.2 and numpy 1.26.4.
The `11_vs_11_stochastic` runs have one agent, on the left.

## CPU throughput

On a single processor, one environment of cuju and one GRF process, both pinned to the same
processor, were compared over three rounds.

| Scenario | GRF, one process | cuju, one environment | Ratio |
| --- | --- | --- | --- |
| `5_vs_5`, 4 agents a side | 207 (4.9%) | 2,018 (1.5%) | 9.75x |
| `11_vs_11_stochastic`, 1 agent | 163 (1.7%) | 914 (1.9%) | 5.61x |

On the whole quota, 31 threads of 16 environments each were compared with 31 GRF processes started
together, over three rounds.

| Scenario | GRF, 31 processes | cuju, 31 threads | Ratio |
| --- | --- | --- | --- |
| `5_vs_5`, 4 agents a side | 5,819 (0.2%) | 38,497 (0.7%) | 6.62x |
| `11_vs_11_stochastic`, 1 agent | 4,391 (0.4%) | 25,088 (0.2%) | 5.71x |

The ratio narrows from one processor to 31. For `5_vs_5`, each GRF process keeps about 91% of its
single-processor rate when 31 run together, and each cuju thread about 62%. The `5_vs_5` rows come
from three commands, for GRF on one processor, for cuju on one processor and for cuju on all 31.

```sh
taskset -c 0 python examples/benchmark.py --grf --scenario 5_vs_5 --left 4 --right 4
taskset -c 0 python examples/benchmark.py --scenario 5_vs_5 --device cpu --environments 1 --left 4 --right 4
python examples/benchmark.py --scenario 5_vs_5 --device cpu --environments 496 --left 4 --right 4
```

The thread count is a key of the machine config, as [Installation](installation.md#machine-config)
describes. The default `threads = 0` takes the processors that the process may use, one under
`taskset -c 0` and 31 under the quota of the machine, which is what the second and third commands
measured. The all-core GRF figure is the sum of 31 copies of the first command run at once. The
11-a-side rows use `--scenario 11_vs_11_stochastic --left 1 --right 0` for GRF and for cuju.

On a laptop with an AMD Ryzen 5 7535HS of 12 threads under WSL2, a build of the release tree for
the same commands gave the following rates over one round.

| Scenario | cuju, one environment | cuju, 12 threads |
| --- | --- | --- |
| `5_vs_5`, 4 agents a side | 859 | 6,146 |
| `11_vs_11_stochastic`, 1 agent | 439 | 3,724 |

## GPU throughput

The figures of the RTX 4090 and of the GPUs of Google Colab were measured on the wheel of cuju
0.1.0, and the first table of the RTX 3050 Laptop GPU on a build of the same release tree, whose
kernels kept a copy of the large lists of the simulation for every thread, as described in
[Batch size](#batch-size). The compact kernels of this release were measured on the RTX 3050 Laptop
GPU alone, against the kernels before them, and the table at the end of this section gives that
comparison.

On the RTX 4090, the machine code of the wheel for compute capability 8.9 was measured over two
rounds, which agree within 0.6%.

| Scenario | 1,024 environments | 4,096 | 16,384 |
| --- | --- | --- | --- |
| `5_vs_5`, 4 agents a side | 9,920 (0.6%) | 22,661 (0.2%) | 32,582 (0.0%) |
| `11_vs_11_stochastic`, 1 agent | 8,679 (0.0%) | 17,885 (0.2%) | 24,221 (0.0%) |

The `5_vs_5` figure at 16,384 environments comes from one command, and `--scenario` and
`--environments` select the others.

```sh
python examples/benchmark.py --scenario 5_vs_5 --left 4 --right 4 --device cuda --environments 16384
```

On the GPUs of Google Colab, the same wheel and method were run for one round each, so these figures
carry no spread. Each runtime has an Intel Xeon host with 12 processors, or eight for the T4.

| GPU (compute capability) | Scenario | 1,024 environments | 4,096 | 16,384 |
| --- | --- | --- | --- | --- |
| A100-SXM4-40GB (8.0) | `5_vs_5`, 4 agents a side | 5,462 | 8,940 | 10,334 |
| | `11_vs_11_stochastic`, 1 agent | 4,350 | 6,342 | 6,981 |
| L4 (8.9) | `5_vs_5`, 4 agents a side | 6,048 | 9,772 | 11,270 |
| | `11_vs_11_stochastic`, 1 agent | 4,503 | 6,627 | 7,316 |
| T4 (7.5) | `5_vs_5`, 4 agents a side | 2,761 | 3,601 | 3,859 |
| | `11_vs_11_stochastic`, 1 agent | 1,954 | 2,369 | 2,335 |

With one warp per environment, throughput favours high clock rates over a large number of slower
cores, so the A100, at 1.41 GHz, stays near the L4 and well behind the 4090. The T4 is saturated at
4,096 environments of `11_vs_11_stochastic`.

The RTX 3050 Laptop GPU of the same laptop, with 4 GB and compute capability 8.6, was measured with
the same build over one round. Its memory holds 4,096 environments of either scenario, while 16,384
need about 6 GiB of buffers besides the local memory of the kernels.

| GPU (compute capability) | Scenario | 1,024 environments | 4,096 |
| --- | --- | --- | --- |
| RTX 3050 Laptop GPU, 4 GB (8.6) | `5_vs_5`, 4 agents a side | 3,332 | 3,982 |
| | `11_vs_11_stochastic`, 1 agent | 2,384 | 2,718 |

The compact kernels of this release, which keep the large lists of the simulation in a work area of
each environment, were measured against the kernels before them on the same RTX 3050 Laptop GPU,
with builds of CUDA 12.0 for compute capability 8.6. The two builds alternated within each round,
with 100 timed steps after 30 discarded ones in four rounds at 1,024 environments, and with 150
timed steps in three rounds at 4,096. The compact kernels cost between 1.7% and 5.8% of the
throughput, most in `5_vs_5` at 4,096 environments. The comparison of single changes showed no one
change behind the cost, since a build with every list back in local memory, a build with the earlier
sorts, a build with the earlier inlining and a build with two removed list copies restored each ran
within 2% of the compact kernels.

| Scenario | Environments | Earlier kernels | Compact kernels | Cost |
| --- | --- | --- | --- | --- |
| `5_vs_5`, 4 agents a side | 1,024 | 3,116 | 3,022 | 3.0% |
| | 4,096 | 3,726 | 3,510 | 5.8% |
| `11_vs_11_stochastic`, 1 agent | 1,024 | 2,348 | 2,310 | 1.7% |
| | 4,096 | 2,662 | 2,600 | 2.3% |

The compact kernels need 6,768 bytes of local memory per thread, where the earlier kernels needed
51,312, in builds of CUDA 12.0 for each of the four compute capabilities of the wheel. Beyond the
stack of 1 KB that the driver keeps for every thread, the reservation of a first batch therefore
falls from 1.15 GiB to 135 MiB on the RTX 3050 Laptop GPU, which runs 24,576 threads at a time, and
from 9.2 GiB to 1.05 GiB on a GPU of 128 multiprocessors, such as the RTX 4090, which runs 196,608.
The drop in free memory around a first creation on the RTX 3050 Laptop GPU agreed with the estimate
within 1.5 MiB for both builds.

GRF runs on the CPU, so the GPU figures compare with the CPU of the same machine. At 16,384
environments the 4090 steps `5_vs_5` about 15% slower than 31 processors, and `11_vs_11_stochastic`
about 3% slower, and on the laptop the 12 threads step both scenarios faster than its GPU. The GPU
earns its place through what it keeps in device memory. The observations, the actions and the policy
stay there, so a training loop copies nothing between the batch and the network, and the CPU remains
available for the rest of the work, such as the optimizer, logging and evaluation.

## Batch size

On the GPU, throughput rises with the number of environments until the device is full and then
levels off. A search for the batch size starts conveniently at 4,096 environments and doubles the
count until the steps per second stop rising or the memory runs out. Past the point where the rate
levels off, more environments add memory and latency per update without adding throughput.

Each environment of `5_vs_5` with four agents and `simple115v2` occupies about 0.38 MiB of device memory, so
16,384 environments occupy about 6 GiB. Creating the first batch of a process on a GPU also makes
the driver reserve the local memory of the kernels, which it then holds until the process ends. The
driver keeps a stack of 1 KB for every thread that the GPU can run at a time from the creation of
its context, and the first launch raises it to the local memory of the largest kernel, about 6,800
bytes per thread, so the reservation adds about 5.6 KB per thread. That amounts to about 1.05 GiB
on an RTX 4090, which runs 196,608 threads at a time, and to about 135 MiB on a 4 GB RTX 3050 Laptop
GPU, which runs 24,576. By default the library allocates outside the allocator of PyTorch, so
`torch.cuda.memory_allocated` leaves this memory out, while `torch.cuda.mem_get_info` counts all of
it. With `allocator = "torch"` in the machine config, PyTorch counts the buffers of the batch, as
described in [Backends](backends.md).

The reservation depends on the size of the GPU and on nothing else. Every environment count, every
scenario and every representation takes the same amount, so a small batch for evaluation reserves as
much as a large one for training. The reservation was once nearly nine times larger, about 50 KB per
thread or 9.2 GiB on a GPU of 128 multiprocessors, because each of the 32 threads that run an
environment kept its own copy of the large lists of the simulation, such as the candidate animations
of a selection and the commands queued for a player. Those threads all hold the same values, so the
kernels now keep one copy of these lists per environment, in a work area of 12 KB of device memory
that the batch allocates beside its worlds. The work area scales with the number of environments,
like the rest of the batch, and `memory_bytes` counts it, while the reservation
stays fixed. For 4,096 environments the work area takes 48 MiB, against a reservation that shrank by
about 1.0 GiB on the RTX 3050 Laptop GPU and by about 8.2 GiB on the RTX 4090.

Several batches in one process share one reservation. The driver makes it at the first launch of
the kernels on a GPU and keeps it for the whole process, so a second batch on the same GPU, for
evaluation or for another scenario, adds only its own buffers. Each process holds a reservation of
its own, so several training processes on one GPU each pay it, and a process that runs several
batches on one GPU pays it once.

`batch.memory_bytes` returns the bytes that a created batch has requested, counting the buffers,
the worlds and the shared tables of the batch, and the work area above. It leaves out the reservation
for the kernels, which the driver makes at the first launch. The drop in free memory that
`torch.cuda.mem_get_info` reports around the first creation on the RTX 3050 Laptop GPU, once the
creation of a second batch of the same size is subtracted, agreed within 1.5 MiB with the reservation
computed from the local memory of each kernel, the stack of the device and the threads that the GPU
runs at a time, for both builds. The difference between the `memory_bytes` of two batches of different
environment counts gives the memory of one environment, and a small batch created first gives a
figure for the fit of a planned count, together with the reservation, the context of PyTorch and the
model, before the large batch is allocated. A batch that does not fit raises `DeviceError` under a
Linux driver, with the bytes requested and the bytes free, and warns on Windows and WSL2, as the
section on [small and shared GPUs](#small-and-shared-gpus) describes. [Recipes](recipes.md) gives a
search for the largest batch that fits a budget.

On the CPU, about 16 environments per thread keep every core busy. The default `threads = 0` of the
machine config uses every processor that the process may run on, within the CPU quota of its
cgroup, as described in [Backends](backends.md).

The representation also affects memory. An observation of `simple115v2` holds 115 `float32` values,
460 bytes per agent, while the SMM of `extracted` at the default 96 by 72 cells holds four channels
of one byte each, 27,648 bytes per agent, and stacking multiplies either by four. An observation of
the SMM is therefore 60 times the size of a vector observation, which matters for rollout buffers
that hold every step of an update. [Observations](observations.md) gives the layout of each
representation.

The memory of a batch by scenario and representation was measured on the RTX 3050 Laptop GPU with
`allocator = "torch"` in the machine config, which counts every buffer of the batch. Each batch also takes about 31 MiB
regardless of its size, and the figures leave out the local memory of the kernels, the context of
PyTorch and the model. The SMM costs more than its observation alone, since the batch also keeps the
last observation of each ended episode in `final_observation`.

| Scenario | Representation | MiB per environment | 1,024 environments | 4,096 |
| --- | --- | --- | --- | --- |
| `5_vs_5`, 4 agents a side | `simple115v2` | 0.38 | 419 MiB | 1,583 MiB |
| | `extracted` | 0.80 | 845 MiB | 3,288 MiB |
| | `raw` | 0.38 | 417 MiB | 1,575 MiB |
| `11_vs_11_stochastic`, 1 agent | `simple115v2` | 0.37 | 413 MiB | 1,557 MiB |
| | `extracted` | 0.42 | 468 MiB | 1,771 MiB |
| | `raw` | 0.37 | 413 MiB | 1,559 MiB |

## Small and shared GPUs

A GPU with a few gigabytes of memory, such as that of a laptop, fills quickly, since the batch, the
local memory of the kernels, the context of PyTorch and the model share it. When the memory runs
out, a Linux driver fails the allocation, and creating the batch raises `DeviceError` with a message
that names the bytes requested and the bytes free on the GPU. A Windows
driver under WDDM, the driver model that WSL2 also uses, may instead move memory into the shared
system memory of the machine, which the GPU reads over the PCIe bus. The allocation then succeeds,
and every kernel that touches that memory runs many times slower. Two training processes that each
held a batch on a 4 GB RTX 3050 Laptop GPU under WSL2 both ran about ten times slower than one
process alone.

A GPU batch compares at creation the memory it requested with the free memory of the GPU
immediately before, and when the request is larger it issues a `UserWarning` that names both
amounts in MiB. With `allocator = "torch"`, the memory that the cache of PyTorch holds unused counts
as free, since the batch may take it. Under a Linux driver such a request fails, and the creation
raises the error that names the memory, so the warning matters on Windows and WSL2, where the batch
is created and runs anyway, and there it is the main sign of a spill. A run that shows the warning
is resized before training begins, with fewer environments or a smaller model on the same GPU. For
the first batch of the process on a GPU, the comparison also counts the local memory that the
creation reserves for the kernels, and the warning names that share. Later batches on the same GPU
find the reservation held and compare their own memory alone.

One process with one batch per GPU, sized to the free memory that `torch.cuda.mem_get_info()[0]`
reports less room for PyTorch and the model, avoids the spill. The `memory_bytes` of a small batch
scaled to the planned count gives the size of the large batch for that comparison, and the figures
of the table above give the reservation that a first batch adds for the kernels. On a laptop,
evaluation on the CPU with `device='cpu'` beside training on the GPU takes the place of a second GPU
batch. A spill appears in the Windows Task Manager as shared GPU memory greater than zero, together
with a drop in steps per second of an order of magnitude. With the key `log = true` in the machine
config, each batch logs at creation the device memory it took, so a log of the run shows how the
memory of the GPU was spent.

```toml
[run]
log = true
```

## Rendering

A batch that renders GRF's frames, with `pixels`, `pixels_gray` or `frames=True`, spends
almost all of its time drawing, on the host's processors whatever the backend. On one core of a
laptop's AMD Ryzen 5 7535HS at about 2.3 GHz, a frame of 1280 by 720 pixels of an eleven a side
match takes about 0.56 seconds, and with four threads about 0.16 seconds, against 0.72 and 0.22
seconds for llvmpipe on the same laptop. A batch of eight environments draws about nine to ten
frames a second in all and a batch of 32 about ten, with each environment on a worker of its own
and about 200 MB of memory per environment. The figures scale with the clock of the processor,
which on that laptop ranges from about 2.3 to 4.5 GHz. The rates of the
tables above therefore apply to batches that do not render, and
[Frames and videos](rendering.md#speed) gives the figures of the renderer, the split of workers and
threads by the keys `render_workers` and `render_threads`, and advice on
[training on pixels](rendering.md#training-on-pixels).

## Timing

The first steps after a reset run faster than later ones, because the players spread out and the
built-in AI has more to consider as a match develops, so a short window after kick-off overstates
the rate, and timings span hundreds of steps. The first call on the GPU loads the kernel and is
excluded from timings. An 11-a-side match costs more than a five-a-side one, roughly in proportion
to its number of players, as the single-processor rates of 2,018 and 914 steps per second show.

## See also

- [Backends](backends.md), for the execution model and the settings of each backend
- [Exactness](exactness.md), for the equality of the CPU and GPU outputs that the figures compare
- [Installation](installation.md), for the wheel, the GPUs it covers and the machine config
- [Recipes](recipes.md), for training loops that keep a batch on the GPU
- [Frames and videos](rendering.md), for the speed and memory of GRF's frames
