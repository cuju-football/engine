# Getting started

This page installs cuju-engine, runs a first training loop and covers what a new user meets first, episodes, seeds, evaluation and saved states. Every argument and method is in the [API reference](api.md).

## Install

cuju needs Python 3.10 or later and PyTorch 2.3 or later, or 2.2 on macOS with an Intel processor. pip installs PyTorch when it is missing.

```sh
pip install cuju-engine
cuju check
```

`cuju check` prints the versions, the processor and the GPUs, plays the shipped matches on each backend and ends with `PASSED` or `FAILED` in one to two minutes. [Exactness](exactness.md) explains each line. If the `cuju` command is not on the path, `python -m cuju check` runs the same check.

| Platform | Backends | Notes |
| --- | --- | --- |
| Linux on x86-64 | CPU and CUDA | needs AVX2, BMI2 and FMA, so Intel since Haswell and AMD since Excavator |
| Windows on x86-64 | CPU and CUDA | install a CUDA build of PyTorch from pytorch.org first for the GPU |
| Windows through WSL2 | CPU and CUDA | use the Linux wheel, with the NVIDIA driver installed on Windows |
| Linux on Arm, macOS | CPU | any 64-bit Arm processor, macOS 11 or later, or 10.15 with PyTorch 2.2 on Intel |

The CUDA backend needs a driver for CUDA 12 and no CUDA toolkit. `device='cuda'` also needs a PyTorch build with `torch.cuda.is_available()` true, and installing PyTorch from the index that pytorch.org lists before cuju selects the build. The wheel holds code for the T4 and RTX 20 series at compute capability 7.5, the A100 at 8.0, the A10 and RTX 30 series at 8.6, and the L4, L40 and RTX 40 series at 8.9. Other GPUs, such as the H100 and the RTX 50 series, run cuju from a source build. The extra `cuju-engine[spaces]` adds Gymnasium for the spaces of a batch, and `cuju-engine[pixels]` adds GRF's media for the pixel representations and frames. [Migrating from GRF](migrating-from-grf.md) covers the extras of GRF's own package.

### Common install errors

| Symptom | Fix |
| --- | --- |
| Illegal instruction, or an error that the processor lacks AVX2, BMI2 or FMA | Use a machine with those instructions, or enable them in the virtual machine. |
| CUDA reports no kernel image for the device | The GPU is outside the table above. Build from source, or use `device='cpu'`. |
| `MissingFileError` naming cuju-grf-media | Run `pip install 'cuju-engine[pixels]'`. |

### Upgrading and source builds

When upgrading from cuju 0.2.1, run `pip uninstall cuju` before installing 0.3.0 or later. If `import cuju` already finds an empty folder, `cuju-repair` prints the repair command. A source build needs CMake 3.24 or later, a C++ compiler and, for the GPU library, the CUDA toolkit 12.0 or later. Clone `https://github.com/cuju-football/engine` and run `pip install .` in it. The build targets the GPU of the build machine, and `CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=90"` names another. Run `cuju check --long` afterward. The repository `Dockerfile` builds an image with both libraries, started with `docker run --rm --gpus all cuju:0.4.0 python3 -m cuju.check`.

## Properties of a batch

A `BatchEnvironment` plays thousands of matches at once, with five properties.

1. Outputs are views of shared buffers. `reset` and `step` return tensors that the next call overwrites, so a value that outlives its step is cloned.
2. The step that ends an episode starts the next one. Its `done` is true, its observation is the first of the next episode, and `information['final_observation']` holds the last observation of the one that ended.
3. Every environment has the same agents. Outputs have the shape `[environment_count, agents, ...]`, with the agent axis kept for one agent.
4. Options that shape the buffers are fixed at creation. Speed settings live in the machine config at the end of this page.
5. Environment `index` of a batch with `seed` plays the matches GRF plays after `random.seed(seed * 2**32 + index)`, whatever the batch size, backend or thread count.

## A first training loop

This advantage actor-critic learns to shoot in `academy_empty_goal_close`. It is a starting point with untuned hyperparameters.

```python
import cuju
import torch


def train(batch, model, optimizer, updates, horizon=16, gamma=0.99):
    observation = batch.reset()[:, 0].clone()  # [environments, 115]
    for update in range(updates):
        log_probabilities, values, rewards, dones = [], [], [], []
        for _ in range(horizon):
            output = model(observation)  # the action logits, then the value
            distribution = torch.distributions.Categorical(logits=output[:, :-1])
            action = distribution.sample()
            observation, reward, done, information = batch.step(action.to(torch.int32))
            observation = observation[:, 0].clone()  # the next step overwrites the buffer
            log_probabilities.append(distribution.log_prob(action))
            values.append(output[:, -1])
            rewards.append(reward[:, 0].clone())
            dones.append(done.float())
        target = model(observation)[:, -1].detach()
        returns = []
        for reward, done in zip(reversed(rewards), reversed(dones)):
            target = reward + gamma * (1 - done) * target  # an end of episode is terminal
            returns.append(target)
        advantage = torch.stack(returns[::-1]) - torch.stack(values)
        loss = -(torch.stack(log_probabilities) * advantage.detach()).mean() + 0.5 * advantage.pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


batch = cuju.BatchEnvironment('academy_empty_goal_close', environment_count=64, device='cpu')
layers = [torch.nn.Linear(115, 128), torch.nn.Tanh(), torch.nn.Linear(128, batch.action_count + 1)]
model = torch.nn.Sequential(*layers)
train(batch, model, torch.optim.Adam(model.parameters(), lr=1e-3), updates=300)
```

On a GPU, pass `device='cuda'` and call `model.cuda()`. A GPU pays off with thousands of environments, and 64 run faster on the CPU. For longer rollouts, copy each step into tensors allocated once, such as `rewards[t].copy_(reward)`, so a learner slices one block.

### Episode return and length

Wrapping `step` gives the return and length of finished episodes. The accumulators stay on the device, so a GPU step never waits.

```python
returns = torch.zeros(batch.environment_count, device=batch.device)
lengths = torch.zeros_like(returns)
totals = torch.zeros(3, device=batch.device)  # return, length and count of finished episodes

def step(actions):
    observation, reward, done, information = batch.step(actions)
    returns.add_(reward[:, 0])
    lengths.add_(1)
    ended = done.float()
    totals.add_(torch.stack([(returns * ended).sum(), (lengths * ended).sum(), ended.sum()]))
    returns.mul_(1 - ended)
    lengths.mul_(1 - ended)
    return observation, reward, done, information
```

Divide the first two totals by the third when logging, which is the only time the host waits.

### Curriculum across scenarios

Create one batch per stage and train one model on each. The policy carries over because `simple115v2` and the default action set have the same size in every scenario with one left agent. [Scenarios](scenarios.md) lists them.

```python
# model and optimizer as in the first loop
stages = [('academy_empty_goal_close', 100), ('academy_3_vs_1_with_keeper', 100), ('5_vs_5', 200)]
for index, (scenario, updates) in enumerate(stages):
    with cuju.BatchEnvironment(scenario, 256, device='cpu', seed=index) as batch:
        train(batch, model, optimizer, updates)
```

### One process per GPU

Start one process per GPU with one batch each and a seed of its own, created inside the process since a batch cannot be pickled.

```python
# train.py, started with `torchrun --nproc-per-node=4 train.py`, model as above
import os, torch.distributed as distributed
rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
distributed.init_process_group('nccl')
batch = cuju.BatchEnvironment('5_vs_5', 4096, device=f'cuda:{rank}', seed=1000 + rank)
model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[rank])
```

## Episodes and resets

`reset()` starts every environment and returns the first observation. After that, an environment whose episode ends restarts inside the same `step`. In the example, `last` holds the final observation where an episode ended and the current one elsewhere.

```python
observation, reward, done, information = batch.step(actions)
last = torch.where(done[:, None, None], information['final_observation'], observation)
```

Use `torch.where` and not `final_observation[done]`, which makes the host wait for the GPU. `final_observation` is meaningful only where `done` is true, and with `representation='raw'` it is a dictionary of arrays.

`reset(mask=mask)` resets the environments where a bool tensor is true and leaves the others untouched. With `auto_reset=False` an environment stays at its end, returning its last observation, a true `done` and a reward of 0, until `reset_done()` or `reset()` restarts it. That suits evaluation with one episode per environment.

To tell the time limit from other ends, create the batch with `raw=True` and test `information['final_raw']['steps_left'][:, 0] == 0` where `done` is true.

## Seeds and evaluation

The default `seed=0` plays the same matches in every run, so give each run of a sweep and each rank its own `seed`. `reset(seeds=...)` names the next episode of each environment by a list, an array, an integer tensor or one integer, and a fixed set of seeds is a fixed evaluation set.

```python
def evaluate(policy, scenario='5_vs_5', first_seed=1000, matches=256):
    seeds = torch.arange(first_seed, first_seed + matches)
    with cuju.BatchEnvironment(scenario, matches, device='cpu', auto_reset=False) as batch:
        observation = batch.reset(seeds=seeds)
        total = torch.zeros((matches, batch.agents))
        finished = torch.zeros(matches, dtype=torch.bool)
        while not finished.all():
            observation, reward, done, information = batch.step(policy(observation))
            total += reward * (~finished)[:, None]
            finished |= done
    return total.mean()
```

Creating the batch inside the function keeps the set fixed. An environment alternates the two forms of a scenario between episodes, so a second `reset(seeds=...)` on the same batch gives the kick-off to the other team in `5_vs_5`. `reset(seeds=..., reverse=...)` also sets the team-order flag to replay an earlier run. [Exactness](exactness.md) covers what a seed fixes.

## Saving and loading states

`get_state(index)` returns one environment as `bytes` in GRF's format, and `set_state(index, state)` loads it into any environment of the same configuration, or into GRF. A state holds the whole match at one step, 50 to 110 KB, and each call copies between the device and the host.

```python
batch = cuju.BatchEnvironment('5_vs_5', environment_count=4, device='cpu', left_agents=4)
batch.reset()
for _ in range(100):
    batch.step(torch.zeros((4, 4), dtype=torch.int32))  # everyone idle
state = batch.get_state(0)
batch.set_state(1, state)  # environment 1 plays on from the moment of environment 0
```

A loaded state continues bit for bit as the match that saved it. `set_state` writes into the tensors the batch already returned, so copy a reward or done flag before loading into an environment that just ended. A state of another scenario, agent count or difficulty raises `cuju.errors.StateError` naming the fields that differ. A batch with `stacked=True` takes `stacked=True` in both calls. A state is a pickle, so load states from trusted sources only. For many matches from one kick-off, a seed is cheaper. [Migrating from GRF](migrating-from-grf.md) covers moving states between the two engines.

## Machine config

The machine config is a TOML file for settings that change speed and leave every output unchanged. Its path is `CUJU_CONFIG` when set, else `~/.config/cuju/engine.toml` on Linux and macOS and `%APPDATA%\cuju\engine.toml` on Windows. A missing file keeps every default.

```toml
[run]
threads = 12        # 0 for every processor the process may use
pin_memory = true
```

A `config={'threads': 4}` argument overrides a key for one batch, and `batch.run_config` shows each value and its source. The [API reference](api.md) lists every key, and [Performance](performance.md) says when each pays.
