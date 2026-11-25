# Recipes

**Recipes** for cuju are short programs and patterns that connect a batch to reinforcement learning
code. They cover a minimal training loop, rollout buffers, bootstrapping at the end of an episode,
staggered episodes, curricula across scenarios, self-play, evaluation against the built-in AI and on
fixed sets of matches, training for the Kaggle competition, seeding, one process per GPU, the size
of a batch, the number of batches per process, a vector adapter in the style of Gymnasium, the
return and length of episodes, and the interface of Google Research Football, or GRF. The learning
algorithm comes from separate training code or a reinforcement learning library, and cuju supplies
the environment side.

Three properties of a batch underlie the recipes. The tensors it returns are overwritten by the next
call, an episode that ends restarts inside the step, and the observations keep one slot for each
agent. Each property causes silent errors in training code that overlooks it. The most frequent are
storing a returned tensor instead of a copy, which makes every stored observation equal to the last
one, and reading `final_observation` where `done` is false instead of selecting with `done`.

Each recipe calls `policy(observation)`, a function that returns `int32` actions of shape
`[environment_count, agents]` on the device of the batch. A stand-in that returns idle actions is
`policy = lambda observation: torch.zeros(observation.shape[:2], dtype=torch.int32, device=observation.device)`.
Actions start at 0, which is the idle action, and run to `batch.action_count - 1`.

## Minimal training loop

The shortest complete loop is an advantage actor-critic, or A2C, on the CPU in
`academy_empty_goal_close`, an academy scenario that a policy solves by learning to shoot. It
serves as a starting point for checking a setup and for adaptation, with untuned hyperparameters.
The loop is a function, so that the recipe on curricula can call it for each of its stages.

```python
import cuju
import torch


def train(batch, model, optimizer, updates, horizon=16, gamma=0.99):
    """Trains a policy of one agent for a number of updates, and prints the goals per episode."""
    observation = batch.reset()[:, 0].clone()  # [environments, 115]
    goals = episodes = 0
    for update in range(updates):
        log_probabilities, values, entropies, rewards, dones = [], [], [], [], []
        for _ in range(horizon):
            output = model(observation)  # the action logits, then the value
            distribution = torch.distributions.Categorical(logits=output[:, :-1])
            action = distribution.sample()
            observation, reward, done, information = batch.step(action.to(torch.int32))
            observation = observation[:, 0].clone()  # the next step overwrites the batch's tensors
            log_probabilities.append(distribution.log_prob(action))
            values.append(output[:, -1])
            entropies.append(distribution.entropy())
            rewards.append(reward[:, 0].clone())
            dones.append(done.float())
            goals += int((information['score_reward'] == 1).sum())
            episodes += int(done.sum())
        # The end of an episode counts as terminal, as most work on GRF treats it.
        with torch.no_grad():
            target = model(observation)[:, -1]
        returns = []
        for reward, done in zip(reversed(rewards), reversed(dones)):
            target = reward + gamma * (1 - done) * target
            returns.append(target)
        advantage = torch.stack(returns[::-1]) - torch.stack(values)
        loss = (
            -(torch.stack(log_probabilities) * advantage.detach()).mean()
            + 0.5 * advantage.pow(2).mean()
            - 0.01 * torch.stack(entropies).mean()
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (update + 1) % 50 == 0:
            print(f'update {update + 1}: {goals / max(episodes, 1):.2f} goals per episode')
            goals = episodes = 0


batch = cuju.BatchEnvironment('academy_empty_goal_close', environment_count=64, device='cpu')
model = torch.nn.Sequential(
    torch.nn.Linear(115, 128), torch.nn.Tanh(), torch.nn.Linear(128, batch.action_count + 1)
)
train(batch, model, torch.optim.Adam(model.parameters(), lr=1e-3), updates=300)
```

The loop plays 300 updates of 16 steps in 64 environments, 307,200 environment steps in all, which
take a few minutes on the processor of a laptop. The goals per episode that it prints rise as the
policy learns to shoot. The observation keeps its agent axis, `[environments, 1, 115]` for the one
agent of the batch, so the loop indexes `[:, 0]` to give the model the shape `[environments, 115]`.
The actions of one agent go to `step` as `[environments]`.

Two lines are essential to its correctness. `observation[:, 0].clone()` keeps the next step from
overwriting the input that the gradient still needs, and `(1 - done)` stops the return at the end of
an episode, which the batch has already restarted. On the GPU, the loop takes `device='cuda'` for
the batch and `model.cuda()`. The `.to(torch.int32)` of the actions then allocates a tensor at every
step, which a copy into an `int32` buffer allocated once avoids, as described in
[Backends](backends.md). With 64 environments the GPU is slower than the CPU, and GPU runs use
thousands of environments.

## Rollout buffers

Since every step overwrites the tensors it returns, a rollout copies them into storage allocated
once, rather than appending clones to a list.

```python
steps, environments = 128, 1024
batch = cuju.BatchEnvironment(
    '5_vs_5', environment_count=environments, left_agents=4, device='cpu'
)
agents = batch.agents
observations = torch.empty((steps, environments, agents, 115))
rewards = torch.empty((steps, environments, agents))
dones = torch.empty((steps, environments), dtype=torch.bool)

observation = batch.reset()
for t in range(steps):
    observations[t].copy_(observation)
    # Random actions in place of a policy.
    actions = torch.randint(0, batch.action_count, (environments, agents), dtype=torch.int32)
    observation, reward, done, information = batch.step(actions)
    rewards[t].copy_(reward)
    dones[t].copy_(done)
```

On the GPU the buffers are allocated on the device as well, and the whole rollout stays there.

A list of clones is also correct, and for a short rollout on the CPU the difference is small. On the
GPU, preallocated buffers save one allocation per tensor per step, hold the rollout in one block of
memory that a learner slices without `torch.stack`, and keep the memory of a run constant from the
first update, so an out-of-memory error occurs at the start of a run rather than hours into it.

## Bootstrapping from the final observation

When an episode ends in the middle of a rollout, the observation that `step` returns already belongs
to the next episode. A value target that bootstraps at the end of the match takes the last
observation of the ended episode from `final_observation`.

```python
observation, reward, done, information = batch.step(actions)
ended = done.view(-1, *[1] * (observation.dim() - 1))  # broadcasts over the agents
next_observation = torch.where(ended, information['final_observation'], observation)
# value(next_observation) is the value after this step, for every environment
```

`torch.where` keeps the step free of device waits. A boolean index such as `observation[done]`
makes the host wait on the GPU at every step, since the size of its result depends on the data.

Whether to bootstrap at the end of a match is a modelling choice with a cost either way. The end of
the match time is a cut rather than a state of the game that the policy can influence, so in
principle the value after it comes from bootstrapping, as for any truncated episode. The vector
observations and the SMM contain no clock, however, so a value function that reads them cannot
anticipate the end, and a target that is cut at some times and bootstrapped at others from similar
observations is noisy. Most work on GRF treats the end as terminal, and comparisons with that work
follow the same convention. Bootstrapping is consistent when the value function receives the time
left, `steps_left` from a batch with `raw=True`, which distinguishes the two cases.
[Episodes and seeds](episodes-and-seeds.md) describes how an end at the time limit is told apart
from the other ends.

## Staggered episodes

Every environment of a batch kicks off at its first reset, so the episodes also end together, and
every few thousand steps a rollout consists entirely of kick-offs. An on-policy update learns from
the states of one rollout, and when those states are the opening seconds of a match in one update
and the closing seconds in another, the gradient varies with the phase of the match rather than
with the behavior of the policy. Cutting each first episode at a random length spreads the ends of
the episodes for the rest of the run.

```python
batch = cuju.BatchEnvironment('5_vs_5', environment_count=4096, left_agents=4, device='cuda')
observation = batch.reset()
generator = torch.Generator(device='cuda').manual_seed(0)
# One episode length per environment.
cut = torch.randint(1, 3001, (4096,), device='cuda', generator=generator)
step = torch.zeros(4096, dtype=torch.int64, device='cuda')
for t in range(3001):
    observation, reward, done, information = batch.step(policy(observation))
    step += 1
    batch.reset(mask=step == cut)  # each environment once, at its own step, without a wait
```

A masked reset starts the next episode of each chosen environment as the `reset()` of GRF would, so
every later episode remains a GRF episode of the seed rule. The cut episodes report no `done`, so
their last step is treated as a time limit or dropped. Academy scenarios that end within tens of
steps stagger themselves within a few episodes, so the technique applies to the full-length
scenarios.

## Curriculum across scenarios

A **curriculum** trains one policy on a sequence of scenarios, usually from a drill that teaches one
skill to the full match. Every environment of a batch plays one scenario with one set of agent
counts, so the buffers of a batch have one shape and the kernels run without a branch on the
scenario. A curriculum therefore creates one batch for each stage and trains the same model on them
in turn. The policy carries over from stage to stage as long as the observation and the action set
keep their size, which `simple115v2` and the `default` action set do for every scenario, with one
agent on the left. The function `train` of the [minimal training loop](#minimal-training-loop) runs
each stage.

```python
stages = [
    # scenario, environments, updates
    ('academy_empty_goal_close', 256, 100),
    ('academy_run_to_score_with_keeper', 256, 100),
    ('academy_3_vs_1_with_keeper', 256, 100),
    ('5_vs_5', 256, 200),
]
model = torch.nn.Sequential(torch.nn.Linear(115, 128), torch.nn.Tanh(), torch.nn.Linear(128, 20))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for index, (scenario, environment_count, updates) in enumerate(stages):
    with cuju.BatchEnvironment(
        scenario, environment_count=environment_count, device='cpu', seed=index
    ) as batch:
        print(scenario, batch.memory_bytes // 2**20, 'MiB')
        train(batch, model, optimizer, updates)
```

The `with` block frees the buffers of a stage when the stage ends, so the memory of the curriculum
is that of its largest stage. A curriculum that returns to earlier stages, or mixes scenarios in
every update, keeps several batches alive and steps each of them in turn, which costs the buffers of
every batch and nothing else. A GPU reserves the local memory of its kernels once per process, at
the first batch, and every later batch of the process runs within the same reservation.

A curriculum that mixes scenarios inside one update takes a rollout from each batch and joins them,
since the rollouts of different scenarios have the same feature size. The `seed` of each batch keeps
its matches apart from those of the others.

## Self-play

A batch with agents on both sides, such as `left_agents=4, right_agents=4`, places the left agents
first and the right agents after them. Each side sees the pitch from its own end and receives the
reward of its own team, so one policy can play both sides. The left agents occupy the slots up to
`batch.left_agents` and the right agents the slots after them, so two policies take their
observations and give their actions by slicing the agent axis.

```python
def random_policy(observation):  # a stand-in, a policy maps observations to actions
    return torch.randint(0, 19, observation.shape[:2], dtype=torch.int32)


batch = cuju.BatchEnvironment(
    '5_vs_5', environment_count=512, left_agents=4, right_agents=4, device='cpu'
)
left = batch.left_agents
observation = batch.reset()
current, past = random_policy, random_policy
actions = torch.cat([current(observation[:, :left]), past(observation[:, left:])], dim=1)
observation, reward, done, information = batch.step(actions)
```

The choice of opponent is the main design decision of self-play. A policy that plays against its
current self can cycle, learning to beat its latest version while losing the ability to beat older
ones. A pool of past snapshots for `past`, sampled at each rollout, is the usual remedy, and
evaluation against the built-in AI is the usual measure of progress against a fixed opponent.

### The built-in AI as opponent

Two mechanisms place the built-in AI on a side, and they produce different matches.

A side with no agents, as with `right_agents=0`, is the scenario of GRF with the built-in AI as the
other team. The papers and leaderboards of GRF evaluate against this form, and it is the form used
for evaluation.

Action 19, `builtin_ai`, in the `v2` and `full` sets, hands the player of an agent to the built-in
AI for that step, while the agent keeps its observations and rewards. Assigning 19 to the slots of
the right agents, as in `actions[:, batch.left_agents:] = 19`, turns the right side of a self-play
batch into a built-in AI side, so one batch serves both self-play and play against the AI. Its
matches differ from those of a batch without right agents from the first step, since the agents
still exist, so it serves training while evaluation uses the first form.

Action 0, `idle`, plays a team that idles, the equivalent of the `lazy` player of GRF.

## Evaluation against the built-in AI

Evaluation uses fixed matches that can be replayed. With `auto_reset=False` each environment plays
exactly one episode, and a dedicated `seed` fixes the matches of the evaluation.

```python
batch = cuju.BatchEnvironment(
    '11_vs_11_stochastic', environment_count=100, device='cpu', auto_reset=False, seed=12345
)
observation = batch.reset()
finished = torch.zeros(100, dtype=torch.bool)
goals_for = torch.zeros(100, dtype=torch.int32)
goals_against = torch.zeros(100, dtype=torch.int32)
while not finished.all():
    observation, reward, done, information = batch.step(policy(observation))
    live = ~finished
    score = information['score_reward']
    goals_for += (score == 1) & live
    goals_against += (score == -1) & live
    finished |= done
print('wins', int((goals_for > goals_against).sum()), 'of 100')
```

The same `seed` replays the same 100 matches, so two policies meet the same opponents.

The number of matches follows from the difference to be detected. A win rate measured over 100
matches has a 95% confidence interval of about 10 percentage points either way near 50%, which
leaves policies at 45% and 55% indistinguishable, while 1,000 matches narrow the interval to about
3 points. Matches in a batch are inexpensive, so evaluations with cuju can use more matches than is
customary with GRF, and the interval is reported with the rate. With `auto_reset=False` the
environments that finish early idle until the last one ends, which costs little in a full match,
since every full match lasts its 3,001 steps.

## Fixed evaluation sets

An **evaluation set** is a list of matches that stays the same from one evaluation to the next, so
that the scores of a policy at different times, or of different policies, compare match by match.
`reset(seeds=...)` names the match of each environment. The seed of environment `index` is the seed
of its episode, a number from 0 to 4,294,967,295, and the same seed plays the same match whatever
the `seed` of the batch, the backend or the batch size. The set of 256 matches below is `seeds = range(1000, 1256)`.

```python
def evaluate(policy, scenario='5_vs_5', first_seed=1000, matches=256, device='cpu'):
    """Returns the mean reward of the policy over the matches of the evaluation set."""
    seeds = torch.arange(first_seed, first_seed + matches)
    with cuju.BatchEnvironment(
        scenario, environment_count=matches, device=device, auto_reset=False
    ) as batch:
        observation = batch.reset(seeds=seeds)
        total = torch.zeros((matches, batch.agents), device=batch.device)
        finished = torch.zeros(matches, dtype=torch.bool, device=batch.device)
        while not finished.all():
            observation, reward, done, information = batch.step(policy(observation))
            total += reward * (~finished)[:, None]
            finished |= done
    return total.mean()
```

Creating the batch inside the function keeps the set fixed. An environment alternates the two forms
of a scenario from one episode to the next, as a GRF environment does over its resets, so the matches
of a second `reset(seeds=...)` on the same batch are those of the other form, with the kick-off to
the other team in `5_vs_5`. A batch kept across evaluations plays the same matches again on every
second evaluation, and the average of two consecutive evaluations covers both forms. A new batch
costs a fraction of a second on the CPU and a moment on the GPU, which is small beside an
evaluation of full matches.

The team-order flag of an environment follows the parity of its first seed, as in GRF, so a set of
consecutive seeds covers both orders in equal shares.

## Kaggle competition

The GRF competition on Kaggle plays `11_vs_11_kaggle` with one agent a side, and each agent receives
the `raw` observation of GRF. A batch plays the same scenario with `left_agents=1` and
`right_agents=1`.

`11_vs_11_kaggle` with one left agent and one right agent plays a ladder match, and with no right
agent plays against the built-in AI at difficulty 1.0, a fixed opponent that stands in for the
agents of the ladder. A submission reads `obs['players_raw'][0]`, a dictionary of the raw arrays of
GRF as lists. `cuju.env.grf_observation(raw, environment, agent, own_players, other_players)`
builds that dictionary from the raw tensors of a batch, with numpy arrays that `tolist()` turns into
the lists of Kaggle. A right agent sees its own team as the left team, so its call swaps the two
player counts.

```python
import cuju
import cuju.env


def agent(obs):  # stands in for the agent function of a submission, which runs unchanged
    return [0]


batch = cuju.BatchEnvironment(
    '11_vs_11_kaggle',
    environment_count=4,
    left_agents=1,
    right_agents=1,
    device='cpu',
    representation='raw',
)
batch.reset()
raw = {name: tensor.cpu() for name, tensor in batch.raw().items()}
entry = cuju.env.grf_observation(raw, 0, 1, batch.right_players, batch.left_players)
players_raw = {
    key: value.tolist() if hasattr(value, 'tolist') else value for key, value in entry.items()
}
action = agent({'players_raw': [players_raw]})  # the right agent's view
```

A loop over the environments and agents of a batch calls the function once per environment and step,
in Python, as Kaggle does, and a submission that plays a trained policy, or itself, runs at the speed
of that loop. The first episode of every environment is the even form with its own team order, as
Kaggle plays it.

The wrappers of GRF, applied to the dictionary of `grf_observation` or to the lists of Kaggle
converted back to numpy arrays, reproduce the observations of cuju.
`Simple115StateWrapper.convert_observation([entry], True)` equals `simple115v2`, and
`observation_preprocessing.generate_smm([entry])` equals `extracted`, bit for bit, so a submission
that applies these wrappers to `players_raw` sees what training saw.

The values agree while the types differ. The raw arrays of cuju are `float32`, whereas a Kaggle
agent holds the same numbers as Python floats, which numpy converts to `float64`. A feature computed
in `float32` during training, such as a distance, can therefore differ in its last bits from the
same numpy code in the submission, and computing such features in `float64`, with `.double()`, makes
the two agree.

[Observations](observations.md) gives the meaning of each raw array and the view of a right agent,
and [Scenarios](scenarios.md) describes the second half of `11_vs_11_kaggle`.

## Seeding

A single `seed` per run determines every match. Environment `index` plays its own sequence of
matches, independent of the batch size and of the number of threads. The default, `seed=0`, plays
the same matches in every run, so the runs of a sweep each take their own `seed`.

An experiment on several machines or GPUs gives each its own `seed`, as in
[One process per GPU](#one-process-per-gpu). Each machine then plays its own sequence of matches, and
no two machines repeat each other. The experiment draws its seeds as `base + rank`, with a `base`
spaced by the number of ranks from the `base` of every other run, so that two runs never share a
`seed`.

`reset(seeds=)` sets the seed of the next episode of each environment, which replays a single
match, as in [Fixed evaluation sets](#fixed-evaluation-sets). Training relies on the distinct
matches that the seed rule draws.

## One process per GPU

The arrangement with the fewest interactions on several GPUs is one process for each GPU, and in
each process one batch. A batch holds device memory and threads that stay with their process, so
every process creates its own batch on its own device, with a `seed` that no other process uses. The
convention `seed=base + rank` gives each rank its own sequence of matches, where environment `i` of
rank `r` plays as GRF after `random.seed((base + r) * 2**32 + i)`. The processes then collect
experience independently, and the learner averages the gradients across them, as `DistributedDataParallel`
does.

The script runs under `torchrun`, which starts one process for each GPU and sets `LOCAL_RANK`.

```python
# train.py, started with `torchrun --nproc-per-node=4 train.py`
import os

import torch
import torch.distributed as distributed

import cuju

base_seed = 1000

rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
distributed.init_process_group('nccl')
batch = cuju.BatchEnvironment(
    '5_vs_5', environment_count=4096, device=f'cuda:{rank}', seed=base_seed + rank, left_agents=1
)
model = torch.nn.Sequential(
    torch.nn.Linear(115, 128), torch.nn.Tanh(), torch.nn.Linear(128, batch.action_count + 1)
).cuda()
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train(batch, model, optimizer, updates=300)  # the function of the minimal training loop
```

`torch.multiprocessing` starts the processes from one script as well. The function that each process
runs receives its rank, and the batch is created inside it, since a batch cannot be pickled and
travel to a worker.

```python
import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing

import cuju


def run(rank, world_size, base_seed=1000):
    torch.cuda.set_device(rank)
    distributed.init_process_group(
        'nccl', init_method='tcp://127.0.0.1:29500', rank=rank, world_size=world_size
    )
    batch = cuju.BatchEnvironment(
        '5_vs_5', environment_count=4096, device=f'cuda:{rank}', seed=base_seed + rank
    )
    ...  # the model, the optimizer and `train`, as above


if __name__ == '__main__':
    count = torch.cuda.device_count()
    multiprocessing.spawn(run, args=(count,), nprocs=count)
```

Each process reserves the local memory of the kernels on its own GPU, and the batch of each GPU
holds as many environments as the memory of that GPU allows, as [Sizing a batch](#sizing-a-batch)
shows. GPUs of different sizes take different environment counts, and the gradients still average
over the processes. The processes share no memory and no state of the batches. On a machine with one
GPU and many cores, the same arrangement with `device='cpu'` and `config={'threads': 4}` puts one
batch on each group of cores.

## Sizing a batch

A batch reports the bytes that it has requested in `memory_bytes`, and the memory of a batch grows
by a fixed amount for each environment. Two small batches give the fixed part and the part for each
environment, so a program can choose the number of environments from the free memory of its GPU
before it creates the batch. The first batch of a process on a GPU also makes the driver reserve the
local memory of the kernels, which `memory_bytes` leaves out and `torch.cuda.mem_get_info` counts as
taken once the first batch exists. It comes to about 5.6 KB for each thread that the GPU runs at a
time, about 135 MiB on a 4 GB laptop GPU and about 1.05 GiB on an RTX 4090, whatever the number of
environments, so it matters most on a small GPU and when several processes share one. The program
creates its two small batches first, so that the free memory it reads already excludes that
reservation. It stops at 16,384 environments, the largest count that
[Performance](performance.md) measures.

```python
import cuju
import torch

scenario = '5_vs_5'


def memory(count):
    with cuju.BatchEnvironment(
        scenario, environment_count=count, left_agents=4, device='cuda'
    ) as batch:
        return batch.memory_bytes


small, large = memory(64), memory(128)
per_environment = (large - small) / 64
fixed = small - 64 * per_environment
reserve = 2 * 2**30  # room for the model, its optimizer and the rollouts
budget = torch.cuda.mem_get_info()[0] - reserve - fixed
count = min(16384, int(budget // per_environment) // 1024 * 1024)
batch = cuju.BatchEnvironment(scenario, environment_count=count, left_agents=4, device='cuda')
print(f'{count} environments, {batch.memory_bytes / 2**20:.0f} MiB')
```

The reserve depends on the model and the rollout buffers, and an observation of `extracted` takes
60 times the memory of a vector observation, as described in [Performance](performance.md). On the
CPU the same program with `device='cpu'` counts host memory, which bounds a large batch of
`extracted` observations on a machine with little memory, and `psutil.virtual_memory().available`
takes the place of `mem_get_info`.

## Batches per process

The arrangement with the fewest interactions is one batch per process and GPU, with as many
environments as fit. More environments per batch cost less than more batches. On an RTX 4090, 16,384
environments of `5_vs_5` run 32,582 steps per second and 1,024 environments run 9,920, so one step
of the large batch takes about 30% of the time of the 16 steps of the small one that cover the same
number of environments. On the CPU, one batch with `threads = 0` in the machine config already uses
every core. On a small GPU under Windows or WSL2, a single process with one batch avoids the tenfold
slowdown that two processes with a batch each cause when the driver spills device memory into system
memory, as described in [Performance](performance.md).

A process that needs a second batch on the same GPU, for evaluation against a fixed opponent or for
another scenario, can create it beside the first at the cost of its own buffers alone. The driver
reserves the local memory of the kernels once per process and GPU, at the first batch, and every
later batch of that process on that GPU runs within the same reservation. Separate processes each
make a reservation of their own, so several training runs that share one GPU each set it aside,
about 1 GiB per process on a GPU of 128 multiprocessors.

## Compiled step

`torch.compile(mode='reduce-overhead')` places the policy and the step in one CUDA graph once the
batch's buffers are marked as static inputs, which a `reset` after `torch.compile` does. The
compiled function therefore comes before the reset.

```python
import cuju
import torch

batch = cuju.BatchEnvironment('5_vs_5', environment_count=1024, left_agents=4, device='cuda')


@torch.compile(mode='reduce-overhead', fullgraph=True)
def act_and_step(observation):
    return batch.step(policy(observation))


with torch.no_grad():
    observation = batch.reset()  # after torch.compile, which marks the buffers
    for _ in range(1000):
        observation, reward, done, information = act_and_step(observation)
```

A batch reset before `torch.compile` plays the same bits, and its step runs outside the graph until
one more `reset` marks the buffers. [Backends](backends.md#compilation-with-torchcompile) describes
the operators behind the compiled step.

## Vector adapter in the style of Gymnasium

Libraries that train on Gymnasium's vector environments expect `reset` to return an observation and
a dictionary, and `step` to return an observation, a reward, `terminated`, `truncated` and a
dictionary, with one entry for each environment. A batch of one agent maps onto that interface by
dropping the agent axis, and a short class does so for a library that reads PyTorch tensors.

```python
import gymnasium
import numpy as np
import torch


class BatchVectorEnv:
    """Gymnasium's vector interface over a BatchEnvironment with one agent on the left."""

    def __init__(self, batch):
        self.batch = batch
        self.num_envs = batch.environment_count
        self.single_observation_space = gymnasium.spaces.Box(
            -np.inf, np.inf, batch.single_observation_space.shape[1:], np.float32
        )
        self.single_action_space = gymnasium.spaces.Discrete(batch.action_count)

    def reset(self, seed=None, options=None):
        # Gymnasium's seed becomes one seed for each environment, `seed + index`.
        seeds = None if seed is None else seed + torch.arange(self.num_envs)
        return self.batch.reset(seeds=seeds)[:, 0].clone(), {}

    def step(self, actions):
        observation, reward, done, information = self.batch.step(
            torch.as_tensor(actions, dtype=torch.int32, device=self.batch.device)
        )
        information = {
            'final_observation': information['final_observation'][:, 0].clone(),
            'score_reward': information['score_reward'],
        }
        truncated = torch.zeros_like(done)
        return observation[:, 0].clone(), reward[:, 0].clone(), done, truncated, information
```

The adapter follows GRF in reporting every end of an episode through `terminated`, with `truncated`
always false, as the Gymnasium interface of `cuju.env` does, and it restarts an episode inside the
step that ends it, so the observation after a `done` already belongs to the next episode and
`information['final_observation']` holds the last one. A library that expects the next call to do the
restart treats such a step as the first step of the next episode. The clones keep the tensors of the
adapter from changing under a library that stores them, at the cost of one copy for each step, and
a library that consumes each step at once drops them.

## Episode return and length

A batch returns the reward and the done flags of each step, and the return and the length of an
episode come from a few lines on top of `step`. The accumulators live on the device of the batch,
and `torch.where` keeps every operation free of waits, so the lines add no cost to a GPU step.

```python
class Episodes:
    """Accumulates the return and the length of the episodes that end in a batch."""

    def __init__(self, batch):
        self.batch = batch
        shape = (batch.environment_count, batch.agents)
        self.returns = torch.zeros(shape, device=batch.device)
        self.lengths = torch.zeros(batch.environment_count, device=batch.device)
        self.return_sum = torch.zeros(batch.agents, device=batch.device)
        self.length_sum = torch.zeros((), device=batch.device)
        self.count = torch.zeros((), device=batch.device)

    def step(self, actions):
        observation, reward, done, information = self.batch.step(actions)
        self.returns += reward
        self.lengths += 1
        ended = done[:, None]
        self.return_sum += torch.where(ended, self.returns, 0.0).sum(0)
        self.length_sum += torch.where(done, self.lengths, 0.0).sum()
        self.count += done.sum()
        self.returns.masked_fill_(ended, 0.0)
        self.lengths.masked_fill_(done, 0.0)
        return observation, reward, done, information

    def report(self):
        count = max(float(self.count), 1.0)
        return (self.return_sum / count).tolist(), float(self.length_sum) / count
```

`report` reads the accumulators once, at the moment of logging, so the host waits for the device
once for a report and never for a step. The return is the sum of the rewards of an agent over its
episode, and the length counts the steps of the episode. Every end of an episode counts, including
the end at the time limit, since GRF reports all ends through `done`. The class serves a batch with
`auto_reset=True`, where each environment ends an episode once and starts the next in the same step.

## GRF interface

For code written against the environment of GRF, `cuju.env.create_environment` takes the arguments
of GRF and returns one environment with the gym 0.21 interface of GRF, or the Gymnasium interface
with `api='gymnasium'`. It builds its spaces with Gymnasium, from `pip install 'cuju-engine[spaces]'`.
[Migrating from GRF](migrating-from-grf.md) gives the details.

## See also

- [API reference](api.md), for the signatures of every name used in the recipes
- [Concepts](concepts.md), for batches, bundles and environments
- [Rewards and actions](rewards-and-actions.md), for the action sets and `builtin_ai`
- [States](states.md), for saving and restoring a single environment
