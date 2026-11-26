# Controllers

A controller sends the actions of a set of agents. It can be a trained network, a scripted bot, a Kaggle `main.py`, GRF's own players, a program in another environment or a container, or anything that answers an observation with actions. `cuju.agents` is the layer between such controllers and the batch. The batch stays a pure simulator, and a `Match` joins it to controllers, one for each side that you want played for you.

## The contract

A controller is an object with three methods and a declared representation.

```python
from cuju.agents import Actions, Controller, Observation, Spec


class Mine(Controller):
    representation = 'simple115v2'

    def start(self, spec: Spec) -> None: ...
    def act(self, observation: Observation) -> Actions: ...
    def close(self) -> None: ...
```

`start` begins a session once for each setup of a batch, and `Spec` tells the controller the scenario, the number of environments and agents, the action set, the control, a seed of its own, and which side it plays. `act` receives the arrays of its representation shaped `[environments, agents, ...]` and answers integer actions shaped `[environments, agents]`. The arrays are the batch's own tensors on the batch's device, with no copy, so a controller reads them and never writes to them. The answer may be a tensor or an array on any device, and it may carry named extras, which are numeric arrays that the match never reads and that a recording can keep.

Every controller sees its own team as the left team, and the engine mirrors the actions back as GRF does. The observation also holds `first`, which is true on the first observation of an episode and is where per-episode state resets, `active`, which is false where the answer is not used, the sizes of both teams, the actions that the engine applied at the last step, the step of each environment and a `meta` dictionary that wrapping parts may fill.

A breach of the contract raises `ControllerError`, which names the side, the step, the environment and a reason. The breaches are a wrong shape, an action outside the controller's set, a crash, a closed pipe, no answer in time and a protocol mismatch. Judging a breach, as a forfeit or as anything else, is up to the caller. An exception that a controller raises itself passes through with the side and the step attached.

## Loading controllers

`cuju.agents.load` builds a controller from a TOML file, a dictionary, an entry string, a folder or a controller. A folder with a `main.py` loads as a Kaggle agent, an entry such as `mypackage.module:factory` names a factory, and a function loads as a `function` controller. The standard names are `function`, `per_environment`, `kaggle`, `grf`, `bot`, `lazy`, `random`, `pool`, `process` and `builtin`. `cuju agents parts` lists them and the parts that installed packages provide through the entry point group `cuju.agents.parts`.

A tree nests, and a node may wrap another through its `inner` key. A factory receives its options, the unbuilt inner node and a `load` function. An unknown key raises an error that names its place in the tree, such as `inner.inner.step_seconds`. Relative paths resolve against the folder of the file that holds them.

```toml
agents = 1
entry = "pool"
weights = [3, 1]

[[inner]]
entry = "kaggle"
path = "submissions/first"

[[inner]]
entry = "random"
```

The nodes above a `process` node run in the host and the nodes below it run in its child, which can load only what its own environment can import and the two loaders built into `client.py`, `kaggle` and `module:object`. Limits, referees and rules are parts that people write and compose in this way. The engine ships none.

## Matches

```python
import cuju
from cuju import agents

batch = cuju.BatchEnvironment(
    '5_vs_5', environment_count=64, left_agents=1, right_agents=1, raw=True, auto_reset=False
)
with agents.Match(batch, left='frozen/v12.toml', right='submissions/other') as match:
    result = match.run(episodes=100)
print(result.summary())
```

You name only the sides that have a controller. A side that you leave out is yours, so `match.reset()` and `match.step(actions)` give you the observations of that side alone, and the controllers of the other side act inside `step`. `step_async` and `step_wait` overlap the answers of process controllers with your own work. `match.set(right=other)` gives a side a new controller that each environment takes at its next episode, and `match.episodes` gives the return and length of the episodes that your side finished, without waiting for the device on every step.

`run` plays complete episodes when every side has a controller. It returns a `Result` with the score and the steps of each episode and a summary of the wins, draws and losses of the left side, the goal difference and a 95 percent interval of the win rate. A breach stops the run and appears in `result.error` beside the episodes that finished. `play` and `play_batch` build the batch from the scenario and run it, and `to_gymnasium` and `to_pettingzoo` turn a match into a vector environment of Gymnasium or a parallel environment of PettingZoo.

A match checks its sides when it is made and refuses a mismatch, naming both numbers. The batch owns the number of agents of a side, and the control of a controller must agree with it. A controller that observes another representation than the batch gives needs a batch made with `raw=True`. A fixed assignment, or a controller on the `v2` set, needs a batch of the `v2` or `full` action set, where action 19 hands a player to the built-in AI. A controller that you pass as an object stays yours to close, and one that the match builds from a file closes with it.

A synchronous match waits for every answer, so a deterministic controller plays the same match every time. Nondeterminism comes from the controller's own code, or from the late answers of a host that steps in real time.

## Recipes

### Opponents in a training batch

Give the opponent a side and keep the other for the learner.

```python
batch = cuju.BatchEnvironment(
    '5_vs_5', environment_count=1024, left_agents=4, right_agents=4, device='cuda', auto_reset=True
)
match = agents.Match(batch, right='frozen/v12.toml')
observation = match.reset()
observation, reward, done, information = match.step(policy(observation))
```

### Leagues and self-play

A `pool` draws one inner controller for each environment at each reset. `set_pool` swaps the checkpoints at the next reset of the environments that you choose, and a controller that is no longer used closes.

```toml
entry = "pool"
weights = [3, 1]

[[inner]]
entry = "function"
function = "league:latest"
representation = "simple115v2"

[[inner]]
entry = "function"
function = "league:oldest"
representation = "simple115v2"
```

Self-play with your own network on both sides needs no controllers. Make a batch with `left_agents=1` and `right_agents=1` and step it with the actions of both. A side with no agents is played by the built-in AI.

### Kaggle agents

A folder with a `main.py` plays as Kaggle plays it. Each environment gets a fresh module instance, which loads again at each first observation, with the folder as the working folder and first on `sys.path`. The agent gets Kaggle's observation, and `configuration` is an empty `Struct`.

```
cuju agents check submission --scenario 5_vs_5 --twice
cuju agents play submission builtin@0.7 --scenario 11_vs_11_kaggle --episodes 20
```

The command line runs a folder in a process of its own, which `--in-process` turns off. `check` plays short matches on both sides, reports each failure with its reason, the actions taken and the first, minimum, mean and maximum time of the calls, and with `--twice` compares two runs. It judges no time. To hold an agent to Kaggle's limits, wrap it in a part of your own.

```python
import time

from cuju.agents import Controller, ControllerError, load


class KaggleLimits(Controller):
    def __init__(self, inner, load=load, step_seconds=0.5, overage_seconds=60.0):
        self.inner = load(inner)
        self.representation = self.inner.representation
        self.control = getattr(self.inner, 'control', None)
        self.step_seconds, self.bank = step_seconds, overage_seconds

    def start(self, spec):
        self.inner.start(spec)

    def act(self, observation):
        observation.meta['overage'] = self.bank
        began = time.perf_counter()
        answer = self.inner.act(observation)
        self.bank -= max(0.0, time.perf_counter() - began - self.step_seconds)
        if self.bank < 0:
            raise ControllerError('the overage of Kaggle ran out', 'timeout')
        return answer

    def close(self):
        self.inner.close()
```

Put the wrapper outside a `process` node, so it times the call from outside.

### Other environments, containers and other languages

A `process` node runs its inner tree in a new process. Give it a `python`, or a `command` for anything else, such as docker, ssh or a program in another language. The command is started with the text `{stub}` replaced by a short program that reads `client.py` from the standard input, so a container needs only Python and numpy. `cuju agents client` prints the file.

```toml
entry = "process"
command = ["docker", "run", "--rm", "-i", "--init", "image", "python", "-X", "utf8", "-c", "{stub}"]

[inner]
entry = "kaggle"
path = "/agent"
```

A process isolates crashes and environments and is not a security boundary. Run untrusted code in a container as above. A `prefix` such as `prlimit` or `systemd-run --scope -p MemoryMax=1G` keeps a memory limit in one line, and `pid` and `terminate()` let a part of your own limit the process further. The default timeout is 300 seconds for the load and for each answer. [The protocol](protocol.md) specifies the messages for a program in another language, and `cuju agents check --command` checks it.

## Control

The control of a controller sits in its `[control]` table or in the `control` argument of a match. `players` is the number of agents, `assignment` is `dynamic` or `fixed`, and with `fixed` the `selection` names the formation indices or the roles of the players that the controller plays. `goalkeeper` is `agent`, `builtin` or `scenario`, and `action_set` is `default`, `v2` or `full`. The control is checked against the controllable players of the scenario. The host fills action 19 where an answer is not used, that is for the players outside a fixed selection, at set pieces where GRF hands the selection back, for players off the pitch and for late answers.

## Training from files

`cuju.recording.load` reads many recordings into arrays for offline learning and decodes only the fields that you ask for. `BatchEnvironment.get_states` and `set_states` save and load every environment in one call, with one copy between the device and the host.
