# Recording

`cuju.recording` keeps one match as one file. The file holds the engine state at the start, the action of every agent at every step and the rewards, so the engine plays the match again bit for bit. It can also hold GRF's raw observation after every step and arrays of your own, such as the values of a policy. The extension is `.cjr`, and `numpy.load` reads the file without cuju installed.

The module needs only the packages that cuju already installs. `import cuju.recording` stays light, and PyTorch loads when you record or convert.

## What a recording holds

A recording is one episode of one environment. The file is a zip archive in numpy's `.npz` layout, deflate compressed, and the same recording writes the same bytes each time. In the shapes, `T` is the number of steps and `A` the number of agents, the left agents first.

| Member | Type | Contents |
| --- | --- | --- |
| `header.json` | JSON | The scenario, the agents of each team, the action set, the reward terms, the difficulties, the number of steps, the players of each team, the engine version, the provenance and the team names when given. It also holds the format name `cuju-recording` and its major version, 1. |
| `state.npy` | `uint8 [bytes]` | The engine state before the first step, in GRF's state format. Missing when the source holds none. |
| `actions.npy` | `int16 [T, A]` | The action of each agent at each step, and -1 where it is unknown. |
| `rewards.npy` | `float32 [T, A]` | The reward of each step. |
| `done.npy` | `bool [T]` | True at the step that ends the episode. |
| `end.npy` | `uint8 []` | 0 for an episode still running, 1 for one the scenario ended and 2 for the time limit. |
| `step/<field>.npy` | as GRF's raw observation | The match before the first step and after each step, `T + 1` rows. Optional. |
| `array/<name>.npy` | any numeric type, `[T + 1, ...]` | An array of yours, row `t` for the observation before step `t`. Optional. |

The step fields are GRF's raw observation keys in the left team's view of the pitch. They are `ball`, `ball_direction`, `ball_rotation`, `ball_owned_team` and `ball_owned_player`, six fields each for `left_team` and `right_team` from `team` to `team_roles`, then `score`, `steps_left` and `game_mode`, and `active`, `designated` and `sticky_actions` for each agent. Values keep the types of the engine. A batch of right agents alone sees the pitch from the right end, so its fields turn back to the left team's view.

Row 0 of a step field or an array belongs to the state before the first step, and row `t` to the state after step `t`. The policy sees row `t` and the engine answers with row `t + 1`. The row after the last step of an array holds no value that a policy saw, so it is NaN for floating point and 0 for other types.

The state and the actions are what replay the match. The step fields and the rewards can be derived from them and stay in the file for readers without the engine. A reader that finds a newer major version than its own refuses the file and names both versions.

```python
import json
import numpy as np

data = np.load('match.cjr')
header = json.loads(data['header.json'])
ball = data['step/ball']
```

## Recording from a batch

`record` follows a batch and hands each finished episode of the environments you list to a function. The function decides where a recording goes, so it can write a file, fill a queue or stay in memory.

```python
import cuju
import cuju.recording as cr

batch = cuju.BatchEnvironment('5_vs_5', environment_count=64, left_agents=4, raw=True)


def keep(match):
    match.write(f'matches/{match.provenance["episode"]}.cjr')


with cr.record(batch, environments=[0, 1], on_episode=keep) as recorder:
    batch.reset()
    for _ in range(3000):
        recorder.add('value', values)
        batch.step(policy(batch))
```

| Argument | Meaning |
| --- | --- |
| `batch` | A `cuju.BatchEnvironment`. |
| `environments` | The indices to follow, or one index. Required. |
| `on_episode` | Called with a `Recording` for each finished episode. |
| `steps` | `True` by default, which stores the raw observation after every step and needs a batch made with `raw=True` or the representation `raw`. With `False` a recording holds the state, actions, rewards and done flags alone. |
| `provenance` | A dictionary kept in the header of every recording. |
| `teams` | The names of the left and right team, kept in the header. |

A recorder reads the state of an environment when each episode starts and the raw observation at every step. A GPU step waits for the device while it is recorded, so list the environments you need and no more.

An episode ends when the batch ends it, and with `auto_reset=True` the recorder starts the next one in the same call. The `provenance` of each recording holds `environment` and `episode`, where `episode` counts the episodes that the recorder handed over, so it names each file. The `end` of a recording is 1 when the scenario ended the episode and 2 at the time limit. An episode that was cut off has the end 0, which happens when `close` runs, when the batch closes or when a reset restarts the environment. A recorder attached after `reset` starts at the state of that moment, and the state it keeps replays from there.

`recorder.add(name, values)` attaches an array for the observation that the policy sees now, before the next `batch.step`. A tensor with the first axis of `environment_count` is sliced to the recorded environments, and a tensor with one row for each recorded environment is taken as it is. The values are copied at once, and a GPU tensor is copied without waiting for the device, so the policy can overwrite its buffers. Each recording holds the array as `array/<name>`. A name keeps its shape and type, and a different one raises `RecordingError`.

A step that runs inside a CUDA graph or `torch.compile` does not call the observers of the batch. Pass its outputs to `recorder.capture` instead, with `{'reset': mask}` after a reset, where `mask` is the mask that `reset` took or `None` for every environment, and with `{'actions': actions, 'reward': reward, 'done': done}` after a step.

`recorder.close()` hands the running episodes over as unfinished, removes the recorder from the batch and may be called more than once. The `with` block calls it, and a batch that closes first does the same.

## Reading and replaying

```python
match = cr.read('match.cjr')  # a path, an open binary file or bytes
match.actions, match.rewards, match.done, match.end
match.step['ball']  # numpy arrays, one row for each step and the start
match.array['value']
match.state, match.provenance, match.header

data = match.write()  # bytes
match.write('copy.cjr')  # a path, whose folders are made
match.write(open_file)  # an open binary file
```

A file that fails to read raises `RecordingError`, which is a `ValueError` and names the cause. A missing path raises `FileNotFoundError`. Code that plays matches its own way builds a `Recording` from arrays.

```python
match = cr.Recording(
    scenario='5_vs_5',
    left_agents=4,
    actions=actions,  # [T, agents], -1 where unknown
    rewards=rewards,  # [T, agents], zeros without it
    end=1,  # done is derived, or pass done
    state=state,  # bytes of the engine state, optional
    step={'ball': ball},  # [T + 1, ...], optional
    array={'value': value},  # [T + 1, ...], optional
    provenance={'run': 'a'},
)
```

The constructor checks every shape and raises `RecordingError` for one that does not fit. The arguments `right_agents`, `action_set`, `difficulty`, `players`, `teams`, `reward_terms` and `engine` complete the header.

`match.replay(representation='simple115v2', device='cpu')` plays the state and the actions in a one-environment batch and returns the observations as numpy arrays, `[T + 1, agents, ...]` for the vector and image representations and a dictionary of such arrays for `raw`. Where the recording holds step fields it compares every row with the replay bit for bit and raises `RecordingError` at the first difference, naming the step and the field. It also raises when the replay ends the episode at another step than the recording does. A replay needs the state and the action of every step. A scenario that is a file rather than a shipped name replays with `scenario='path.py'`, because the header keeps the name alone. Both backends give the same bits, so a recording made on a GPU replays on a CPU.

`match.with_steps(device='cpu')` returns a copy whose `step` comes from the replay. It turns a recording made with `steps=False` into one that holds the raw observation.

## Converting

`cr.from_grf_dump(path, device='cpu')` converts a dump that GRF or cuju wrote. It reads the dump with `cuju.dumps`, which runs no code of the file. The converter then plays the dump's actions again from its configuration, compares the observation of every step with the dump, and returns the recording only when every step is equal. A dump that differs, that does not start at its kick off, that has players other than agents or that uses another number of physics steps than 10 raises `RecordingError` with the reason. The source may also be bytes or an open binary file.

`cr.from_kaggle(path)` converts an episode of the GRF competition from Kaggle. The episode holds no engine state, so the recording holds the steps, the actions and the end, with no state and zero rewards.

## Command line

`cuju recording` follows the contract of the other commands. With `--json` each command prints one object whose first key is `schema`, and the exit status is 0 for success, 1 for a negative result such as a file that does not read or a replay that differs, and 2 for a usage error.

| Command | Meaning |
| --- | --- |
| `cuju recording show FILE [--json]` | The header, the number of steps, the score at the end and the names of the step fields and arrays. |
| `cuju recording check FILE [--replay] [--device D] [--json]` | Validates every member. With `--replay` it also plays the file in the installed engine and exits 1 when it does not reproduce. |
| `cuju recording convert PATH... -o DIR [--device D] [--overwrite] [--json]` | Converts GRF dumps and Kaggle episodes to files of the folder `DIR`. A path that fails is reported and the others go on. |

```sh
cuju recording show match.cjr
cuju recording check match.cjr --replay --device cpu
cuju recording convert episode_done.dump -o recordings
```

The schemas are `recording.show/1`, `recording.check/1` and `recording.convert/1`, and errors use `error/1`.
