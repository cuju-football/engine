# Scene

`cuju.scene` gives the exact scene of a match, everything a renderer or an analysis needs that comes from the match state and changes only when GRF changes. It holds no camera and no look, only numbers in numpy arrays, with the step, the clock, the score, the referee, the ball, and the position, rotation and state of every player and official.

Reading the scene changes nothing in a match. Every output and every state stays as it was, and the extraction runs only when you ask for it, so a batch that never reads a scene pays nothing.

## Reading a scene

```python
import cuju
import cuju.scene as scene

batch = cuju.BatchEnvironment('5_vs_5', environment_count=64)
batch.reset()

frames = scene.of_batch(batch, environments=[0, 1])
frames.ball_position.shape     # (2, 3)
frames.joints.shape            # (2, 25, 13, 3)
```

`of_batch` reads the scene of the chosen environments as they are now, on the CPU or the GPU where the batch lives, and returns `Frames` with one frame for each environment. Leave out `environments` to read all of them.

`follow` keeps the scene after each step, with the events of that step.

```python
for frames, events in scene.follow(batch, environments=[0]):
    batch.step(actions)
    if events[0]:
        print(events[0])
```

The first item is the scene at the start, with no events. Each later item is the scene after one step or reset of the batch since the previous item, so the loop body may step the batch, or the batch may run first and the loop read the steps afterward. The iteration ends when no step is waiting. `follow` returns a `Follower`, which also works as a context manager and stops following with `close`.

An environment that resets itself at the end of an episode shows the first scene of the next episode after that step, without events, because the finished world is gone by then. Create the batch with `auto_reset=False` to see the last scene of an episode.

`of_recording` replays a recording from its state and actions and returns the scene of its start and of every step.

```python
import cuju.recording as cr

frames, events = scene.of_recording(cr.read('match.cjr'), device='cpu')
frames.shape      # (steps + 1, 1)
events[0]         # the events of the match
```

A match followed live equals the same match replayed from its recording, bit for bit.

## Frames

`Frames` holds numpy arrays whose leading axes are `[environments]` for one step, or `[steps, environments]` for a followed or replayed match. Fields are attributes, `frames.data` is the structured array behind them, and `frames[index]` and `Frames.stack` work as on arrays. Integers are 32 bit and floats are 32 bit. A value of -1 means none.

| Field | Shape after the leading axes | Holds |
| --- | --- | --- |
| `episode` | | The count of resets of the environment, which changes when a new episode starts. |
| `step` | | GRF's step counter, which is -1 right after a reset and counts the steps in play. |
| `steps_per_second` | | The steps in a second of the match, 10 for GRF's ten ticks of 10 ms. |
| `duration` | | The steps of the match, GRF's `game_duration`. |
| `match_ms` | | The match clock in milliseconds. |
| `score` | 2 | The goals of the left and the right team. |
| `half` | | 0 before the match, 1 the first half, 2 the second. The shipped scenarios play one half, as GRF's do, and a scenario reaches 2 only when it sets GRF's `second_half`. |
| `in_play` | | 1 while the ball is in play. |
| `game_mode` | | GRF's game mode of the observation, the set piece being played, 0 for normal play. |
| `set_piece` | | The set piece that the referee has stopped play for, 0 when none. |
| `restart_position` | 3 | Where that set piece restarts. |
| `possession` | 2 | The team and the player that own the ball. |
| `last_touch` | 3 | The team, the player and the kind of the last touch, a value of `TOUCHES`. |
| `offside_player`, `offside_called` | | The player that is offside, and whether the referee called it. |
| `foul` | 3 | The kind, a value of `FOULS`, the offender and the victim. |
| `goal` | 2 | The team and the scorer of the last goal. |
| `ball_position` | 3 | The ball in meters. |
| `ball_rotation` | 4 | The ball's orientation, a quaternion. |
| `ball_touches_net` | | 1 while the ball touches a net. |
| `card_kind`, `card_position`, `card_rotation` | 3 or 4 | The card in the referee's hand, 0 none, 1 yellow, 2 red, with its position and rotation. |

The fields of the players have one more axis, `[players]`, in front of their own shape. It holds 25 entries in the default build, the room of 11 for the left team, 11 for the right team and 3 officials, the referee and two linesmen. A team of fewer players leaves the end of its room empty. Every field that names a player, such as `possession`, `last_touch`, `offside_player`, `foul` and `goal`, holds an index into this axis.

| Field | Shape after `[players]` | Holds |
| --- | --- | --- |
| `team` | | 0 left, 1 right, 2 an official, -1 for an empty entry. |
| `slot` | | The index in the team, or the official's number. |
| `present` | | GRF's `isActive`, 0 for a player who was sent off. |
| `role` | | A value of `ROLES`, or -1 for an official. |
| `cards` | 2 | The yellow cards shown and whether the player is sent off. |
| `controlled`, `designated` | | Whether an agent or a person controls the player, and whether the team designated the player. |
| `animation` | | GRF's `functionType`, a value of `ANIMATIONS`, what the body is doing. |
| `touch` | | 1 plus the kind of a touch of the ball within the last step, else 0. |
| `height` | | The player's height in meters. |
| `root_position`, `root_rotation` | 3, 4 | The root node of the skeleton in the world. |
| `joints`, `joint_rotations` | 13 by 3, 13 by 4 | The other nodes of the skeleton in the world, in the order of `JOINTS`. |

An entry that is not present keeps its last values, as the state holds them.

## Conventions

- Positions are in meters. The pitch lies between -55 and 55 along x and between -36 and 36 along y, with z up. The left team defends the goal at negative x and the right team the goal at positive x.
- Quaternions are stored as x, y, z, w, and a rotation is the same as its negative.
- Every position and rotation, of the ball, the roots and the joints, is in this one world frame, with nothing mirrored. GRF's saved state keeps the right team mirrored at its root, and the scene undoes that, so both teams stand on their own sides. The roots keep their world position, and the joints are world transforms, not offsets from the root.
- The scene reads the state, not GRF's render cache, so it does not depend on the render mode of the batch.
- Floats come straight from the engine in float32, with no quantization, and the CPU and the CUDA backends give the same bytes.

## Events

`follow` and `of_recording` return events as lists, one list for each environment, of `Event(step, kind, data)`. An event is the difference between two consecutive frames of the same episode, never a guess, and its step is the step of the later frame. A transition that happens inside a step shows at its end.

| Kind | When | Data |
| --- | --- | --- |
| `kickoff` | the game mode becomes a kick off | |
| `restart` | the referee stops play for a set piece | `set_piece` |
| `goal` | a team's score rises | `team`, `scorer` |
| `foul` | a foul, yellow card or red card is called | `kind`, `offender`, `victim` |
| `card` | a player's yellow or red card count rises | `player`, `kind`, 1 yellow or 2 red |
| `offside` | the referee calls an offside | `player` |
| `halftime` | the second half begins | |
| `fulltime` | the step reaches the duration | |

## Constants

The module holds GRF's constants in one place. `PITCH` and `FULL_PITCH` give half the length and half the width in meters, 55 by 36 inside the lines and 60 by 40 with the ground around them. `GOAL` gives half the width, the height and the depth, and `LINE_HALF_WIDTH` the line. `NODES` and `PARENTS` give the skeleton, `JOINTS` its joints, and `TEAMS`, `HALVES`, `ROLES`, `GAME_MODES`, `TOUCHES`, `FOULS`, `ANIMATIONS` and `CARDS` give the names of the values of the fields, by index. A test checks each against the engine's own enumerations and values.

## Versions

`scene.VERSION` and `Frames.version` give the version of the fields, which is 1. A new release adds fields under a new version and never changes the meaning of an existing field.
