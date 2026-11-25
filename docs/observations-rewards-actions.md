# Observations, rewards and actions

Every step of a batch returns one observation and one reward per agent and takes one action per agent. All three follow Google Research Football, or GRF, under GRF's names and with GRF's values, and every output keeps the agent axis. The constructor options named here are listed in [the API reference](api.md).

```python
import cuju
import torch

batch = cuju.BatchEnvironment('5_vs_5', environment_count=16, device='cpu', left_agents=4)
observation = batch.reset()  # [16, 4, 115] float32
actions = torch.zeros((16, 4), dtype=torch.int32)  # 0 is idle
observation, reward, done, information = batch.step(actions)  # reward [16, 4], done [16] bool
```

## Representations

The option `representation` selects the observation. The default is `simple115v2`, where GRF's `create_environment` defaults to `extracted`.

| Representation | Shape | Type | Content |
| --- | --- | --- | --- |
| `simple115` | `[environments, agents, 115]` | `float32` | GRF's 115 numbers, in its first layout |
| `simple115v2` | `[environments, agents, 115]` | `float32` | the same numbers with a fixed slot for each player |
| `extracted` | `[environments, agents, height, width, 4]` | `uint8` | the super mini map, four planes on a grid |
| `raw` | dictionary of `[environments, agents, ...]` | by key | every value of GRF's observation, by name |
| `pixels` | `[environments, agents, height, width, 3]` | `uint8` | GRF's rendered frame, shrunk to `channel_dimensions` |
| `pixels_gray` | `[environments, agents, height, width, 1]` | `uint8` | the same frame in gray |

`channel_dimensions=(width, height)` sets the size of `extracted`, `pixels` and `pixels_gray`, 96 by 72 by default. With `stacked=True` the last four observations of an agent join along the last axis, oldest first, as in GRF. A reset fills all four places with its first observation. Stacking takes `simple115`, `simple115v2` and `extracted`, and raises `cuju.errors.ConfigurationError` with `raw` and the pixel representations.

A vector takes 460 bytes per agent and the map 27,648, which is 60 times more. A batch of 16,384 environments with four agents holds 3.6 GB of maps, and four times that when stacked. A smaller `environment_count` or `channel_dimensions` fits a GPU with little memory.

`simple115v2` suits a multilayer perceptron. It holds positions, directions, the ball, the active player and the game mode. Tiredness, cards, roles, the score, the time left and the sticky actions appear only in `raw`. With `raw=True` the batch writes the raw arrays beside any representation, and `batch.raw()` returns them for the last step. `information['final_raw']` holds them for the step that ended each episode. Writing them adds to the cost of every step.

```python
batch = cuju.BatchEnvironment('5_vs_5', environment_count=4, device='cpu', left_agents=4, raw=True)
observation = batch.reset()
print(batch.raw()['ball_owned_team'].shape)  # torch.Size([4, 4])
```

Every output is a view of the batch's buffers, and the next call overwrites it. Clone a tensor that has to outlive the next `step`, `reset`, `set_state`, `raw()` or `frame()`.

The spaces `single_observation_space` and `single_action_space` describe one environment, and `observation_space` and `action_space` the whole batch. They need `pip install 'cuju-engine[spaces]'`. The spaces of `raw` are `None`, as in GRF.

### Agent axis

Observations, rewards and actions keep the agent axis for any number of agents. One agent in 8 environments gives observations of `[8, 1, 115]` and rewards of `[8, 1]`, and `step` takes `[8, 1]` or `[8]`. The environments of `cuju.env` drop the axis for one agent as GRF does, as [Migrating from GRF](migrating-from-grf.md) shows.

When a team has as many agents as controllable players, each agent keeps one player for the whole episode. With fewer agents, the agents switch players during play as in GRF, and `raw['active']` changes. In `5_vs_5` the goalkeepers belong to the built-in AI, so four agents a side control players 1 to 4. An agent can have no player, with `raw['active'] == -1` and an all-zero active one-hot. That happens to the last agents of a team with more agents than controllable players, and to the agent of a player who was sent off. Multiply a loss by `raw['active'] != -1` to count only the agents that play.

`batch.left_players` and `batch.right_players` give the number of players of each team.

### simple115v2 layout

| Indices | Numbers | Content |
| --- | --- | --- |
| 0 to 21 | 22 | positions of the agent's team, `x` then `y` of 11 slots |
| 22 to 43 | 22 | directions of the agent's team |
| 44 to 65 | 22 | positions of the other team |
| 66 to 87 | 22 | directions of the other team |
| 88 to 90 | 3 | position of the ball, `x`, `y`, `z` |
| 91 to 93 | 3 | direction of the ball |
| 94 to 96 | 3 | owner of the ball, one-hot over none, the agent's team, the other team |
| 97 to 107 | 11 | active player, one-hot, all zero without one |
| 108 to 114 | 7 | game mode, one-hot over the values of `game_mode` |

Player `k` has its position at indices `2k` and `2k + 1`. Unused slots hold -1. In a stack, frame `k` adds `115 * k` to every index.

`simple115` writes the four player arrays one after the other and pads the 88 player numbers with -1 at the end, so in `5_vs_5` the directions start at index 10. The ball, owner, active player and game mode stay at 88 to 114 in both layouts.

### Super mini map

`extracted` draws four planes of `height` by `width` cells, for the agent's team, the other team, the ball and the active player. A cell holds 255 where something stands and 0 elsewhere. The map is drawn on the device with the rest of the observation.

### Raw keys

All keys are `[environments, agents, ...]`, and the table gives the shape after the agent axis.

| Key | Shape | Type | Values |
| --- | --- | --- | --- |
| `ball`, `ball_direction`, `ball_rotation` | `[3]` | `float32` | `z` in meters |
| `ball_owned_team` | none | `int32` | -1 none, 0 the agent's team, 1 the other |
| `ball_owned_player` | none | `int32` | index in the owner's arrays, -1 none |
| `left_team`, `left_team_direction` | `[players, 2]` | `float32` | positions and directions |
| `left_team_tired_factor` | `[players]` | `float32` | 0 when fresh |
| `left_team_yellow_card`, `left_team_active` | `[players]` | `bool` | active is false for a player sent off |
| `left_team_roles` | `[players]` | `int32` | 0 to 9, listed below |
| the six keys above for `right_team` | as for `left_team` | as for `left_team` | |
| `score` | `[2]` | `int32` | own goals first |
| `steps_left` | none | `int32` | steps until the match time ends |
| `game_mode` | none | `int32` | 0 normal, 1 kick-off, 2 goal kick, 3 free kick, 4 corner, 5 throw-in, 6 penalty |
| `active` | none | `int32` | the player the agent controls, -1 none |
| `designated` | none | `int32` | the player a single agent of the team would control |
| `sticky_actions` | `[10]` or `[13]` | `uint8` | one flag per sticky action |

The roles are 0 goalkeeper, 1 center back, 2 left back, 3 right back, 4 defensive midfielder, 5 central midfielder, 6 left midfielder, 7 right midfielder, 8 attacking midfielder and 9 center forward.

```python
batch = cuju.BatchEnvironment(
    '5_vs_5', environment_count=4, device='cpu', left_agents=1, right_agents=1, representation='raw'
)
raw = batch.reset()
print(raw['left_team'].shape)  # torch.Size([4, 2, 5, 2]), 4 environments, 2 agents, 5 players
```

### Pitch coordinates

GRF divides `x` by 54.4 and `y` by -83.6 for the observation. The ball's `z` stays in meters. The pitch is 110 m by 72 m, so the lines fall slightly beyond GRF's rounded figures.

| Feature | Meters | Raw units |
| --- | --- | --- |
| goal line | `x` = ±55 | ±1.0110 |
| touchline | `y` = ±36 | ±0.4306 |
| half the width of the goal | 3.7 | 0.0443 |
| crossbar, the ball's `z` | 2.5 | 2.5 |

`y` grows toward the bottom touchline, so a plot with `y` upward appears upside down. One unit of `x` is 54.4 m and one of `y` is 83.6 m, so multiply by those factors before computing a distance in meters. Clipping observations to the interval from -1 to 1 cuts off the goal lines and the players behind them. Normalize with the bounds of the pitch instead.

### What a right agent sees

Every agent observes its own team as the left team, defending the goal at `x = -1.011`. For a right agent the pitch turns by 180 degrees, as in GRF.

| Key | For a right agent |
| --- | --- |
| `ball`, `ball_direction` | `x` and `y` change sign, `z` stays |
| `ball_rotation` | unturned, as in GRF |
| `ball_owned_team` | 0 is always the agent's own team |
| `left_team*` | the right team's arrays, with `x` and `y` of positions and directions negated |
| `right_team*` | the left team's arrays, turned the same way |
| `score` | own goals first |
| `sticky_actions` | the direction flags turn with the pitch |
| `game_mode`, `steps_left` | the same for both sides |

The vectors and the map follow the same rules, and the actions of a right agent turn back, so one network can play either side. A scenario file gives each team's positions in that team's own view.

## Rewards

The option `rewards` names the rewards as in GRF.

| `rewards` | Reward of an agent at each step |
| --- | --- |
| `scoring` | 1 when its team scores, -1 when the other team scores, 0 otherwise |
| `scoring,checkpoints` | the scoring reward plus the checkpoint reward |

The reward is `[environments, agents]` of `float32`. A batch needs `scoring`, so `rewards='checkpoints'` alone raises `cuju.errors.ConfigurationError`. An unknown part draws a warning.

The checkpoint reward divides the distance from the ball to the opponent's goal into 10 zones, with thresholds from 0.99 down to 0.19 in raw units. Each time an agent's active player owns the ball in a zone it has not reached, the agent gets 0.1 for that zone and any zone it skipped. A goal pays the zones that remain, so a scoring step pays at most 2.0. The count of zones resets with the episode.

`information['score_reward']` is `[environments]` of `int32`, the scoring signal from the left team's point of view, before the checkpoints. The own scoring reward of each agent follows from signing it by side.

```python
is_right = torch.arange(batch.agents) >= batch.left_agents  # right agents come last
sign = torch.where(is_right, -1, 1).to(reward.device)
own_score = information['score_reward'][:, None] * sign  # [environments, agents]
```

With equal numbers of agents on both sides and no checkpoints, the rewards of an environment sum to zero at every goal. With other counts a goal sums to the difference of the counts.

GRF adds the checkpoint reward with numpy, and numpy 2 changes the last bits of a `float32` sum. A batch adds in the precision of numpy 1. The environments of `cuju.env` follow the installed numpy.

## Actions

An action is an integer from 0 to `action_count - 1`, and 0 is `idle`. `step` takes `[environments, agents]`, or `[environments]` for one agent. The option `action_set` selects the set and `batch.action_count` gives its size.

| `action_set` | Actions | Contents |
| --- | --- | --- |
| `default` | 19 | GRF's `action_set_v1`, used by GRF's papers and Kaggle |
| `v2` | 20 | the 19, then `builtin_ai` |
| `full` | 33 | the 20, then 13 more |

| Index | Action | Index | Action |
| --- | --- | --- | --- |
| 0 | `idle` | 10 | `high_pass` |
| 1 | `left` | 11 | `short_pass` |
| 2 | `top_left` | 12 | `shot` |
| 3 | `top` | 13 | `sprint` |
| 4 | `top_right` | 14 | `release_direction` |
| 5 | `right` | 15 | `release_sprint` |
| 6 | `bottom_right` | 16 | `sliding` |
| 7 | `bottom` | 17 | `dribble` |
| 8 | `bottom_left` | 18 | `release_dribble` |
| 9 | `long_pass` | 19 | `builtin_ai`, in `v2` and `full` |

From index 20, `full` adds `keeper_rush`, `pressure`, `team_pressure`, `switch`, `release_long_pass`, `release_high_pass`, `release_short_pass`, `release_shot`, `release_keeper_rush`, `release_sliding`, `release_pressure`, `release_team_pressure` and `release_switch`.

`builtin_ai` lets the built-in AI play the agent's player for one step, which suits curricula and partly learned teams. A side whose agents all play it still has its agents, observations and rewards.

### Sticky actions

The eight directions and `sprint` and `dribble` stay pressed once taken, as on a gamepad, and in `full` so do `keeper_rush`, `pressure` and `team_pressure`. A direction holds until `release_direction` or another direction, the other buttons hold until their own release action, and `idle` leaves everything pressed. Directions turn with the side, so `left` points toward the agent's own goal and `right` toward the opponent's.

`raw['sticky_actions']` holds one flag per sticky action.

| Index | Sticky action |
| --- | --- |
| 0 to 3 | `left`, `top_left`, `top`, `top_right` |
| 4 to 7 | `right`, `bottom_right`, `bottom`, `bottom_left` |
| 8 | `sprint` |
| 9 | `dribble` |
| 10 to 12 | `keeper_rush`, `pressure`, `team_pressure`, in `full` only |

### Range checks

A floating-point action raises `TypeError`. On the CPU, and on a GPU batch with `debug = true` in the `[run]` table of the machine config, an index outside the range raises `cuju.errors.ArgumentError` that names the action, its agent and its environment. The check waits for the device at every step, so keep it for new policies and action pipelines. With `debug` off, a GPU batch plays an out of range index as `idle`. Actions from an `argmax` over `action_count` logits are always in range. [The API reference](api.md) describes the config.

The environments of `cuju.env` also take GRF's negative indices, from `-action_count` to -1, and turn them into the same positive actions.

## Frames and videos

A batch draws GRF's frames with its own software renderer, which matches GRF's rendering byte for byte. The pixel representations return the frame shrunk to `channel_dimensions`, and `frames=True` adds the full frames with any representation. `batch.frame()` returns them as `[environments, height, width, 3]` of `uint8`, rows from the top, for the last step. The frame has the size `frame_size=(width, height)`, 1280 by 720 by default. The width must be a multiple of 4, each side is at most 16,384, and `frame_size=800` alone gives 800 by 720 as in GRF. All agents of an environment share one frame.

Drawing takes a few hundred milliseconds of processor time per frame, so [Performance](performance.md) covers the cost. Calling `frame()` without `frames=True` or a pixel representation raises `cuju.errors.LibraryError`. Stacking pixel observations is left to the training code, which copies each observation before the next step.

The renderer reads GRF's models and textures from the package cuju-grf-media, which `pip install 'cuju-engine[pixels]'` installs. Videos of a batch come from its frames, and videos of recorded matches come from GRF's dumps through `cuju.gfootball`, as [Migrating from GRF](migrating-from-grf.md) shows.

```python
import cv2
import torch

import cuju

batch = cuju.BatchEnvironment('11_vs_11_stochastic', environment_count=4, device='cpu', frames=True)
batch.reset()
video = cv2.VideoWriter('match.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (1280, 720))
for _ in range(300):
    batch.step(torch.zeros((4, 1), dtype=torch.int32))
    video.write(cv2.cvtColor(batch.frame()[0].cpu().numpy(), cv2.COLOR_RGB2BGR))
video.release()
```

A batch holds one frame per step, where GRF's own video also holds the frames of the internal steps of a stoppage. Matches played with frames on are those of a rendering GRF, which differ from the matches without frames.
