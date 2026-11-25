# Scenarios

A scenario is the description of a match in Google Research Football, or GRF. It sets the teams, their players and roles, the ball, the length of an episode and the difficulty of each team. cuju ships the 19 scenarios of GRF, from full 11-a-side matches to short academy drills, and takes custom scenarios written in GRF's file format.

```python
import cuju

print(cuju.scenario_names())
batch = cuju.BatchEnvironment('academy_corner', environment_count=64, device='cpu', left_agents=3)
```

## Shipped scenarios

`cuju.scenario_names()` lists the names below. The arguments of the batch are in [API reference](api.md).

| Scenario | Players | Steps | Difficulty, left and right | What it is |
| --- | --- | --- | --- | --- |
| `11_vs_11_stochastic` | 11 v 11 | 3,001 | 1.0, 0.6 | A full match against the built-in AI |
| `11_vs_11_easy_stochastic` | 11 v 11 | 3,001 | 1.0, 0.05 | The same against a weak right team |
| `11_vs_11_hard_stochastic` | 11 v 11 | 3,001 | 1.0, 0.95 | The same against a strong right team |
| `11_vs_11_competition` | 11 v 11 | 3,001 | 1.0, 1.0 | Both teams at full strength |
| `11_vs_11_kaggle` | 11 v 11 | 3,001 | 1.0, 1.0 | The Kaggle competition match, with a second half from step 1,500 |
| `5_vs_5` | 5 v 5 | 3,001 | 0.05, 0.05 | A full five-a-side match |
| `1_vs_1_easy` | 1 v 1 | 501 | 0, 0 | One player against one |
| `academy_empty_goal_close` | 2 v 1 | up to 401 | 1.0, 0.6 | Score from close to an empty goal |
| `academy_empty_goal` | 2 v 1 | up to 401 | 1.0, 0.6 | Score into an empty goal from the middle of the pitch |
| `academy_run_to_score` | 2 v 6 | up to 401 | 1.0, 0.6 | Run to an empty goal with five defenders chasing |
| `academy_run_to_score_with_keeper` | 2 v 6 | up to 401 | 1.0, 0.6 | The same, with a goalkeeper |
| `academy_pass_and_shoot_with_keeper` | 3 v 2 | up to 401 | 1.0, 0.6 | Two attackers at the edge of the box, the one with the ball marked |
| `academy_run_pass_and_shoot_with_keeper` | 3 v 2 | up to 401 | 1.0, 0.6 | The same, with the one in the centre marked |
| `academy_3_vs_1_with_keeper` | 4 v 2 | up to 401 | 1.0, 0.6 | Three attackers against one defender and a goalkeeper |
| `academy_corner` | 11 v 11 | up to 401 | 1.0, 0.6 | Score from a corner |
| `academy_counterattack_easy` | 11 v 11 | up to 401 | 1.0, 0.6 | Four attackers against one defender |
| `academy_counterattack_hard` | 11 v 11 | up to 401 | 1.0, 0.6 | Four attackers against two defenders |
| `academy_single_goal_versus_lazy` | 11 v 11 | up to 3,001 | 1.0, 0.6 | Score once against outfield players who stand still and intercept only nearby balls |
| `test_example_multiagent` | 4 v 2 | up to 101 | 1.0, 0.6 | GRF's example of several agents, with the right goalkeeper placed in the left goal |

Players include the goalkeepers. Steps count an episode the way GRF does. Academies end early on a goal, when the ball leaves play, and in every academy but the corner when the other team gains possession. `test_example_multiagent` ends on the same three events.

In `11_vs_11_kaggle`, play restarts after step 1,500 with a kick-off by the other team, and the teams keep their ends. `5_vs_5` and the `11_vs_11` scenarios alternate the kick-off by the parity of the episode number, so the first episode of a batch plays the even form. The academies, `1_vs_1_easy` and `test_example_multiagent` play the same form every episode.

Academy episodes often end within tens of steps and the automatic reset starts the next one in the same step, so compare results with GRF by episodes rather than steps.

### Constant-action baselines

A single repeated action solves some drills. A policy that always shoots scores in 98% of the episodes of `academy_empty_goal_close` and 91% of those of `academy_empty_goal`, measured over 64 environments and 400 steps. A high scoring rate there says little about what a policy has learned, so weigh a result against a constant-action baseline measured on the same scenario. It costs a few seconds on the CPU. The same policy scored four goals in over 2,000 episodes of `academy_pass_and_shoot_with_keeper`, `academy_run_to_score_with_keeper` and `academy_3_vs_1_with_keeper`.

## Agents on each side

`scenario` is a plain name. `left_agents` and `right_agents` set how many players each team's agents control, from 0 to the players of the team, as GRF's `number_of_left_players_agent_controls` and `number_of_right_players_agent_controls` do. A name that carries agent counts is refused with a message naming the plain scenario, and more agents than players raise `cuju.errors.ConfigurationError`.

| Scenario | Arguments | Agents |
| --- | --- | --- |
| `5_vs_5` | the defaults | one on the left, as GRF's `create_environment` gives |
| `5_vs_5` | `left_agents=4` | four on the left against the built-in AI |
| `5_vs_5` | `left_agents=4, right_agents=4` | four on each side, for self-play |
| `5_vs_5` | `left_agents=0` | none, the built-in AI plays both teams |
| `11_vs_11_stochastic` | `left_agents=11` | a whole team of agents |

The agents of an environment come in a fixed order, left agents first. A right agent sees the pitch turned by 180 degrees, its actions turn the same way, and its rewards carry the sign of its own team, as [Observations, rewards and actions](observations-rewards-actions.md) describes.

With as many agents as controllable players, each agent keeps its player for the whole episode. With fewer, each agent controls the player GRF makes active for it, usually the one nearest the ball, which can be the goalkeeper in the 11-a-side scenarios. The built-in AI drives everyone else, as in GRF's single-agent benchmarks.

A scenario can mark players as not controllable. In `5_vs_5` both goalkeepers are, so four players per team can be controlled. `left_agents=5` there still gives five agents and the fifth holds its slot on the agent axis without a player. In the 11-a-side scenarios every player can be controlled.

One batch plays one scenario with one agent layout, so every output has a fixed shape. A curriculum across scenarios, or self-play beside play against the built-in AI, holds one batch for each and steps them in turn.

```python
selfplay = cuju.BatchEnvironment(
    '5_vs_5', environment_count=64, device='cpu', left_agents=4, right_agents=4
)
against_ai = cuju.BatchEnvironment('5_vs_5', environment_count=64, device='cpu', left_agents=4)
full_match = cuju.BatchEnvironment('11_vs_11_stochastic', environment_count=32, device='cpu')
print(selfplay.agents, against_ai.agents, full_match.agents)  # 8 4 1
```

## Difficulty

Each team has a difficulty from 0 to 1, which the scenario sets. `difficulty=(left, right)` overrides it for every environment of the batch, and `None` keeps the scenario's value.

```python
batch = cuju.BatchEnvironment(
    '5_vs_5', environment_count=64, device='cpu', left_agents=4, difficulty=(None, 0.95)
)
```

`set_difficulty(values, mask=None)` changes it from the next reset, which serves a curriculum that strengthens the opponents during training. `values` is a `(left, right)` pair, or `[environment_count, 2]` values with NaN where the scenario's value stays. `mask` selects the environments, so one batch can play several difficulties at once.

```python
import torch

batch.set_difficulty((None, 0.6), mask=torch.arange(64) < 32)
```

Difficulty belongs to the team, as in GRF. It scales the statistics of every player of the team by `0.3 + 0.7 * difficulty`, the players of agents included, and the players the built-in AI drives react up to 100 milliseconds later as it falls. Lowering the right difficulty weakens the opponents, and lowering the left difficulty weakens the agents' own players too. In `11_vs_11_stochastic` the right team's 0.6 scales its statistics by 0.72 against 1.0, so an even match sets `difficulty=(1.0, 1.0)`. `batch.difficulty` is a copy of the `[environment_count, 2]` difficulties in effect.

## Custom scenarios

A custom scenario is a Python file in GRF's format that defines `build_scenario(builder)`. It runs once, when the batch is created, and cuju builds it into a bundle without GRF installed. The builder computes what GRF's engine computes when it builds a match, so the batch plays GRF's episodes for that file. GRF's own scenario files build as they stand, and so do files written for GRF.

```python
# my_scenario.py
from . import *


def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().end_episode_on_score = True
    builder.SetBallPosition(0.6, 0.0)

    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(0.6, 0.0, e_PlayerRole_CF)

    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(-0.8, 0.1, e_PlayerRole_CB)
```

`from . import *` gives the file GRF's names for roles and teams. A file that names a package for them takes them from `cuju.gfootball.scenarios`, GRF's package over cuju's engine. The position given to `AddPlayer` is where the player starts. Its spot during play and at other set pieces comes from GRF's own 11-player formation by the player's order in the team, and the role stays the one the file gives.

Pass the path in place of a scenario name. A file gets one left agent by default, and `left_agents` and `right_agents` set others.

```python
batch = cuju.BatchEnvironment('my_scenario.py', environment_count=256, device='cpu', left_agents=2)
```

### Bundles

A build takes about a second. A batch keeps the bundle of a scenario file in a cache folder in the user's cache directory, so later batches in the same process or another load it at once. The key is the file's bytes, so an edited file builds again. `cuju.scenarios.cache_directory()` names the folder, which is safe to delete at any time. `cuju.scenarios.cached_bundle` uses the cache explicitly.

`cuju.scenarios.build_bundle` writes a bundle to a file of your choice, and a batch loads it like a shipped one.

```python
from cuju import scenarios

scenarios.build_bundle('my_scenario.py', 'my_scenario_l0_r0.cjb')
batch = cuju.BatchEnvironment(
    'my_scenario_l0_r0.cjb', environment_count=256, device='cpu', left_agents=1
)
```

The bundle holds the scenario without agents, and the batch sets up the agents that `left_agents` and `right_agents` ask for, whatever the file is called. A build is deterministic and both backends write the same bytes, so a bundle can be cited by its SHA-256. `name=` sets the name stored in it. `build_bundle` warns when a form lasts longer than a batch plays at GRF's default of 10 ticks a step, 5,119 steps.

A scenario file is Python and runs with the permissions of the user who builds it, as in GRF, so build only files you trust.

### What a file may do

The builder raises `cuju.scenarios.ScenarioError` with the reason for a scenario that GRF refuses or crashes on, or that a bundle cannot hold. A scenario builds when it meets these conditions.

| Condition | Reason |
| --- | --- |
| At most 11 players per team | GRF's engine crashes on a larger team |
| A goalkeeper as the first player of each team | GRF's engine hangs in open play without one |
| Values of the types GRF's bindings convert, such as numbers for coordinates | GRF raises on other types |
| A dependence on the episode number through its parity alone | A bundle holds one form for even episodes and one for odd. The builder runs episodes 1 to 8 and a pair at each power of ten from 100 to one billion, so a change only beyond those plays its early form |
| The same team sizes in every episode | A bundle holds one set of team sizes |
| Python's `random` module and NumPy's global generator left untouched | GRF draws each episode's seed from `random` right after the scenario, so a draw would shift every later seed |
| `controllable_left_players`, `controllable_right_players` and `dynamic_player_selection` unread while the scenario builds | GRF computes them at the first read, before it sets the agents |
| `reverse_team_processing` off in a deterministic scenario | A bundle holds no field for it |

A file runs only while its bundle is built, so anything it reads besides the episode parity, such as `os.environ` or another file, is read then and stays fixed for the batch. GRF runs the file at every reset, so a file that changes its scenario that way plays on as built. A deterministic scenario, with `builder.config().deterministic = True`, plays the seed it sets, 42 by default, at every episode, as in GRF.

A scenario that sets `control_all_players` plays when each side has either no agent or one per controllable player. GRF plays it through `MultiAgentToSingleAgent`, which `cuju.wrappers.MultiAgentToSingleAgent` ports, as [Migrating from GRF](migrating-from-grf.md) describes.

For variety in positions, write several files and give each its own batch. Each is built once and keeps the seeds of every episode as GRF draws them, which positions drawn from `random` inside one file would shift.

Every shipped scenario was built from GRF's own file, which you can start from in [GRF's repository](https://github.com/google-research/football/tree/3d9e754720a95621bba6475c4d3b0d56fe919014/gfootball/scenarios). The cost of each scenario is in [Performance](performance.md), and every known difference from GRF is in [Exactness](exactness.md).
