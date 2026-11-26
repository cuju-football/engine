# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""A PettingZoo parallel environment over a `Match` of one environment."""

from typing import Any

import torch

from ..errors import ConfigurationError

__all__ = ['to_pettingzoo']


def to_pettingzoo(match: Any) -> Any:
    """Returns a PettingZoo `ParallelEnv` over a match of one environment.

    The agents are the players that you left to yourself, named `left_0` and on, or `right_0`
    and on, each with its own observation, reward, action and spaces. An episode that ends
    empties `agents`, and `reset` starts the next one.
    """
    import gymnasium
    import pettingzoo

    if not match.yours:
        raise ConfigurationError('expected a side without a controller, which PettingZoo plays')
    if match.environments != 1:
        raise ConfigurationError(
            f'expected a batch of one environment for PettingZoo, found {match.environments}'
        )
    inner = match._inner
    single = inner.single_observation_space
    if single is None:
        raise ConfigurationError(
            "expected a batch with a vector or map representation for PettingZoo, found 'raw', "
            "so create the batch with representation='simple115v2' and raw=True"
        )
    names = []
    for side in match.yours:
        count = inner.left_agents[0] if side == 0 else inner.right_agents[0]
        names.extend(f'{("left", "right")[side]}_{index}' for index in range(int(count)))
    observation_space = gymnasium.spaces.Box(
        low=single.low.flat[0], high=single.high.flat[0], shape=single.shape[1:], dtype=single.dtype
    )
    action_space = gymnasium.spaces.Discrete(inner.action_count)

    class MatchParallelEnv(pettingzoo.ParallelEnv):
        """The parallel environment of one `Match`."""

        metadata = {'name': 'cuju_match_v0', 'is_parallelizable': True}  # noqa: RUF012

        def __init__(self) -> None:
            self.match = match
            self.possible_agents = list(names)
            self.agents: list[str] = []
            self.render_mode = None

        def observation_space(self, agent: str) -> Any:
            """Returns the space of an agent's observation."""
            return observation_space

        def action_space(self, agent: str) -> Any:
            """Returns the space of an agent's action."""
            return action_space

        def _observations(self, observation: Any) -> dict[str, Any]:
            return {
                agent: observation[0, index].detach().cpu().numpy()
                for index, agent in enumerate(self.possible_agents)
            }

        def reset(self, seed: int | None = None, options: Any = None) -> tuple[dict, dict]:
            """Starts an episode and returns the observations and the empty infos."""
            observation = self.match.reset(seeds=None if seed is None else [int(seed)])
            self.agents = list(self.possible_agents)
            return self._observations(observation), {agent: {} for agent in self.agents}

        def step(self, actions: dict[str, Any]) -> tuple[dict, dict, dict, dict, dict]:
            """Plays one step with an action for each agent and returns the five dictionaries."""
            missing = [agent for agent in self.agents if agent not in actions]
            if missing:
                raise ConfigurationError(f'expected an action for {", ".join(missing)}, found none')
            row = torch.tensor([[int(actions[agent]) for agent in self.possible_agents]])
            observation, reward, done, _ = self.match.step(row)
            ended = bool(done[0])
            length = self.match._length
            truncated = (
                ended and length is not None and int(self.match._finished_steps[0]) >= length
            )
            found = self._observations(observation)
            rewards = {
                agent: float(reward[0, index]) for index, agent in enumerate(self.possible_agents)
            }
            infos = {agent: {} for agent in self.possible_agents}
            flags = dict.fromkeys(self.possible_agents, ended and not truncated)
            cut = dict.fromkeys(self.possible_agents, truncated)
            if ended:
                self.agents = []
            return found, rewards, flags, cut, infos

        def close(self) -> None:
            """Leaves the match open, which the caller closes."""

    return MatchParallelEnv()
