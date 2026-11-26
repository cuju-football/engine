# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""A Gymnasium vector environment over a `Match`, with the caller's side as the agents."""

from typing import Any

import torch

from ..errors import ConfigurationError

__all__ = ['to_gymnasium']


def _spaces(match: Any, gymnasium: Any) -> tuple[Any, Any]:
    inner = match._inner
    columns = match._yours_columns()
    single = inner.single_observation_space
    if single is None:
        raise ConfigurationError(
            'expected a batch with a vector or map representation for Gymnasium, found '
            "'raw', so create the batch with representation='simple115v2' and raw=True"
        )
    shape = (len(columns), *single.shape[1:])
    observation = gymnasium.spaces.Box(
        low=single.low.flat[0], high=single.high.flat[0], shape=shape, dtype=single.dtype
    )
    action = gymnasium.spaces.MultiDiscrete([inner.action_count] * len(columns))
    return observation, action


def to_gymnasium(match: Any) -> Any:
    """Returns a Gymnasium vector environment over a match, with same-step auto-reset.

    The agents are the sides that you left to yourself. A step returns the observation after the
    reset when an episode ends, and `info['final_observation']` holds the last one of the
    environments that ended, marked by `info['_final_observation']`. `truncated` is true when the
    episode ended by the time limit, and `reward` sums the rewards of your agents.
    """
    import gymnasium

    if not match.yours:
        raise ConfigurationError('expected a side without a controller, which Gymnasium plays')
    if not match._inner.auto_reset:
        raise ConfigurationError('expected a batch with auto_reset=True for same-step reset')
    single_observation, single_action = _spaces(match, gymnasium)

    class MatchVectorEnv(gymnasium.vector.VectorEnv):
        """The vector environment of one `Match`."""

        def __init__(self) -> None:
            self.match = match
            self.num_envs = match.environments
            self.single_observation_space = single_observation
            self.single_action_space = single_action
            self.observation_space = gymnasium.vector.utils.batch_space(
                single_observation, self.num_envs
            )
            self.action_space = gymnasium.vector.utils.batch_space(single_action, self.num_envs)
            self.metadata = {}
            modes = getattr(gymnasium.vector, 'AutoresetMode', None)
            if modes is not None:
                self.metadata['autoreset_mode'] = modes.SAME_STEP
            self.render_mode = None
            self.closed = False

        def reset(self, *, seed: Any = None, options: Any = None) -> tuple[Any, dict[str, Any]]:
            """Starts every episode and returns the observations."""
            seeds = None
            if seed is not None:
                seeds = [int(seed) + index for index in range(self.num_envs)]
            return self.match.reset(seeds=seeds), {}

        def step(self, actions: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
            """Steps every environment, and resets those that ended."""
            observation, reward, done, information = self.match.step(actions)
            length = self.match._length
            ran = self.match._finished_steps
            truncated = done & (ran >= length) if length is not None else torch.zeros_like(done)
            terminated = done & ~truncated
            low, high = self.match._yours_columns()[0], self.match._yours_columns()[-1] + 1
            final = information['final_observation']
            final = final[:, low:high] if not isinstance(final, dict) else final
            info = {'final_observation': final, '_final_observation': done}
            return observation, reward.sum(1), terminated, truncated, info

        def close_extras(self, **kwargs: Any) -> None:
            """Leaves the match open, which the caller closes."""

    return MatchVectorEnv()
