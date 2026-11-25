# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""GRF's `MultiAgentToSingleAgent` wrapper, over every environment of a batch."""

from __future__ import annotations

from typing import Any

import torch

from .dumps import _CORE_ACTIONS

__all__ = ['MultiAgentToSingleAgent']

_ACTION_COUNTS = {'default': 19, 'v2': 20, 'full': 33}
# GRF: `football_action_set.action_builtin_ai`, index 19 of `v2` and `full`.
_BUILTIN_AI = 19


def _select(value, index):
    rows = torch.arange(index.shape[0], device=index.device)[:, None]
    if isinstance(value, dict):
        return {name: tensor[rows, index] for name, tensor in value.items()}
    return value[rows, index]


def _broadcast(flags, like):
    return flags.view((-1,) + (1,) * (like.dim() - 1))


class MultiAgentToSingleAgent:
    """GRF's `MultiAgentToSingleAgent` over a batch whose scenario sets `control_all_players`."""

    def __init__(
        self,
        batch: Any,
        left_players: int,
        right_players: int,
        action_set: str | None = None,
        stacked: bool = False,
        squeeze: bool = False,
        _slots: list[int] | None = None,
    ) -> None:
        """Wraps a batch."""
        if left_players not in (0, 1) or right_players not in (0, 1):
            raise ValueError(
                f'expected 0 or 1 left and right players, found {left_players} and {right_players}'
            )
        if batch.action_count < _ACTION_COUNTS['v2']:
            raise ValueError(
                'expected a batch whose action set holds builtin_ai; create it with '
                "action_set='v2', which plays GRF's default set with the same indices, or 'full'"
            )
        if batch.stacked:
            raise ValueError('expected a batch without a stack; pass stacked=True here instead')
        if getattr(batch, '_raw_fields', None) is None:
            raise ValueError('expected a batch with the raw representation or with_raw=True')
        if stacked and batch._representation == 'raw':
            raise ValueError('expected simple115, simple115v2 or extracted for a stack, found raw')
        covered = torch.ones(batch.agents, dtype=torch.bool)
        if _slots is not None:
            covered = torch.zeros(batch.agents, dtype=torch.bool)
            covered[list(_slots)] = True
        left = torch.arange(batch.agents) < batch.left_agents[:, None]
        mask = batch.agent_mask & covered
        side_masks = [left & mask, ~left & mask]
        for side, players, agents in (
            ('left', left_players, side_masks[0].sum(1)),
            ('right', right_players, side_masks[1].sum(1)),
        ):
            if bool(((agents > 0) != bool(players)).any()):
                raise ValueError(
                    f'expected {"agents" if players else "no agent"} on the {side} side of every '
                    f'environment, since the policy {"plays" if players else "does not play"} it'
                )
        self.players = left_players + right_players
        if squeeze and self.players != 1:
            raise ValueError('expected one side to squeeze the axis of the sides')
        if action_set is None:
            self.action_count = batch.action_count
        else:
            if action_set not in _ACTION_COUNTS:
                raise KeyError(f'expected the action set {", ".join(map(repr, _ACTION_COUNTS))}')
            self.action_count = _ACTION_COUNTS[action_set]
            if self.action_count > batch.action_count:
                raise ValueError(
                    f'expected a batch with the actions of {action_set!r}, found '
                    f'{batch.action_count}'
                )
        self.batch = batch
        self._sides = [side for side, players in ((0, left_players), (1, right_players)) if players]
        self._stacked = stacked
        self._squeeze = squeeze
        device = batch._torch_device
        self._side_masks = [side_mask.to(device) for side_mask in side_masks]
        self._index = None
        self._frames = None
        self._done = None
        self.last_actions = None

    def _selection(self, raw, only=None):
        match = raw['active'] == raw['designated']
        columns = []
        for side in self._sides:
            found = match & self._side_masks[side]
            wrong = found.sum(1) != 1
            if only is not None:
                wrong &= only
            if bool(wrong.any()):
                environment = int(torch.nonzero(wrong)[0, 0])
                raise RuntimeError(
                    f'environment {environment} has {int(found[environment].sum())} agents on the '
                    f'{("left", "right")[side]} side whose active player is the designated one, '
                    'where the wrapper needs exactly one'
                )
            columns.append(found.int().argmax(1))
        return torch.stack(columns, 1)

    def _shape(self, observation):
        if not self._squeeze or isinstance(observation, dict):
            # GRF: the raw observation of one agent keeps its agent axis.
            return observation
        return observation.squeeze(1)

    def _current(self):
        return _select(self.batch.observation(), self._index)

    def observation(self) -> Any:
        """Returns the policy's observation of the last `reset` or `step`."""
        if self._index is None:
            raise RuntimeError('reset must be called before the observation')
        observation = torch.cat(self._frames, dim=-1) if self._stacked else self._current()
        return self._shape(observation)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        """Resets the batch, with its arguments, and returns the policy's observation."""
        self.batch.reset(*args, **kwargs)
        self._index = self._selection(self.batch.raw())
        if self._stacked:
            self._frames = [self._current()] * 4
        self._done = None
        return self.observation()

    def _inner_actions(self, actions):
        count = self.action_count
        environments = self.batch.environment_count
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)
        if actions.is_floating_point() or actions.is_complex():
            raise TypeError(f'expected integer action indices, found {actions.dtype}')
        if self.players == 1 and tuple(actions.shape) == (environments,):
            actions = actions[:, None]
        if tuple(actions.shape) != (environments, self.players):
            raise ValueError(
                f'expected actions of the shape {(environments, self.players)}, found '
                f'{tuple(actions.shape)}'
            )
        # GRF: an action indexes the policy's set, so one outside it raises.
        if actions.device.type == 'cpu' and bool(((actions < -count) | (actions >= count)).any()):
            raise IndexError(f'expected actions from {-count} to {count - 1}')
        actions = actions.to(self._index.device, torch.int64)
        actions = torch.where(actions < 0, actions + count, actions)
        inner = torch.full(
            (environments, self.batch.agents), _BUILTIN_AI, dtype=torch.int64, device=actions.device
        )
        inner.scatter_(1, self._index, actions)
        return inner.to(torch.int32)

    def step(
        self, actions: Any, *, _others: list[tuple[int, int]] | None = None
    ) -> tuple[Any, Any, Any, dict[str, Any]]:
        """Runs one step of the batch."""
        if self._index is None:
            raise RuntimeError('reset must be called before step')
        self.last_actions = self._inner_actions(actions)
        for slot, action in _others or []:
            self.last_actions[:, slot] = action
        _, reward, done, information = self.batch.step(self.last_actions)
        ended = done.bool().to(self._index.device)
        final = _select(
            information['final_observation'], self._selection(self.batch.final_raw(), ended)
        )
        self._index = self._selection(self.batch.raw())
        if self._stacked:
            current = self._current()
            frames = [*self._frames[1:], torch.where(_broadcast(ended, current), final, current)]
            final = torch.cat(frames, dim=-1)
            if self.batch.auto_reset:
                frames = [torch.where(_broadcast(ended, current), current, f) for f in frames]
            self._frames = frames
        self._done = ended
        if self._squeeze:
            reward = reward[:, 0]
        return (
            self.observation(),
            reward,
            done,
            {**information, 'final_observation': self._shape(final)},
        )

    def reset_done(self) -> None:
        """Resets the environments whose episode ended, as the batch's `reset_done`."""
        if self._index is None:
            raise RuntimeError('reset must be called before reset_done')
        self.batch.reset_done()
        self._index = self._selection(self.batch.raw())
        if self._stacked and self._done is not None:
            current = self._current()
            ended = _broadcast(self._done, current)
            self._frames = [torch.where(ended, current, frame) for frame in self._frames]
        self._done = None

    def close(self) -> None:
        """Closes the batch."""
        self.batch.close()

    @staticmethod
    def get_observation(observation: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Returns the raw observations of one environment whose active player is designated."""
        assert 'designated' in observation[0], 'Only raw observations can be converted'
        return [entry for entry in observation if entry['designated'] == entry['active']]

    @staticmethod
    def get_action(actions: Any, original_observation: list[dict[str, Any]]) -> list[Any]:
        """Returns an action for each raw observation of `original_observation`."""

        assert 'designated' in original_observation[0], 'Only raw observations can be converted'
        result = [_CORE_ACTIONS['builtin_ai']] * len(original_observation)
        action_index = 0
        for index, entry in enumerate(original_observation):
            if entry['designated'] == entry['active']:
                assert action_index < len(actions)
                result[index] = actions[action_index]
                action_index += 1
        return result
