# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""The contract between controllers and the engine: the data they exchange and their error."""

from typing import Any

from ..errors import ConfigurationError, Error
from .client import Actions, Observation, Spec, as_actions

__all__ = ['Actions', 'Controller', 'ControllerError', 'Observation', 'Spec', 'as_actions']

ROLES = ('GK', 'CB', 'LB', 'RB', 'DM', 'CM', 'LM', 'RM', 'AM', 'CF')
ACTION_COUNTS = {'default': 19, 'v2': 20, 'full': 33}
SIDES = ('left', 'right')
_CONTROL_KEYS = ('players', 'assignment', 'selection', 'goalkeeper', 'action_set')


class ControllerError(Error, RuntimeError):
    """A breach of the contract: a wrong shape, a wrong action, a crash, a silence or a mismatch."""

    def __init__(
        self,
        message: str,
        reason: str = 'error',
        side: int | None = None,
        step: int | None = None,
        environment: int | None = None,
    ) -> None:
        """Names the breach, its reason and where it happened."""
        super().__init__(message)
        self.message = message
        self.reason = reason
        self.side = side
        self.step = step
        self.environment = environment

    def __str__(self) -> str:
        where = []
        if self.side is not None:
            where.append(f'side {SIDES[self.side]}')
        if self.step is not None:
            where.append(f'step {self.step}')
        if self.environment is not None:
            where.append(f'environment {self.environment}')
        return f'{self.message} ({", ".join(where)})' if where else self.message

    def __reduce__(self) -> Any:
        return (type(self), (self.message, self.reason, self.side, self.step, self.environment))


class Controller:
    """The interface: anything that sends actions for a set of agents, one environment or many."""

    # What the controller wants to observe, which it declares once.
    representation: str = 'raw'
    # The control that the controller asks for, or `None` for the defaults. See `normalize`.
    control: dict[str, Any] | None = None
    # Free data that the host passes through to `Spec.options`.
    options: dict[str, Any] | None = None
    # Whether the controller answers by `send` and `receive`, as a process does.
    asynchronous: bool = False

    def start(self, spec: Spec) -> None:
        """Starts a session, once for each setup of a batch."""

    def act(self, observation: Observation) -> Actions:
        """Returns the actions of the active agents of every environment."""
        raise NotImplementedError

    def close(self) -> None:
        """Ends the session and frees what the controller holds."""

    def send(self, observation: Observation) -> None:
        """Sends an observation without waiting, for a host that steps in real time."""
        raise NotImplementedError

    def receive(self, timeout: float) -> Actions | None:
        """Returns the answer to the last `send`, or `None` if none came in `timeout` seconds."""
        raise NotImplementedError


def normalize(control: Any) -> dict[str, Any]:
    """Returns a control table with its defaults, or raises if a key or a value is wrong."""
    control = {} if control is None else dict(control)
    unknown = sorted(set(control) - set(_CONTROL_KEYS))
    if unknown:
        raise ConfigurationError(
            f'expected the control keys {", ".join(_CONTROL_KEYS)}, found {", ".join(unknown)}'
        )
    found = {
        'players': control.get('players'),
        'assignment': control.get('assignment', 'dynamic'),
        'selection': list(control.get('selection', [])),
        'goalkeeper': control.get('goalkeeper', 'scenario'),
        'action_set': control.get('action_set', 'default'),
    }
    players = found['players']
    if players is not None and (
        isinstance(players, bool) or not isinstance(players, int) or players < 1
    ):
        raise ConfigurationError(
            f'expected players to be a whole number, 1 or more, found {players!r}'
        )
    for key, choices in (
        ('assignment', ('dynamic', 'fixed')),
        ('goalkeeper', ('agent', 'builtin', 'scenario')),
        ('action_set', ('default', 'v2', 'full')),
    ):
        if found[key] not in choices:
            raise ConfigurationError(
                f'expected {key} to be one of {", ".join(choices)}, found {found[key]!r}'
            )
    selection = found['selection']
    if found['assignment'] == 'dynamic' and selection:
        raise ConfigurationError('expected no selection with the dynamic assignment')
    if found['assignment'] == 'dynamic' and found['goalkeeper'] == 'builtin':
        raise ConfigurationError(
            "expected the goalkeeper 'agent' or 'scenario' with the dynamic assignment, found "
            "'builtin', which needs assignment = 'fixed'"
        )
    if found['assignment'] == 'fixed':
        if not selection:
            raise ConfigurationError('expected a selection with the fixed assignment, found none')
        indices = all(type(item) is int and 0 <= item <= 10 for item in selection)
        roles = all(isinstance(item, str) and item in ROLES for item in selection)
        if not (indices or roles) or (indices and len(set(selection)) != len(selection)):
            raise ConfigurationError(
                'expected a selection of distinct formation indices from 0 to 10, or of roles '
                f'from {", ".join(ROLES)}, found {selection!r}'
            )
        if players is not None and players != len(selection):
            raise ConfigurationError(
                f'expected as many players as the selection holds, {len(selection)}, found '
                f'{players}'
            )
        found['players'] = len(selection)
    return found
