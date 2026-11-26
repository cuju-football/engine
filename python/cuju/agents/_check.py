# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""Checks that a controller plays, `check`, and lists the parts that a tree may name."""

import dataclasses
import time
from typing import Any

import numpy as np

from ..types import Device
from ._contract import Controller
from ._load import installed, load, node_of, standard

__all__ = ['Report', 'check', 'parts']


@dataclasses.dataclass
class Report:
    """What `check` found, with the raw timing of the calls and the actions taken."""

    scenario: str
    steps: int
    failures: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    timing: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    actions: dict[str, list[list[int]]] = dataclasses.field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether the controller played every match without a failure."""
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        """Returns the report as plain data."""
        return {
            'ok': self.ok,
            'scenario': self.scenario,
            'steps': self.steps,
            'failures': self.failures,
            'timing': self.timing,
            'actions': self.actions,
        }


class _Timed(Controller):
    """Times the calls of a controller, which the check reads."""

    def __init__(self, inner: Controller) -> None:
        self.inner = inner
        self.calls: list[float] = []
        self.load = 0.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    @property
    def representation(self) -> str:  # type: ignore[override]
        return self.inner.representation

    @property
    def control(self) -> dict[str, Any] | None:  # type: ignore[override]
        return getattr(self.inner, 'control', None)

    asynchronous = False

    def start(self, spec: Any) -> None:
        began = time.perf_counter()
        self.inner.start(spec)
        self.load = time.perf_counter() - began

    def act(self, observation: Any) -> Any:
        began = time.perf_counter()
        answer = self.inner.act(observation)
        self.calls.append(time.perf_counter() - began)
        return answer

    def close(self) -> None:
        self.inner.close()


def _wrap(spec: Any, process: Any, command: list[str] | None) -> Any:
    if command is not None:
        return {'entry': 'process', 'command': list(command), 'inner': {'entry': 'random'}}
    node = node_of(spec)
    if process and isinstance(node, dict) and node.get('entry') != 'process':
        wrapped: dict[str, Any] = {'entry': 'process', 'inner': node}
        if isinstance(process, str):
            wrapped['python'] = process
        return wrapped
    return node


def _play(spec: Any, side: int, scenario: str, steps: int, device: Device, report: Report) -> Any:
    from ._match import Match, _batch_for

    controller = None
    timed = None
    try:
        began = time.perf_counter()
        controller = load(spec)
        built = time.perf_counter() - began
        timed = _Timed(controller)
        opponent = load('random')
        pair = [timed, opponent] if side == 0 else [opponent, timed]
        batch = _batch_for(pair, scenario, 1, 0, device, {})
        try:
            with Match(batch, *pair, seed=0) as match:
                match.reset()
                moves = []
                for _ in range(steps):
                    match.step()
                    moves.append(match._applied.cpu().numpy().copy())
            return moves, timed, built
        finally:
            batch.close()
    except Exception as error:
        report.failures.append(
            {
                'side': ('left', 'right')[side],
                'reason': getattr(error, 'reason', type(error).__name__),
                'message': ' '.join(str(error).split()),
            }
        )
        return None, timed, 0.0
    finally:
        if controller is not None:
            controller.close()


def check(
    spec: Any,
    scenario: str,
    steps: int = 10,
    twice: bool = False,
    process: bool | str = False,
    command: list[str] | None = None,
    device: Device = 'cpu',
) -> Report:
    """Plays short matches with the controller on each side and reports what happened.

    The report holds each failure with its reason, the actions taken and the raw timing of the
    calls, the first, the minimum, the mean and the maximum. It judges no time.

    Args:
        spec: Anything that `load` takes, such as a folder, a file or an entry.
        scenario: The scenario to play.
        steps: The steps of each match.
        twice: Whether to play each side twice and compare the actions, which finds nondeterminism.
        process: Whether to run the controller through `client.py` in a new process, or the path
            of the interpreter to run it in, which finds hidden host dependencies.
        command: The command of a child that speaks the protocol itself, to check it.
        device: The device of the batch.
    """
    report = Report(scenario, steps)
    try:
        node = _wrap(spec, process, command)
    except Exception as error:
        report.failures.append({'side': None, 'reason': 'load', 'message': str(error)})
        return report
    for side in (0, 1):
        name = ('left', 'right')[side]
        runs = []
        for _ in range(2 if twice else 1):
            moves, timed, built = _play(node, side, scenario, steps, device, report)
            if moves is None:
                break
            runs.append(moves)
            calls = timed.calls if timed is not None else []
            report.timing.append(
                {
                    'side': name,
                    'load_seconds': built + (timed.load if timed else 0.0),
                    'calls': len(calls),
                    'first_seconds': calls[0] if calls else None,
                    'min_seconds': min(calls) if calls else None,
                    'mean_seconds': float(np.mean(calls)) if calls else None,
                    'max_seconds': max(calls) if calls else None,
                }
            )
        if runs:
            report.actions[name] = [step.reshape(-1).tolist() for step in runs[0]]
        if (
            twice
            and len(runs) == 2
            and not all(np.array_equal(a, b) for a, b in zip(runs[0], runs[1], strict=True))
        ):
            report.failures.append(
                {
                    'side': name,
                    'reason': 'nondeterministic',
                    'message': 'two runs with the same seed played different actions',
                }
            )
    return report


def parts() -> list[dict[str, str]]:
    """Returns the names that a tree may use, the standard controllers and the installed parts."""
    found = []
    for name, factory in sorted(standard().items()):
        summary = (factory.__doc__ or '').strip().splitlines()[0] if factory.__doc__ else ''
        found.append({'name': name, 'kind': 'standard', 'summary': summary})
    for name, entry in sorted(installed().items()):
        dist = entry.dist.name if entry.dist else entry.value
        found.append({'name': name, 'kind': 'installed', 'summary': f'from {dist}'})
    return found
