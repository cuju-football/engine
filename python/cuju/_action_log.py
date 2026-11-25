# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Digests of a batch's outputs, and logs of a run's actions that replay it bit for bit."""

from __future__ import annotations

import ctypes
import hashlib
import importlib.metadata
import json
import zipfile
from typing import Any

import torch

from ._batch import Batch

__all__ = ['ActionLog', 'Digest', 'digest', 'replay']

_FORMAT = 1
_MACHINE_OPTIONS = (
    'device',
    'library',
    'threads',
    'pin_memory',
    'allocator',
    'pin_threads',
    'numa',
    'debug',
    'check_actions',
)
_REPLAY_OPTIONS = _MACHINE_OPTIONS[1:]


class Digest:
    """The SHA-256 of a sequence of outputs."""

    def __init__(self) -> None:
        self._hash = hashlib.sha256()

    def update(self, value: Any) -> None:
        """Adds a tensor, a dictionary or sequence of them, or bytes."""
        if isinstance(value, torch.Tensor):
            tensor = value.detach().to('cpu').contiguous()
            self._hash.update(
                ctypes.string_at(tensor.data_ptr(), tensor.numel() * tensor.element_size())
            )
        elif isinstance(value, dict):
            for key in sorted(value):
                self.update(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                self.update(item)
        elif isinstance(value, (bytes, bytearray)):
            self._hash.update(value)
        else:
            raise TypeError(f'expected tensors, dictionaries, sequences or bytes, found {value!r}')

    def copy(self) -> Digest:
        """Returns a digest of the same outputs, which goes on apart."""
        other = Digest()
        other._hash = self._hash.copy()
        return other

    def hexdigest(self) -> str:
        """Returns the SHA-256 so far in hexadecimal."""
        return self._hash.hexdigest()


def _states(batch, value):
    if not batch.stacked:
        for index in range(batch.environment_count):
            value.update(batch.get_state(index))
    return value


def _digest_actions(step: int, environment_count: int, agents: int, action_count: int) -> Any:
    values = [
        [(step * 7 + environment * 3 + agent * 5) % action_count for agent in range(agents)]
        for environment in range(environment_count)
    ]
    return torch.tensor(values, dtype=torch.int32)


def digest(
    assets: Any,
    device: str = 'cpu',
    steps: int = 100,
    seed: int = 7,
    environment_count: int = 4,
    **options: Any,
) -> str:
    """Plays a batch with fixed actions and returns the SHA-256 of its outputs."""
    batch = Batch(assets, environment_count=environment_count, device=device, seed=seed, **options)
    try:
        value = Digest()
        value.update(batch.reset())
        for step in range(steps):
            actions = _digest_actions(step, environment_count, batch.agents, batch.action_count)
            observation, reward, done, _ = batch.step(actions.to(batch._torch_device))
            value.update(observation)
            value.update(reward)
            value.update(done)
        return _states(batch, value).hexdigest()
    finally:
        batch.close()


def _plain(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


def _host_values(values):
    return None if values is None else _plain(torch.as_tensor(values))


class ActionLog:
    """A batch whose calls, actions and outputs are logged for `replay`."""

    def __init__(self, assets: Any, **options: Any) -> None:
        """Creates the batch."""
        self._assets = _plain(assets)
        self._options = {
            key: _plain(value) for key, value in options.items() if key not in _MACHINE_OPTIONS
        }
        json.dumps([self._assets, self._options])
        self.batch = Batch(assets, **options)
        self.calls: list[dict[str, Any]] = []
        self._actions: list[bytes] = []
        self._states: list[bytes] = []
        self._digest = Digest()
        self._digests: list[bytes] = []

    def _logged(self, call, *outputs):
        for output in outputs:
            self._digest.update(output)
        self.calls.append(call)
        self._digests.append(bytes.fromhex(self._digest.hexdigest()))

    def reset(self, seeds: Any = None, reverse: Any = None) -> Any:
        """Calls the batch's `reset` and logs it."""
        observation = self.batch.reset(seeds=seeds, reverse=reverse)
        call = {'call': 'reset', 'seeds': _host_values(seeds), 'reverse': _host_values(reverse)}
        self._logged(call, observation)
        return observation

    def step(self, actions: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
        """Calls the batch's `step` and logs it with its actions."""
        result = self.batch.step(actions)
        host = torch.as_tensor(actions).to('cpu', torch.int32).contiguous()
        self._actions.append(ctypes.string_at(host.data_ptr(), host.numel() * host.element_size()))
        self._logged({'call': 'step', 'shape': list(host.shape)}, *result[:3])
        return result

    def reset_done(self) -> None:
        """Calls the batch's `reset_done` and logs it."""
        self.batch.reset_done()
        self._logged({'call': 'reset_done'}, self.batch.observation())

    def set_state(self, index: int, state: bytes) -> None:
        """Calls the batch's `set_state` and logs it with the state."""
        self.batch.set_state(index, state)
        self._states.append(bytes(state))
        self._logged({'call': 'set_state', 'index': int(index)}, self.batch.observation())

    def hexdigest(self) -> str:
        """Returns the SHA-256 of every output so far and of the state of every environment."""
        return _states(self.batch, self._digest.copy()).hexdigest()

    def save(self, path: str) -> None:
        """Writes the log to a ZIP file at `path`, which `replay` reads."""
        try:
            version = importlib.metadata.version('cuju-engine')
        except importlib.metadata.PackageNotFoundError:
            version = None
        header = {
            'format': _FORMAT,
            'cuju': version,
            'assets': self._assets,
            'options': self._options,
            'device': self.batch.device,
            'calls': self.calls,
            'digest': self.hexdigest(),
        }
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('log.json', json.dumps(header, indent=1))
            archive.writestr('actions.bin', b''.join(self._actions))
            archive.writestr('digests.bin', b''.join(self._digests))
            for number, state in enumerate(self._states):
                archive.writestr(f'states/{number}.bin', state)

    def close(self) -> None:
        """Closes the batch."""
        self.batch.close()

    def __enter__(self) -> ActionLog:
        return self

    def __exit__(self, *exception: Any) -> None:
        self.close()


def replay(path: str, device: str | None = None, **options: Any) -> str:
    """Plays a saved `ActionLog` again and checks the digest after every call."""
    unknown = sorted(set(options) - set(_REPLAY_OPTIONS))
    if unknown:
        raise ValueError(
            f'expected only {", ".join(_REPLAY_OPTIONS)}, which change no output, found {unknown}'
        )
    with zipfile.ZipFile(path) as archive:
        header = json.loads(archive.read('log.json'))
        if header.get('format') != _FORMAT:
            raise ValueError(
                f'expected an action log of format {_FORMAT}, found {header.get("format")}'
            )
        actions = archive.read('actions.bin')
        digests = archive.read('digests.bin')
        states = [
            archive.read(f'states/{number}.bin')
            for number in range(sum(call['call'] == 'set_state' for call in header['calls']))
        ]
    batch_options = {
        key: value for key, value in header['options'].items() if key not in _MACHINE_OPTIONS
    }
    for key in ('channel_dimensions',):
        if key in batch_options:
            batch_options[key] = tuple(batch_options[key])
    for key in ('difficulty', 'agent_counts'):
        if batch_options.get(key) is not None:
            batch_options[key] = [tuple(pair) for pair in batch_options[key]]
    batch = Batch(header['assets'], device=device or header['device'], **batch_options, **options)
    try:
        value = Digest()
        offset = 0
        state = 0
        for number, call in enumerate(header['calls']):
            kind = call['call']
            if kind == 'reset':
                value.update(batch.reset(seeds=call['seeds'], reverse=call['reverse']))
            elif kind == 'step':
                count = 1
                for size in call['shape']:
                    count *= size
                step_actions = torch.frombuffer(
                    bytearray(actions[offset : offset + 4 * count]), dtype=torch.int32
                ).reshape(call['shape'])
                offset += 4 * count
                observation, reward, done, _ = batch.step(step_actions.to(batch._torch_device))
                value.update((observation, reward, done))
            elif kind == 'reset_done':
                batch.reset_done()
                value.update(batch.observation())
            elif kind == 'set_state':
                batch.set_state(call['index'], states[state])
                state += 1
                value.update(batch.observation())
            else:
                raise ValueError(f'expected the calls of an action log, found {kind!r}')
            expected = digests[32 * number : 32 * number + 32].hex()
            if value.hexdigest() != expected:
                raise ValueError(
                    f'the outputs of call {number} ({kind}) differ from the log: the digest is '
                    f'{value.hexdigest()}, the log holds {expected}'
                )
        result = _states(batch, value).hexdigest()
    finally:
        batch.close()
    if result != header['digest']:
        raise ValueError(
            f'the states after the last call differ from the log: the digest is {result}, the log '
            f'holds {header["digest"]}'
        )
    return result
