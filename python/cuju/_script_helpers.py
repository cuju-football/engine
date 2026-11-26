# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch

from . import _replays, dumps

__all__ = ['ScriptHelpers']


def _grf_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.numpy()
    if isinstance(value, dict):
        return {key: _grf_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_grf_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_grf_value(item) for item in value)
    return value


def _grf_steps(steps):
    converted = [_grf_value(step) for step in steps]
    for step in converted:
        debug = step.get('debug')
        # GRF puts actions in the debug information.
        if isinstance(debug, dict) and 'action' in debug:
            debug['action'] = [dumps._CORE_ACTIONS[name] for name in debug['action']]
    return converted


def _dump_text(steps, include_debug):
    if not include_debug:
        for step in steps:
            step.pop('debug', None)
    return str(steps)


class ScriptHelpers:
    def load_dump(self, dump_file: str) -> list[dict[str, Any]]:
        return _grf_steps(dumps.read_dump(dump_file))

    def dump_to_txt(self, dump_file: str, output: str, include_debug: bool) -> None:
        text = _dump_text(self.load_dump(dump_file), include_debug)
        with open(output, 'w') as handle:
            handle.write(text)

    def replay(
        self,
        dump: str,
        fps: int = 10,
        config_update: dict[str, Any] | None = None,
        directory: str | None = None,
        render: bool = True,
    ) -> None:
        if render:
            raise ValueError('cuju renders nothing, so it replays with render=False only')
        steps = dumps.read_dump(dump)
        if not steps or 'config' not in steps[0].get('debug', {}):
            raise ValueError(f'expected a dump whose first step holds the configuration in {dump}')
        ticks = steps[0]['debug']['config']['physics_steps_per_frame']
        # GRF: `__modify_trace` adds idle steps unless the rate equals the dump's.
        if fps != ticks or 100 // fps != ticks:
            raise ValueError(
                f"at {fps} frames per second GRF's replay plays another match than the dump of "
                f'{ticks} ticks a step, and only 10 replays a dump of 10 ticks a step'
            )
        result = _replays.replay(dump, overrides=config_update)
        if result.difference is not None:
            raise ValueError(f'the replay differs from the dump at {result.difference}')
