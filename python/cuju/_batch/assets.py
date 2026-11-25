# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

from .._assets import _DATA_DIRECTORY, _split_name, bundled_assets, scenario_names
from ..errors import MissingFileError


def _resolve_asset(asset: str) -> tuple[str, tuple[int, int]]:
    if callable(asset):
        raise TypeError(
            'expected the path of a scenario file or of a bundle, found a function; pass the path '
            'of its file, or build a bundle with cuju.scenarios.build_bundle'
        )
    # GRF: `create_environment` gives a scenario file or a bare name one left agent.
    if os.path.isfile(asset):
        return asset, (1, 0) if asset.endswith('.py') else (-1, -1)
    if asset.endswith(('.py', '.cjb')) or os.sep in asset or '/' in asset:
        raise MissingFileError(f'expected a scenario file or a bundle at {asset!r}, found no file')
    directory = os.path.join(_DATA_DIRECTORY, 'assets')
    shipped = os.path.join(directory, asset + '.cjb')
    if os.path.isfile(shipped):
        return shipped, (-1, -1)
    split = _split_name(asset)
    if split is None and asset in scenario_names():
        split = (asset, 1, 0)
    if split:
        scenario, left, right = split
        bases = [name for name in bundled_assets() if (_split_name(name) or ('',))[0] == scenario]
        if bases:
            base = min(bases, key=lambda name: sum(_split_name(name)[1:]))
            return os.path.join(directory, base + '.cjb'), (left, right)
    raise MissingFileError(
        f'expected the path of an asset bundle or `<scenario>_l<left>_r<right>` of one of '
        f'{scenario_names()}, found {asset!r}'
    )
