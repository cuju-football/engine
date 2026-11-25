# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""The asset bundles that the package ships, and the bundle that a name or path asks for."""

import os
import re

_DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def bundled_assets() -> list[str]:
    """Returns the names of the asset bundles that the package ships, such as `5_vs_5_l4_r0`.

    `BatchEnvironment` takes such a name `<scenario>_l<left agents>_r<right agents>` in place of a
    path, with any number of agents of a shipped scenario.
    """
    directory = os.path.join(_DATA_DIRECTORY, 'assets')
    if not os.path.isdir(directory):
        return []
    return sorted(name[: -len('.cjb')] for name in os.listdir(directory) if name.endswith('.cjb'))


def _split_name(name):
    """Returns the scenario, left and right agents of `<scenario>_l<left>_r<right>`, or `None`."""
    match = re.fullmatch(r'(.+)_l(\d+)_r(\d+)', name)
    return None if match is None else (match.group(1), int(match.group(2)), int(match.group(3)))


def bundled_scenarios() -> list[str]:
    """Returns the scenarios of the asset bundles that the package ships, such as `5_vs_5`."""
    return sorted({split[0] for split in map(_split_name, bundled_assets()) if split})
