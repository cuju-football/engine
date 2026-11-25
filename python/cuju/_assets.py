# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

import os
import re

_DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def bundled_assets() -> list[str]:
    directory = os.path.join(_DATA_DIRECTORY, 'assets')
    if not os.path.isdir(directory):
        return []
    return sorted(name[: -len('.cjb')] for name in os.listdir(directory) if name.endswith('.cjb'))


def _split_name(name):
    match = re.fullmatch(r'(.+)_l(\d+)_r(\d+)', name)
    return None if match is None else (match.group(1), int(match.group(2)), int(match.group(3)))


def scenario_names() -> list[str]:
    return sorted({split[0] for split in map(_split_name, bundled_assets()) if split})
