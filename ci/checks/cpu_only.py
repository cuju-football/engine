# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Turns pyproject.toml into the CPU-only distribution `cuju-engine-cpu`."""

import re
import sys

# Each pattern must match exactly once, so a change of pyproject.toml can't slip past silently.
REPLACEMENTS = [
    (r'(?m)^name = "cuju-engine"$', 'name = "cuju-engine-cpu"'),
    (r'(?m)^(description = ".*), for CUDA and CPU"$', r'\1, for the CPU only"'),
    (r'(?m)^    "Environment :: GPU :: NVIDIA CUDA :: 12",\n', ''),
    (r'(?m)^CMAKE_CUDA_ARCHITECTURES = .*$', 'CUJU_CUDA = "OFF"'),
]


def cpu_only(text):
    """Returns the text of pyproject.toml for `cuju-engine-cpu`."""
    for pattern, replacement in REPLACEMENTS:
        text, count = re.subn(pattern, replacement, text)
        if count != 1:
            raise ValueError(f'expected one match of {pattern!r} in pyproject.toml, found {count}')
    return text


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'pyproject.toml'
    with open(path, encoding='utf-8') as handle:
        text = handle.read()
    with open(path, 'w', encoding='utf-8', newline='\n') as handle:
        handle.write(cpu_only(text))


if __name__ == '__main__':
    main()
