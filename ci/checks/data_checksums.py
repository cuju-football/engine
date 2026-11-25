# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Writes `python/cuju/data/SHA256SUMS`, the SHA-256 of the shipped bundles and `animations.bin`."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))

from cuju import _checksums


def main():
    text = _checksums.format_checksums(_checksums.data_checksums())
    with open(_checksums.CHECKSUMS, 'w', encoding='ascii', newline='\n') as handle:
        handle.write(text)
    print(f'{len(text.splitlines())} files in {_checksums.CHECKSUMS}')


if __name__ == '__main__':
    main()
