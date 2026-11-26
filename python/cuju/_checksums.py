# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
import hashlib
import os

DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
CHECKSUMS = os.path.join(DATA_DIRECTORY, 'SHA256SUMS')


def file_sha256(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def data_files(directory: str = DATA_DIRECTORY) -> list[str]:
    names = ['animations.bin'] if os.path.isfile(os.path.join(directory, 'animations.bin')) else []
    assets = os.path.join(directory, 'assets')
    if os.path.isdir(assets):
        names += sorted(f'assets/{name}' for name in os.listdir(assets) if name.endswith('.cjb'))
    return names


def data_checksums(directory: str = DATA_DIRECTORY) -> dict[str, str]:
    return {name: file_sha256(os.path.join(directory, name)) for name in data_files(directory)}


def format_checksums(checksums: dict[str, str]) -> str:
    return ''.join(f'{digest}  {name}\n' for name, digest in checksums.items())


def read_checksums(path: str = CHECKSUMS) -> dict[str, str]:
    checksums = {}
    with open(path, encoding='ascii') as handle:
        for number, line in enumerate(handle, 1):
            digest, separator, name = line.rstrip('\r\n').partition('  ')
            if not separator or len(digest) != 64 or digest.strip('0123456789abcdef') or not name:
                raise ValueError(f'expected `<SHA-256>  <path>` at line {number} of {path}')
            checksums[name] = digest
    return checksums
