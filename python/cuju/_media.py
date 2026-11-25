# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.resources
import json
import os
import pathlib
from typing import Any

from . import _checksums
from .errors import MissingFileError

DISTRIBUTION = 'cuju-grf-media'
PACKAGE = 'cuju_grf_media'
FORMAT = 1
INSTALL = "pip install 'cuju-engine[pixels]'"
MESSAGE = (
    f"GRF's media for cuju's renderer, the package {DISTRIBUTION}, is not installed; "
    f'install it with {INSTALL}'
)


def root() -> pathlib.Path:
    try:
        folder = importlib.resources.files(PACKAGE)
    except ModuleNotFoundError:
        raise MissingFileError(MESSAGE) from None
    path = pathlib.Path(str(folder))
    if not (path / 'manifest.json').is_file():
        raise MissingFileError(
            f'{DISTRIBUTION} at {path} lacks its manifest.json; reinstall it with {INSTALL}'
        )
    return path


def manifest(folder: str | os.PathLike | None = None) -> dict[str, Any]:
    path = pathlib.Path(folder) if folder is not None else root()
    try:
        with open(path / 'manifest.json', encoding='ascii') as handle:
            content = json.load(handle)
    except FileNotFoundError:
        raise MissingFileError(f'{path} holds no manifest.json of {DISTRIBUTION}') from None
    if content.get('format') != FORMAT:
        raise MissingFileError(
            f'{DISTRIBUTION} {content.get("version")} at {path} holds media of format '
            f'{content.get("format")}, and this cuju reads format {FORMAT}; install the version '
            f'that fits with {INSTALL}'
        )
    return content


def path(name: str, folder: str | os.PathLike | None = None) -> pathlib.Path:
    base = pathlib.Path(folder) if folder is not None else root()
    content = manifest(base)
    listed = {entry['path'] for entry in content['files'] + content['generated']}
    if name not in listed:
        raise MissingFileError(f'the manifest of {DISTRIBUTION} at {base} lists no {name}')
    file = base.joinpath(*name.split('/'))
    if not file.is_file():
        raise MissingFileError(
            f'{DISTRIBUTION} at {base} lacks {name}; reinstall it with {INSTALL}'
        )
    return file


def problems(folder: str | os.PathLike | None = None) -> list[str]:
    base = pathlib.Path(folder) if folder is not None else root()
    content = manifest(base)
    try:
        listed = _checksums.read_checksums(str(base / 'SHA256SUMS'))
    except (FileNotFoundError, ValueError) as error:
        return [str(error)]
    found = []
    for entry in content['files'] + content['generated']:
        if listed.get(entry['path']) != entry['sha256']:
            found.append(f'SHA256SUMS and the manifest disagree on {entry["path"]}')
    for name, digest in listed.items():
        file = base.joinpath(*name.split('/'))
        if not file.is_file():
            found.append(f'missing {name}')
        elif _checksums.file_sha256(file) != digest:
            found.append(f'{name} differs from its SHA-256 in SHA256SUMS')
    return found
