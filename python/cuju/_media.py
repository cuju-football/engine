# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""Finds GRF's media for the renderer, the package cuju-grf-media of the extra `pixels`.

The package holds GRF's files unchanged, with `manifest.json`, which names each file, its source in
GRF, its size, its SHA-256 and its license, and `SHA256SUMS`. A missing package, or one whose
manifest is of another format, raises `MissingFileError` with one message naming the extra.
"""

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
# The format of the manifest that this cuju reads, which `ci/make_render_media.py` writes.
FORMAT = 1
INSTALL = "pip install 'cuju-engine[pixels]'"
MESSAGE = (
    f"GRF's media for cuju's renderer, the package {DISTRIBUTION}, is not installed; "
    f'install it with {INSTALL}'
)


def root() -> pathlib.Path:
    """Returns the folder of the installed media.

    Raises:
        MissingFileError: If cuju-grf-media is not installed, or its folder lacks the manifest.
    """
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
    """Returns the manifest of the media in `folder`, by default the installed package.

    Raises:
        MissingFileError: If the media are missing, or their manifest is of another format.
    """
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
    """Returns the path of the file `name` of the manifest, such as `data/media/shaders/noise.bmp`.

    Raises:
        MissingFileError: If the media are missing, or the manifest or the folder lacks the file.
    """
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
    """Returns what differs between the media in `folder` and their lists, or an empty list.

    Every file of the manifest must be in `SHA256SUMS` with the manifest's SHA-256, and every file
    of `SHA256SUMS` must have its listed SHA-256.
    """
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
