# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import io
import json
import os
import zipfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ._errors import RecordingError

FORMAT = 'cuju-recording'
VERSION = 1
HEADER = 'header.json'
# A fixed time makes the same recording the same bytes.
_TIME = (1980, 1, 1, 0, 0, 0)


@contextlib.contextmanager
def opened(source: Any) -> Iterator[Any]:
    if isinstance(source, (bytes, bytearray, memoryview)):
        yield io.BytesIO(source)
    elif hasattr(source, 'read'):
        yield source if getattr(source, 'seekable', lambda: False)() else io.BytesIO(source.read())
    else:
        with open(os.fspath(source), 'rb') as handle:
            yield handle


def read_bytes(source: Any) -> bytes:
    with opened(source) as handle:
        return handle.read()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f'{type(value).__name__} is not JSON data')


def dumps(header: Mapping[str, Any]) -> bytes:
    return json.dumps(header, default=_json_default).encode()


def _checked(header: Any) -> dict[str, Any]:
    if not isinstance(header, dict):
        raise RecordingError('expected a header that is an object, found something else')
    if header.get('format') != FORMAT:
        raise RecordingError(f'expected the format {FORMAT!r}, found {header.get("format")!r}')
    version = header.get('version')
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise RecordingError(f'expected a version that is a whole number, found {version!r}')
    if version > VERSION:
        raise RecordingError(
            f'expected version {VERSION} or older, found version {version}, '
            'which a newer cuju-recording wrote'
        )
    return header


def read_archive(source: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    arrays: dict[str, np.ndarray] = {}
    try:
        with opened(source) as handle, zipfile.ZipFile(handle) as archive:
            names = archive.namelist()
            if HEADER not in names:
                raise RecordingError(f'expected the member {HEADER}, found none')
            header = _checked(json.loads(archive.read(HEADER)))
            for name in names:
                if name.endswith('.npy'):
                    with archive.open(name) as member:
                        arrays[name[:-4]] = np.lib.format.read_array(member, allow_pickle=False)
    except RecordingError:
        raise
    except (zipfile.BadZipFile, ValueError, EOFError) as error:
        raise RecordingError(f'expected a recording, found {error}') from error
    return header, arrays


def write_archive(header: Mapping[str, Any], arrays: Mapping[str, np.ndarray], target: Any) -> Any:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(_member(HEADER), dumps(header))
        for name, array in arrays.items():
            data = io.BytesIO()
            np.lib.format.write_array(data, array, allow_pickle=False)
            archive.writestr(_member(f'{name}.npy'), data.getvalue())
    data = buffer.getvalue()
    if target is None:
        return data
    if hasattr(target, 'write'):
        target.write(data)
    else:
        path = Path(os.fspath(target))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return None


def _member(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, _TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info
