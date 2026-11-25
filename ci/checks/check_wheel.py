# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks wheels of cuju: the license files, and the libraries each one holds."""

import hashlib
import json
import re
import sys
import zipfile

LICENSES = [
    'LICENSE',
    'NOTICE',
    'LICENSES/LGPL-2.1-or-later.txt',
    'LICENSES/MIT.txt',
    'LICENSES/MIT-gym.txt',
    'LICENSES/MIT-Mesa.txt',
    'LICENSES/Zlib.txt',
    'LICENSES/SunPro.txt',
    'LICENSES/BSL-1.0.txt',
    'LICENSES/BSD-3-Clause.txt',
]
CPU_LIBRARIES = ('libcuju_cpu.so', 'cuju_cpu.dll', 'libcuju_cpu.dylib')
CUDA_LIBRARIES = ('libcuju_cuda.so', 'cuju_cuda.dll')
RENDER_LIBRARIES = ('libcuju_render.so', 'cuju_render.dll', 'libcuju_render.dylib')
RENDER_LOADER = 'cuju/_pixels.py'
# The folders of the engine, so that a folder of any other package in the wheel is an error.
FOLDERS = ('_batch', 'data', 'gfootball', 'gfootball_engine')
ALIAS_FILES = (
    'cuju_gfootball/__init__.py',
    'cuju_gfootball/__main__.py',
    'cuju_gfootball/_startup.py',
    'cuju_gfootball/_version.py',
)
ALIAS_HOOK = 'cuju_gfootball.pth'
ALIAS_HOOK_TEXT = 'import cuju_gfootball._startup\n'

MEDIA_PACKAGE = 'cuju_grf_media/'
MEDIA_LICENSE = 'Apache-2.0 AND Unlicense AND OFL-1.1'
MEDIA_FORMAT = 1
# Each license file of the media wheel, with a line that its text must hold.
MEDIA_LICENSES = {
    'LICENSE': 'Apache License',
    'NOTICE': 'Bastiaan Konings',
    'LICENSES/Unlicense.txt': 'This is free and unencumbered software released into the public',
    'LICENSES/OFL-1.1.txt': 'Copyright 2013 The Alegreya Sans Project Authors',
}
MEDIA_OWN = ('__init__.py', 'manifest.json', 'SHA256SUMS')


def problems(path, cpu_only):
    """Returns what is wrong with the wheel of cuju at `path`, or an empty list."""
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
    found = [
        f'missing {name}'
        for name in LICENSES
        if not any(n.endswith(f'licenses/{name}') for n in names)
    ]
    portions = sorted(
        name
        for name in names
        if name.startswith('cuju/') and name.count('/') > 1 and name.split('/')[1] not in FOLDERS
    )
    if portions:
        found.append(f'expected no other package, found {portions[0]} and {len(portions) - 1} more')
    media = sorted(name for name in names if name.startswith(MEDIA_PACKAGE))
    if media:
        found.append(f'expected no media, found {media[0]} and {len(media) - 1} more')
    libraries = {name.rpartition('/')[2] for name in names if name.startswith('cuju/')}
    if not libraries & set(CPU_LIBRARIES):
        found.append('missing the CPU library')
    if RENDER_LOADER in names and not libraries & set(RENDER_LIBRARIES):
        found.append('missing the renderer library')
    cuda = sorted(libraries & set(CUDA_LIBRARIES))
    if cpu_only and cuda:
        found.append(f'expected no CUDA library, found {", ".join(cuda)}')
    if not cpu_only and not cuda:
        found.append('missing the CUDA library')
    return found


def alias_problems(path):
    """Returns what is wrong with the wheel of cuju-gfootball at `path`, or an empty list."""
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        hook = wheel.read(ALIAS_HOOK).decode() if ALIAS_HOOK in names else None
        metadata = ''.join(
            wheel.read(name).decode() for name in names if name.endswith('.dist-info/METADATA')
        )
    found = [f'missing {name}' for name in ALIAS_FILES if name not in names]
    if hook is None:
        found.append(f'missing {ALIAS_HOOK}')
    elif hook != ALIAS_HOOK_TEXT:
        found.append(f'expected {ALIAS_HOOK} to hold {ALIAS_HOOK_TEXT!r}, found {hook!r}')
    if not any(name.endswith('.dist-info/licenses/LICENSE') for name in names):
        found.append('missing LICENSE')
    others = sorted(
        name
        for name in names
        if name not in (*ALIAS_FILES, ALIAS_HOOK) and '.dist-info/' not in name
    )
    if others:
        found.append(f'expected the alias alone, found {others[0]} and {len(others) - 1} more')
    if not re.search(r'^Requires-Dist: cuju-engine[<>=!,.0-9]+$', metadata, re.MULTILINE):
        found.append('missing the dependency on a range of cuju')
    return found


def media_problems(path):
    """Returns what is wrong with the wheel of cuju-grf-media at `path`, or an empty list."""
    if not str(path).replace('\\', '/').rpartition('/')[2].endswith('-py3-none-any.whl'):
        return ['expected a py3-none-any wheel']
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        metadata = sorted(name for name in names if name.endswith('.dist-info/METADATA'))
        if len(metadata) != 1:
            return ['expected one METADATA']
        dist_info = metadata[0].rpartition('/')[0] + '/'
        lines = wheel.read(metadata[0]).decode('utf-8').splitlines()
        found = []
        if f'License-Expression: {MEDIA_LICENSE}' not in lines:
            found.append(f'expected the license expression {MEDIA_LICENSE}')
        for name, line in MEDIA_LICENSES.items():
            member = f'{dist_info}licenses/{name}'
            if member not in names:
                found.append(f'missing {name}')
            elif line not in wheel.read(member).decode('utf-8'):
                found.append(f'{name} lacks "{line}"')
        if f'{MEDIA_PACKAGE}manifest.json' not in names:
            return [*found, 'missing manifest.json']
        raw = wheel.read(f'{MEDIA_PACKAGE}manifest.json')
        manifest = json.loads(raw)
        if manifest.get('format') != MEDIA_FORMAT:
            found.append(f'expected the manifest format {MEDIA_FORMAT}')
        entries = manifest.get('files', []) + manifest.get('generated', [])
        for entry in entries:
            member = MEDIA_PACKAGE + entry['path']
            if entry.get('license') not in manifest.get('licenses', {}):
                found.append(f'{entry["path"]} has the license {entry.get("license")}')
            if member not in names:
                found.append(f'missing {entry["path"]}')
                continue
            data = wheel.read(member)
            if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
                found.append(f'{entry["path"]} differs from its manifest entry')
        expected = {entry['path']: entry['sha256'] for entry in entries}
        expected['manifest.json'] = hashlib.sha256(raw).hexdigest()
        if f'{MEDIA_PACKAGE}SHA256SUMS' not in names:
            found.append('missing SHA256SUMS')
        else:
            text = wheel.read(f'{MEDIA_PACKAGE}SHA256SUMS').decode('ascii')
            pairs = [line.partition('  ') for line in text.splitlines()]
            sums = {name: digest for digest, _, name in pairs}
            if sums != expected:
                found.append('SHA256SUMS differs from the manifest')
    allowed = {MEDIA_PACKAGE + name for name in [*expected, *MEDIA_OWN]}
    others = sorted(
        name for name in names if not name.startswith(dist_info) and name not in allowed
    )
    if others:
        found.append(
            f'expected only the listed files, found {others[0]} and {len(others) - 1} more'
        )
    return found


def main():
    arguments = sys.argv[1:]
    cpu_only = '--cpu-only' in arguments
    alias = '--alias' in arguments
    media = '--grf-media' in arguments
    flags = ('--cpu-only', '--alias', '--grf-media')
    wheels = [argument for argument in arguments if argument not in flags]
    if not wheels or cpu_only + alias + media > 1:
        sys.exit(
            'usage: python ci/checks/check_wheel.py [--cpu-only | --alias | --grf-media] <wheel>...'
        )
    failed = False
    for path in wheels:
        if alias:
            found = alias_problems(path)
        elif media:
            found = media_problems(path)
        else:
            found = problems(path, cpu_only)
        for problem in found:
            print(f'{path}: {problem}')
            failed = True
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
