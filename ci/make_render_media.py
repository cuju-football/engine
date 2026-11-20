# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Copies the media of GRF that cuju's renderer reads into the package cuju-grf-media, unchanged.

    python ci/make_render_media.py <GRF tree> [--generated <directory>]

The GRF tree is a checkout of Google Research Football at `GRF_COMMIT`. Every file that
`packaging/cuju-grf-media/sources.txt` lists must have its pinned SHA-256 there, as must GRF's two
license files, or nothing is written. The files are copied byte for byte into
`packaging/cuju-grf-media/src/cuju_grf_media`, with `manifest.json`, which names each file, its
source, size, SHA-256, license and role, and `SHA256SUMS`, which `sha256sum -c` reads. The license
texts come from `LICENSE` and `LICENSES/` of this repository, which must hold GRF's texts.

Files that other ci scripts produce, such as a glyph atlas of the font, are listed in `GENERATED`
and copied from `--generated`, at the same relative path, with their producer in the manifest.
Then `python -m build --wheel packaging/cuju-grf-media` builds the wheel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys

GRF_REPOSITORY = 'https://github.com/google-research/football'
GRF_COMMIT = '3d9e754720a95621bba6475c4d3b0d56fe919014'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT = os.path.join(ROOT, 'packaging', 'cuju-grf-media')
SOURCES = os.path.join(PROJECT, 'sources.txt')
# The format of the manifest, which `cuju._media.FORMAT` must equal.
FORMAT = 1
# The package folder of each source folder of GRF.
FOLDERS = {
    'data/': 'third_party/gfootball_engine/data/',
    'fonts/': 'third_party/fonts/',
}
# The text of each license, the same path in this repository and in the project of the wheel.
LICENSE_TEXTS = {
    'Apache-2.0': 'LICENSE',
    'Unlicense': 'LICENSES/Unlicense.txt',
    'OFL-1.1': 'LICENSES/OFL-1.1.txt',
}
LICENSES = tuple(LICENSE_TEXTS)
ROLES = ('scene', 'model', 'texture', 'adboard', 'pitch', 'overlay', 'font')
# GRF's license files, which the repository's copies must match.
GRF_UNLICENSE = (
    'third_party/gfootball_engine/LICENSE',
    '6b0382b16279f26ff69014300541967a356a666eb0b91b422f6862f6b7dad17e',
)
GRF_OFL = (
    'third_party/fonts/LICENSE',
    '0e46be800f72df6aa97e43c8c2cddb953c5ca9321c5a397ba853651e074200ac',
)
# GRF's copy of the OFL opens with this line where an OFL.txt has the font's copyright, which is
# name 0 of the font's name table.
OFL_HEADER = 'See each family directory OFL.txt file for the Copyright information.\n\n'
FONT_COPYRIGHT = (
    'Copyright 2013 The Alegreya Sans Project Authors '
    '(https://github.com/huertatipografica/Alegreya-Sans)\n\n'
)
# Files that other ci scripts produce, each as (path, license, role, producer). The converter
# copies each from `--generated` and fails if one is missing.
GENERATED: tuple[tuple[str, str, str, str], ...] = (
    ('fonts/caption_glyphs.bin', 'OFL-1.1', 'font', 'ci/make_glyph_atlas.py'),
)
# The SHA-256 of each generated file as the release builds it, on Ubuntu 24.04 with SDL2_ttf
# 2.22.0, FreeType 2.13.2 and HarfBuzz 8.3.0. A file that differs is refused.
GENERATED_SHA256 = {
    'fonts/caption_glyphs.bin': '6cb41d66d34c36fa14c08c7c7634b2f17063dfa43492303ea4cad4f89ee17dfd',
}
# What the converter writes, by path in the project, and removes before it writes.
OUTPUTS = (
    'LICENSE',
    'LICENSES',
    'src/cuju_grf_media/data',
    'src/cuju_grf_media/fonts',
    'src/cuju_grf_media/manifest.json',
    'src/cuju_grf_media/SHA256SUMS',
)


class ConversionError(Exception):
    """A GRF tree, a license or a generated file that the converter refuses."""


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_sources(path: str = SOURCES) -> list[dict]:
    """Returns the pinned files of `sources.txt`, each with its path, SHA-256, license and role."""
    entries = []
    with open(path, encoding='ascii') as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith('#'):
                continue
            fields = line.split()
            if (
                len(fields) != 4
                or not re.fullmatch('[0-9a-f]{64}', fields[0])
                or fields[1] not in LICENSES
                or fields[2] not in ROLES
            ):
                raise ConversionError(
                    f'expected `<SHA-256> <license> <role> <path>` at line {number}'
                )
            entries.append(
                {'path': fields[3], 'sha256': fields[0], 'license': fields[1], 'role': fields[2]}
            )
    paths = [entry['path'] for entry in entries]
    if len(set(paths)) != len(paths):
        raise ConversionError(f'{path} lists a file twice')
    return entries


def source_path(path: str) -> str:
    """Returns the path in GRF's repository of a path of the package."""
    for folder, source in FOLDERS.items():
        if path.startswith(folder):
            return source + path[len(folder) :]
    raise ConversionError(f'{path} lies outside {", ".join(FOLDERS)}')


def tree_problems(tree: str, entries: list[dict]) -> list[str]:
    """Returns every pinned file that is missing from the GRF tree or differs from its pin."""
    pins = [(source_path(entry['path']), entry['sha256']) for entry in entries]
    problems = []
    for source, digest in [*pins, GRF_UNLICENSE, GRF_OFL]:
        path = os.path.join(tree, *source.split('/'))
        if not os.path.isfile(path):
            problems.append(f'missing {source}')
        elif file_sha256(path) != digest:
            problems.append(f'{source} differs from the pinned SHA-256 {digest}')
    return problems


def read_bytes(path: str) -> bytes:
    with open(path, 'rb') as handle:
        return handle.read()


def license_problems(tree: str, root: str = ROOT) -> list[str]:
    """Returns where the repository's license texts differ from GRF's."""
    problems = []
    unlicense = read_bytes(os.path.join(tree, *GRF_UNLICENSE[0].split('/')))
    if read_bytes(os.path.join(root, 'LICENSES', 'Unlicense.txt')) != unlicense:
        problems.append(f"LICENSES/Unlicense.txt differs from GRF's {GRF_UNLICENSE[0]}")
    ofl = read_bytes(os.path.join(tree, *GRF_OFL[0].split('/'))).decode('ascii')
    if not ofl.startswith(OFL_HEADER):
        problems.append(f"GRF's {GRF_OFL[0]} lacks its first line")
    expected = (FONT_COPYRIGHT + ofl[len(OFL_HEADER) :]).encode('ascii')
    if read_bytes(os.path.join(root, 'LICENSES', 'OFL-1.1.txt')) != expected:
        problems.append(
            f"LICENSES/OFL-1.1.txt is not the font's copyright line and GRF's {GRF_OFL[0]}"
        )
    return problems


def package_version(project: str) -> str:
    with open(os.path.join(project, 'src', 'cuju_grf_media', '__init__.py'), encoding='utf-8') as f:
        match = re.search(r"^__version__ = '([^']+)'$", f.read(), re.MULTILINE)
    if not match:
        raise ConversionError('expected `__version__` in cuju_grf_media/__init__.py')
    return match.group(1)


def copy(source: str, destination: str) -> None:
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.copyfile(source, destination)


def convert(
    tree: str,
    generated: str | None = None,
    project: str = PROJECT,
    root: str = ROOT,
    sources: str = SOURCES,
    generated_files: tuple[tuple[str, str, str, str], ...] = GENERATED,
) -> dict:
    """Checks the GRF tree, writes the package's files, and returns the manifest.

    Raises:
        ConversionError: If a pinned file is missing or differs, a license text differs from GRF's,
            or a generated file is missing or differs from its pin.
    """
    entries = read_sources(sources)
    problems = tree_problems(tree, entries) + license_problems(tree, root)
    for path, *_ in generated_files:
        made_path = os.path.join(generated or '', *path.split('/'))
        if generated is None or not os.path.isfile(made_path):
            problems.append(f'missing the generated file {path}')
        elif path in GENERATED_SHA256 and file_sha256(made_path) != GENERATED_SHA256[path]:
            problems.append(
                f'the generated file {path} has the SHA-256 {file_sha256(made_path)}, pinned '
                f'{GENERATED_SHA256[path]}'
            )
    if problems:
        raise ConversionError(
            f'the GRF tree {tree} is not GRF {GRF_COMMIT} as pinned, or its licenses differ. '
            'A checkout with converted line endings differs too.\n  ' + '\n  '.join(problems)
        )
    package = os.path.join(project, 'src', 'cuju_grf_media')
    for output in OUTPUTS:
        path = os.path.join(project, *output.split('/'))
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)
    files = []
    for entry in entries:
        source = source_path(entry['path'])
        destination = os.path.join(package, *entry['path'].split('/'))
        copy(os.path.join(tree, *source.split('/')), destination)
        files.append(
            {
                'path': entry['path'],
                'source': source,
                'bytes': os.path.getsize(destination),
                'sha256': entry['sha256'],
                'license': entry['license'],
                'role': entry['role'],
            }
        )
    made = []
    for path, license_name, role, producer in generated_files:
        destination = os.path.join(package, *path.split('/'))
        copy(os.path.join(generated, *path.split('/')), destination)
        made.append(
            {
                'path': path,
                'producer': producer,
                'bytes': os.path.getsize(destination),
                'sha256': file_sha256(destination),
                'license': license_name,
                'role': role,
            }
        )
    for name in LICENSE_TEXTS.values():
        copy(os.path.join(root, *name.split('/')), os.path.join(project, *name.split('/')))
    manifest = {
        'format': FORMAT,
        'package': 'cuju-grf-media',
        'version': package_version(project),
        'source': {'repository': GRF_REPOSITORY, 'commit': GRF_COMMIT},
        'licenses': LICENSE_TEXTS,
        # GRF sorts the boards by name and picks each one by its index, so the order is part of
        # every frame.
        'adboards': sorted(entry['path'] for entry in files if entry['role'] == 'adboard'),
        'files': files,
        'generated': made,
    }
    with open(os.path.join(package, 'manifest.json'), 'w', encoding='ascii', newline='\n') as f:
        json.dump(manifest, f, indent=1)
        f.write('\n')
    listed = ['manifest.json'] + [entry['path'] for entry in files + made]
    with open(os.path.join(package, 'SHA256SUMS'), 'w', encoding='ascii', newline='\n') as f:
        for name in sorted(listed):
            f.write(f'{file_sha256(os.path.join(package, *name.split("/")))}  {name}\n')
    return manifest


def main(arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('tree', help=f'a checkout of {GRF_REPOSITORY} at {GRF_COMMIT}')
    parser.add_argument('--generated', help='the folder of the files that GENERATED lists')
    options = parser.parse_args(arguments)
    try:
        manifest = convert(options.tree, options.generated)
    except ConversionError as error:
        sys.exit(str(error))
    entries = manifest['files'] + manifest['generated']
    total = sum(entry['bytes'] for entry in entries)
    print(f'{len(entries)} files, {total} bytes, in {PROJECT}')


if __name__ == '__main__':
    main()
