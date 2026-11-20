# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Regenerates `python/cuju/gfootball/`, GRF's Python package with its import lines changed.

    python ci/port_gfootball.py --grf <checkout> [--gym <gym-0.21.0>]

`<checkout>` is a checkout of Google Research Football at commit 3d9e754, its release 2.10.3. The
script copies each file of GRF's package that `GRF_PATTERNS` names, changes the lines that
`GRF_REWRITES` lists and nothing else, adds the two lines of `GRF_NOTICE` below GRF's license
block, and writes the SHA-256 of each original to `GRF_SOURCES`. `<gym-0.21.0>` is the source of
gym 0.21.0, as its sdist holds it. With it, the script also writes the core of gym that
`cuju.gfootball._gym` vendors, from the files of `GYM_FILES` with the changes of `GYM_REWRITES`,
and their digests to `GYM_SOURCES`. Every other file of the tree is cuju's own, listed in
`OWN_FILES`, and stays as it is.

Each rewrite is invertible. `tests/test_gfootball_port.py` removes the added lines, undoes the
rewrites with the same tables and compares the digests, so the claim that the port is GRF's code
stays checked without GRF's sources.
"""

import argparse
import dataclasses
import fnmatch
import hashlib
import os
import re
import subprocess
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
TARGET = os.path.join(ROOT, 'python', 'cuju', 'gfootball')
GRF_COMMIT = '3d9e754720a95621bba6475c4d3b0d56fe919014'

# The files of GRF's package `gfootball/` that the port holds, as patterns of their paths.
GRF_PATTERNS = (
    '__init__.py',
    'play_game.py',
    'replay.py',
    'dump_to_txt.py',
    'dump_to_video.py',
    'env/*.py',
    'env/players/*.py',
    'scenarios/*.py',
    'scenarios/tests/*.py',
    'eval_server/*.py',
    'eval_server/proto/*.py',
    'eval_server/proto/*.proto',
    'examples/*.py',
)
# The files of gym 0.21.0's package `gym/` that `_gym/_vendored` holds.
GYM_FILES = (
    'core.py',
    'error.py',
    'logger.py',
    'version.py',
    'spaces/__init__.py',
    'spaces/space.py',
    'spaces/box.py',
    'spaces/dict.py',
    'spaces/discrete.py',
    'spaces/multi_binary.py',
    'spaces/multi_discrete.py',
    'spaces/tuple.py',
    'spaces/utils.py',
    'utils/__init__.py',
    'utils/closer.py',
    'utils/colorize.py',
    'utils/ezpickle.py',
    'utils/json_utils.py',
    'utils/seeding.py',
    'envs/registration.py',
    'wrappers/order_enforcing.py',
    'wrappers/time_limit.py',
)
VENDORED = '_gym/_vendored'
GRF_SOURCES = 'GRF_SOURCES'
GYM_SOURCES = f'{VENDORED}/GYM_SOURCES'
# The files of the tree that cuju wrote, which the script leaves as they are.
OWN_FILES = (
    '_cv2.py',
    '_gym/__init__.py',
    f'{VENDORED}/__init__.py',
    f'{VENDORED}/envs/__init__.py',
    f'{VENDORED}/wrappers/__init__.py',
)


@dataclasses.dataclass(frozen=True)
class Rewrite:
    """One invertible change, from `original` in the upstream file to `ported` in the port.

    A `line` rewrite matches a whole line after its indentation, a `prefix` rewrite the start of a
    line after its indentation, and a `literal` rewrite any occurrence within a line. The
    indentation and the rest of the line stay as they are.
    """

    kind: str
    original: str
    ported: str


# GRF's import lines and the strings that name its modules, each with its counterpart in cuju. The
# protobuf names `package=`, `full_name=` and the serialized descriptors stay GRF's, since they
# are the wire names of GRF's servers.
GRF_REWRITES = (
    Rewrite(
        'line', 'import gfootball_engine as libgame', 'from cuju import gfootball_engine as libgame'
    ),
    Rewrite('prefix', 'from gfootball_engine import ', 'from cuju.gfootball_engine import '),
    Rewrite(
        'line', 'import gfootball.env as football_env', 'import cuju.gfootball.env as football_env'
    ),
    Rewrite('line', 'import gfootball', 'from cuju import gfootball'),
    Rewrite('line', 'import gym', 'from cuju.gfootball import _gym as gym'),
    Rewrite(
        'line',
        'from gym.envs.registration import register',
        'from cuju.gfootball._gym import register',
    ),
    Rewrite('line', 'import cv2', 'from cuju.gfootball import _cv2 as cv2'),
    Rewrite('prefix', 'from gfootball.', 'from cuju.gfootball.'),
    Rewrite('literal', "'gfootball.scenarios.{}'", "'cuju.gfootball.scenarios.{}'"),
    Rewrite('literal', "'gfootball.env.players.{}'", "'cuju.gfootball.env.players.{}'"),
    Rewrite(
        'literal', "'gfootball.env:create_environment'", "'cuju.gfootball.env:create_environment'"
    ),
    Rewrite('literal', "'gfootball:GFootball-", "'cuju.gfootball:GFootball-"),
    Rewrite(
        'literal',
        "'__module__' : 'gfootball.eval_server.proto.",
        "'__module__' : 'cuju.gfootball.eval_server.proto.",
    ),
)
# gym's imports of itself, pointed inside the vendored package, and its plugin loader, which reads
# the entry points as Python 3.10 and later serve them.
GYM_REWRITES = (
    Rewrite('line', 'import gym', 'from cuju.gfootball._gym import _vendored as gym'),
    Rewrite('prefix', 'from gym import ', 'from cuju.gfootball._gym._vendored import '),
    Rewrite('prefix', 'from gym.', 'from cuju.gfootball._gym._vendored.'),
    Rewrite(
        'literal',
        'metadata.entry_points().get(entry_point, [])',
        'metadata.entry_points(group=entry_point)',
    ),
)

MODIFIED = (
    '# Modified by the cuju Authors in 2026, in the import lines that ci/port_gfootball.py lists.'
)
# The lines below GRF's license block of each Python file.
GRF_NOTICE = (MODIFIED, '# SPDX-License-Identifier: Apache-2.0')
GRF_LICENSE_END = '# limitations under the License.'
# The lines above each `.proto` file of GRF's, which carries no header of its own.
PROTO_NOTICE = ('// Copyright 2019 Google LLC', '// SPDX-License-Identifier: Apache-2.0')
# The lines above each file of gym's, whose files carry no header of their own.
GYM_NOTICE = (
    '# Copyright (c) 2016 OpenAI (https://openai.com)',
    '# Modified by the cuju Authors in 2026, in the lines that ci/port_gfootball.py lists.',
    '# SPDX-License-Identifier: MIT',
    '',
)
# An import that the tables leave pointing outside the port.
_ESCAPED_IMPORT = re.compile(r'^[ \t]*(import|from)[ \t]+(gfootball|gfootball_engine|gym|cv2)\b')


def _rewrite_line(line, rewrites, forward):
    """Applies the first matching line or prefix rewrite to `line`, then every literal one."""
    pairs = [
        (r.kind, r.original, r.ported) if forward else (r.kind, r.ported, r.original)
        for r in rewrites
    ]
    literals = [(source, target) for kind, source, target in pairs if kind == 'literal']
    if not forward:
        # The inverse undoes the literals first, as the forward pass applies them last.
        line = _replace_literals(line, literals)
    body = line.lstrip(' \t')
    indentation = line[: len(line) - len(body)]
    for kind, source, target in pairs:
        if kind == 'line' and body == source:
            line = indentation + target
            break
        if kind == 'prefix' and body.startswith(source):
            line = indentation + target + body[len(source) :]
            break
    if forward:
        line = _replace_literals(line, literals)
    return line


def _replace_literals(line, literals):
    for source, target in literals:
        line = line.replace(source, target)
    return line


def rewrite(text, rewrites, forward=True):
    """Returns `text` with each of its lines rewritten, forward or, with `forward=False`, back."""
    return '\n'.join(_rewrite_line(line, rewrites, forward) for line in text.split('\n'))


def hits(text, rewrites):
    """Returns the rewrites that change at least one line of `text`."""
    used = []
    for each in rewrites:
        lines = text.split('\n')
        if any(_rewrite_line(line, (each,), True) != line for line in lines):
            used.append(each)
    return used


def add_notice(text, path, notice_kind):
    """Returns `text` with its notice added, below GRF's license block or above gym's code."""
    lines = text.split('\n')
    if notice_kind == 'gym':
        return '\n'.join([*GYM_NOTICE, *lines])
    if path.endswith('.proto'):
        return '\n'.join([*PROTO_NOTICE, *lines])
    end = lines.index(GRF_LICENSE_END)
    return '\n'.join([*lines[: end + 1], *GRF_NOTICE, *lines[end + 1 :]])


def remove_notice(text, path, notice_kind):
    """Returns `text` without the lines that `add_notice` added, or raises `ValueError`."""
    lines = text.split('\n')
    if notice_kind == 'gym':
        start, notice = 0, GYM_NOTICE
    elif path.endswith('.proto'):
        start, notice = 0, PROTO_NOTICE
    else:
        start, notice = lines.index(GRF_LICENSE_END) + 1, GRF_NOTICE
    if tuple(lines[start : start + len(notice)]) != notice:
        raise ValueError(f'{path} lacks the added notice at line {start + 1}')
    return '\n'.join(lines[:start] + lines[start + len(notice) :])


def port(original, path, rewrites, notice_kind):
    """Returns the ported text of the upstream file `path`, whose text is `original`."""
    return add_notice(rewrite(original, rewrites), path, notice_kind)


def unport(ported, path, rewrites, notice_kind):
    """Returns the upstream text of the ported file `path`, inverting `port`."""
    return rewrite(remove_notice(ported, path, notice_kind), rewrites, forward=False)


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def format_sources(digests):
    """Returns the lines of `GRF_SOURCES` or `GYM_SOURCES`, as `sha256sum` writes them."""
    return ''.join(f'{digests[path]}  {path}\n' for path in sorted(digests))


def read_sources(path):
    """Returns the digests that the file `path` lists, by the path of each upstream file."""
    digests = {}
    with open(path, encoding='ascii') as handle:
        for number, line in enumerate(handle, 1):
            match = re.fullmatch(r'([0-9a-f]{64})  (\S+)\n', line)
            if match is None:
                raise ValueError(f'{path}: malformed line {number}')
            digests[match[2]] = match[1]
    return digests


def upstream_files(root, patterns):
    """Returns the paths under `root`, with forward slashes, that match one of `patterns`."""
    found = []
    for folder, folders, names in os.walk(root):
        folders[:] = sorted(name for name in folders if name != '__pycache__')
        for name in names:
            relative = os.path.relpath(os.path.join(folder, name), root).replace(os.sep, '/')
            if any(
                fnmatch.fnmatchcase(relative, pattern) and relative.count('/') == pattern.count('/')
                for pattern in patterns
            ):
                found.append(relative)
    return sorted(found)


def _read(path):
    with open(path, 'rb') as handle:
        return handle.read().decode('utf-8')


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as handle:
        handle.write(text.encode('utf-8'))


def _check_commit(checkout):
    """Exits when `checkout` is a git checkout at another commit than GRF's 2.10.3."""
    try:
        result = subprocess.run(
            ['git', '-c', 'safe.directory=*', '-C', checkout, 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        result = None
    if result is None or result.returncode != 0:
        print(f'note: the commit of {checkout} is unknown, expected {GRF_COMMIT[:7]}')
    elif result.stdout.strip() != GRF_COMMIT:
        sys.exit(f'{checkout} is at {result.stdout.strip()[:7]}, expected {GRF_COMMIT[:7]}')


def _port_tree(source, files, rewrites, notice_kind, prefix, usage):
    """Ports `files` of `source` into the tree, and returns their digests by upstream path.

    It adds the ported path of each file to `usage[rewrite]` for each rewrite that changes it.
    """
    digests = {}
    for path in files:
        original = _read(os.path.join(source, *path.split('/')))
        ported = port(original, path, rewrites, notice_kind)
        if unport(ported, path, rewrites, notice_kind) != original:
            sys.exit(f'{path}: the rewrites do not invert on this file')
        for number, line in enumerate(ported.split('\n'), 1):
            if _ESCAPED_IMPORT.match(line):
                sys.exit(f'{path}:{number}: an import that no rewrite covers: {line.strip()}')
        _write(os.path.join(TARGET, *f'{prefix}{path}'.split('/')), ported)
        digests[path] = digest(original)
        for each in hits(original, rewrites):
            usage.setdefault(each, []).append(f'{prefix}{path}')
    return digests


def _remove_strays(kept, keep_vendored):
    """Removes the files of the tree outside `kept`, and the folders left empty."""
    for folder, _, names in os.walk(TARGET, topdown=False):
        relative_folder = os.path.relpath(folder, TARGET).replace(os.sep, '/')
        if '__pycache__' in relative_folder.split('/'):
            continue
        for name in names:
            relative = os.path.normpath(os.path.join(relative_folder, name)).replace(os.sep, '/')
            if relative in kept or (keep_vendored and relative.startswith(f'{VENDORED}/')):
                continue
            os.remove(os.path.join(folder, name))
            print(f'removed {relative}')
        if folder != TARGET and not os.listdir(folder):
            os.rmdir(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--grf', required=True, help='a checkout of GRF at commit 3d9e754')
    parser.add_argument('--gym', help="gym 0.21.0's source, the folder that holds `gym/`")
    arguments = parser.parse_args()
    _check_commit(arguments.grf)
    usage = {}
    package = os.path.join(arguments.grf, 'gfootball')
    grf_files = upstream_files(package, GRF_PATTERNS)
    grf_digests = _port_tree(package, grf_files, GRF_REWRITES, 'grf', '', usage)
    _write(os.path.join(TARGET, GRF_SOURCES), format_sources(grf_digests))
    kept = {*grf_files, *OWN_FILES, GRF_SOURCES}
    rewrites = GRF_REWRITES
    if arguments.gym is not None:
        gym_package = os.path.join(arguments.gym, 'gym')
        gym_digests = _port_tree(gym_package, GYM_FILES, GYM_REWRITES, 'gym', f'{VENDORED}/', usage)
        _write(os.path.join(TARGET, *GYM_SOURCES.split('/')), format_sources(gym_digests))
        kept |= {f'{VENDORED}/{path}' for path in GYM_FILES} | {GYM_SOURCES}
        rewrites = (*GRF_REWRITES, *GYM_REWRITES)
    _remove_strays(kept, keep_vendored=arguments.gym is None)
    for path in OWN_FILES:
        if not os.path.exists(os.path.join(TARGET, *path.split('/'))):
            print(f'note: the own file {path} is missing')
    for each in rewrites:
        files = usage.get(each, [])
        print(f'{each.kind} {each.original!r} -> {each.ported!r}, {len(files)} files')
        for path in files:
            print(f'    {path}')
        if not files:
            sys.exit('a rewrite that changes no file, so the table is out of date')
    print(f'{len(grf_files)} files of GRF in {TARGET}')


if __name__ == '__main__':
    main()
