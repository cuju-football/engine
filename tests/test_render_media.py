# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks GRF's media for the renderer: the locator, the converter, the pinned list and the wheel.

The converter runs on a small tree laid out as GRF's, so these checks need no GRF checkout. The
checks of the installed package run where cuju-grf-media is installed.
"""

import hashlib
import importlib.util
import json
import os
import re
import zipfile

import pytest

import cuju
from cuju import _media

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTALL = "pip install 'cuju-engine[pixels]'"


def source_file(*parts):
    """Returns the path of a file of the source tree, or skips where the tree isn't at hand."""
    path = os.path.join(ROOT, *parts)
    if not os.path.isfile(path):
        pytest.skip(f'needs the source tree, which lacks {"/".join(parts)}')
    return path


def load_script(*parts):
    """Imports a script of the source tree as a module."""
    path = source_file(*parts)
    specification = importlib.util.spec_from_file_location(parts[-1][: -len('.py')], path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


# A tree laid out as GRF's, with one file of each kind, and the bytes of each.
FAKE_FILES = {
    'data/media/objects/balls/generic.object': (b'<object></object>\n', 'Unlicense', 'scene'),
    'data/media/objects/balls/generic.ase': (b'*3DSMAX_ASCIIEXPORT 200\n', 'Unlicense', 'model'),
    'data/media/objects/balls/ball.bmp': (b'BM' + bytes(range(256)), 'Unlicense', 'texture'),
    'data/media/textures/adboards/ad_b.bmp': (b'BMb', 'Unlicense', 'adboard'),
    'data/media/textures/adboards/ad_a.bmp': (b'BMa', 'Unlicense', 'adboard'),
    'data/databases/logo.bmp': (b'BMlogo', 'Apache-2.0', 'overlay'),
    'fonts/AlegreyaSansSC-ExtraBold.ttf': (b'\0\1\0\0font', 'OFL-1.1', 'font'),
}


@pytest.fixture
def converter():
    return load_script('ci', 'make_render_media.py')


@pytest.fixture
def fake(tmp_path, converter):
    """A GRF tree with `FAKE_FILES` and GRF's license files, its pinned list and a project."""
    tree = tmp_path / 'grf'
    lines = []
    for name, (data, license_name, role) in FAKE_FILES.items():
        write(tree / converter.source_path(name), data)
        lines.append(f'{sha256(data)}  {license_name}  {role}  {name}\n')
    sources = tmp_path / 'sources.txt'
    sources.write_text('# the pinned files\n' + ''.join(lines), encoding='ascii')
    # GRF's two license files, rebuilt from the repository's texts. Their pins hold only if the
    # repository's texts are GRF's.
    unlicense = open(source_file('LICENSES', 'Unlicense.txt'), 'rb').read()
    write(tree / converter.GRF_UNLICENSE[0], unlicense)
    ofl = open(source_file('LICENSES', 'OFL-1.1.txt'), 'rb').read().decode('ascii')
    assert ofl.startswith(converter.FONT_COPYRIGHT)
    grf_ofl = converter.OFL_HEADER + ofl[len(converter.FONT_COPYRIGHT) :]
    write(tree / converter.GRF_OFL[0], grf_ofl.encode('ascii'))
    project = tmp_path / 'project'
    write(project / 'src' / 'cuju_grf_media' / '__init__.py', b"__version__ = '0.3.0'\n")
    return tree, sources, project


def convert(converter, fake, **options):
    tree, sources, project = fake
    # The fake tree has none of the real generated files unless a test names its own.
    options.setdefault('generated_files', ())
    return converter.convert(
        str(tree), project=str(project), root=ROOT, sources=str(sources), **options
    )


def test_missing_media_name_the_extra(monkeypatch):
    monkeypatch.setattr(_media, 'PACKAGE', 'cuju_no_such_media_package')
    with pytest.raises(cuju.MissingFileError) as caught:
        _media.root()
    assert str(caught.value) == _media.MESSAGE
    assert INSTALL in str(caught.value) and 'cuju-grf-media' in str(caught.value)
    assert isinstance(caught.value, FileNotFoundError)
    with pytest.raises(cuju.MissingFileError, match=re.escape(INSTALL)):
        _media.path('data/media/shaders/noise.bmp')


def test_media_of_another_format_are_refused(tmp_path):
    (tmp_path / 'manifest.json').write_text(json.dumps({'format': 2, 'version': '9.0.0'}))
    with pytest.raises(
        cuju.MissingFileError, match=r'format 2.*reads format 1.*cuju-engine\[pixels\]'
    ):
        _media.manifest(tmp_path)


def test_the_formats_agree(converter):
    check = load_script('ci', 'check_wheel.py')
    assert converter.FORMAT == _media.FORMAT == check.MEDIA_FORMAT
    assert ' AND '.join(converter.LICENSES) == check.MEDIA_LICENSE


def test_the_license_texts_are_grfs(converter, fake):
    tree, _, _ = fake
    # The fixture rebuilt GRF's files from the repository's texts, so their pins hold.
    assert converter.tree_problems(str(tree), []) == []
    assert converter.license_problems(str(tree), ROOT) == []


def test_the_pinned_list_covers_what_the_renderer_reads(converter):
    entries = converter.read_sources(source_file('packaging', 'cuju-grf-media', 'sources.txt'))
    paths = [entry['path'] for entry in entries]
    assert len(paths) == len(set(paths)) == 106
    roles = {}
    for entry in entries:
        roles.setdefault(entry['role'], []).append(entry['path'])
    assert sorted(roles) == sorted(converter.ROLES)
    # GRF draws each board by its index in the sorted list of all 16, so all 16 ship.
    assert len(roles['adboard']) == 16
    assert all(path.endswith('.bmp') for path in roles['adboard'])
    assert roles['font'] == ['fonts/AlegreyaSansSC-ExtraBold.ttf']
    licenses = {entry['path']: entry['license'] for entry in entries}
    assert {path for path, name in licenses.items() if name == 'OFL-1.1'} == set(roles['font'])
    assert sorted(path for path, name in licenses.items() if name == 'Apache-2.0') == [
        'data/databases/default/images_teams/primeradivision/fcbarcelona_logo.bmp',
        'data/databases/default/images_teams/primeradivision/realmadrid_logo.bmp',
    ]
    for path in paths:
        assert converter.source_path(path).startswith('third_party/')


def test_the_converter_copies_every_file_unchanged(converter, fake):
    manifest = convert(converter, fake)
    _, _, project = fake
    package = project / 'src' / 'cuju_grf_media'
    assert [entry['path'] for entry in manifest['files']] == list(FAKE_FILES)
    for name, (data, license_name, role) in FAKE_FILES.items():
        assert (package / name).read_bytes() == data
        entry = next(entry for entry in manifest['files'] if entry['path'] == name)
        assert entry == {
            'path': name,
            'source': converter.source_path(name),
            'bytes': len(data),
            'sha256': sha256(data),
            'license': license_name,
            'role': role,
        }
    assert manifest['adboards'] == [
        'data/media/textures/adboards/ad_a.bmp',
        'data/media/textures/adboards/ad_b.bmp',
    ]
    assert manifest['source']['commit'] == converter.GRF_COMMIT
    assert manifest['version'] == '0.3.0'
    assert json.loads((package / 'manifest.json').read_text()) == manifest
    for name in converter.LICENSE_TEXTS.values():
        assert (project / name).read_bytes() == open(source_file(*name.split('/')), 'rb').read()
    # The locator reads what the converter writes, and the lists agree with the files.
    assert _media.manifest(package) == manifest
    assert _media.problems(package) == []
    assert _media.path('fonts/AlegreyaSansSC-ExtraBold.ttf', package).read_bytes().endswith(b'font')
    with pytest.raises(cuju.MissingFileError, match='lists no'):
        _media.path('data/unlisted.bmp', package)


def test_the_checksums_find_a_changed_file(converter, fake):
    convert(converter, fake)
    package = fake[2] / 'src' / 'cuju_grf_media'
    (package / 'data' / 'media' / 'objects' / 'balls' / 'ball.bmp').write_bytes(b'BM changed')
    assert _media.problems(package) == [
        'data/media/objects/balls/ball.bmp differs from its SHA-256 in SHA256SUMS'
    ]
    (package / 'fonts' / 'AlegreyaSansSC-ExtraBold.ttf').unlink()
    assert 'missing fonts/AlegreyaSansSC-ExtraBold.ttf' in _media.problems(package)


def test_a_tree_of_another_commit_writes_nothing(converter, fake):
    tree, _, project = fake
    write(tree / 'third_party/gfootball_engine/data/media/objects/balls/ball.bmp', b'BM other')
    (tree / 'third_party/gfootball_engine/data/media/objects/balls/generic.ase').unlink()
    with pytest.raises(converter.ConversionError) as caught:
        convert(converter, fake)
    message = str(caught.value)
    assert converter.GRF_COMMIT in message
    assert 'third_party/gfootball_engine/data/media/objects/balls/ball.bmp differs' in message
    assert 'missing third_party/gfootball_engine/data/media/objects/balls/generic.ase' in message
    assert not (project / 'src' / 'cuju_grf_media' / 'manifest.json').exists()


def test_generated_files_are_required_and_listed(converter, fake, tmp_path):
    generated = (('fonts/atlas.bin', 'OFL-1.1', 'font', 'ci/make_atlas.py'),)
    with pytest.raises(converter.ConversionError, match='missing the generated file fonts/atlas'):
        convert(converter, fake, generated_files=generated)
    write(tmp_path / 'made' / 'fonts' / 'atlas.bin', b'glyphs')
    manifest = convert(converter, fake, generated=str(tmp_path / 'made'), generated_files=generated)
    assert manifest['generated'] == [
        {
            'path': 'fonts/atlas.bin',
            'producer': 'ci/make_atlas.py',
            'bytes': 6,
            'sha256': sha256(b'glyphs'),
            'license': 'OFL-1.1',
            'role': 'font',
        }
    ]
    package = fake[2] / 'src' / 'cuju_grf_media'
    assert _media.problems(package) == []
    assert _media.path('fonts/atlas.bin', package).read_bytes() == b'glyphs'


def test_the_glyph_atlas_is_generated_and_pinned(converter, fake, tmp_path, monkeypatch):
    assert converter.GENERATED == (
        ('fonts/caption_glyphs.bin', 'OFL-1.1', 'font', 'ci/make_glyph_atlas.py'),
    )
    assert set(converter.GENERATED_SHA256) == {path for path, *_ in converter.GENERATED}
    generated = (('fonts/atlas.bin', 'OFL-1.1', 'font', 'ci/make_atlas.py'),)
    write(tmp_path / 'made' / 'fonts' / 'atlas.bin', b'other glyphs')
    monkeypatch.setitem(converter.GENERATED_SHA256, 'fonts/atlas.bin', sha256(b'glyphs'))
    with pytest.raises(converter.ConversionError, match=r'fonts/atlas.bin has the SHA-256'):
        convert(converter, fake, generated=str(tmp_path / 'made'), generated_files=generated)
    write(tmp_path / 'made' / 'fonts' / 'atlas.bin', b'glyphs')
    manifest = convert(converter, fake, generated=str(tmp_path / 'made'), generated_files=generated)
    assert manifest['generated'][0]['sha256'] == sha256(b'glyphs')


def test_a_wheel_with_the_loader_holds_the_renderer(tmp_path):
    check = load_script('ci', 'check_wheel.py')
    path = tmp_path / 'cuju-0.3.0-py3-none-manylinux_2_28_x86_64.whl'
    for members, problems in (
        (('cuju/libcuju_cpu.so', 'cuju/_pixels.py'), ['missing the renderer library']),
        (('cuju/libcuju_cpu.so', 'cuju/_pixels.py', 'cuju/libcuju_render.so'), []),
    ):
        with zipfile.ZipFile(path, 'w') as wheel:
            for name in check.LICENSES:
                wheel.writestr(f'cuju-0.3.0.dist-info/licenses/{name}', '')
            for name in members:
                wheel.writestr(name, '')
        assert check.problems(str(path), cpu_only=True) == problems


def test_a_wheel_of_cuju_holds_no_folder_of_another_package(tmp_path):
    check = load_script('ci', 'check_wheel.py')
    path = tmp_path / 'cuju-0.3.0-py3-none-manylinux_2_28_x86_64.whl'
    members = ('cuju/libcuju_cpu.so', 'cuju/libcuju_cuda.so', 'cuju/gfootball/__init__.py')
    for extra, problems in (
        ((), []),
        (('cuju/fakepkg/__init__.py', 'cuju/fakepkg/x.py'), ['expected no other package']),
    ):
        with zipfile.ZipFile(path, 'w') as wheel:
            for name in check.LICENSES:
                wheel.writestr(f'cuju-0.3.0.dist-info/licenses/{name}', '')
            for name in (*members, *extra):
                wheel.writestr(name, '')
        found = check.problems(str(path), cpu_only=False)
        assert len(found) == len(problems)
        if problems:
            assert (
                found[0] == 'expected no other package, found cuju/fakepkg/__init__.py and 1 more'
            )


def media_wheel(path, project, extra=None):
    """Writes a wheel of cuju-grf-media from a converted project, as hatchling lays it out."""
    package = project / 'src' / 'cuju_grf_media'
    dist_info = 'cuju_grf_media-0.3.0.dist-info'
    with zipfile.ZipFile(path, 'w') as wheel:
        for file in sorted(package.rglob('*')):
            if file.is_file():
                wheel.write(file, 'cuju_grf_media/' + file.relative_to(package).as_posix())
        for name in ('LICENSE', 'LICENSES/Unlicense.txt', 'LICENSES/OFL-1.1.txt'):
            wheel.write(project / name, f'{dist_info}/licenses/{name}')
        wheel.write(
            source_file('packaging', 'cuju-grf-media', 'NOTICE'), f'{dist_info}/licenses/NOTICE'
        )
        wheel.writestr(
            f'{dist_info}/METADATA',
            'Metadata-Version: 2.4\nName: cuju-grf-media\nVersion: 0.3.0\n'
            'License-Expression: Apache-2.0 AND Unlicense AND OFL-1.1\n',
        )
        if extra:
            wheel.writestr(extra, b'')


def test_wheel_check_reads_the_media_wheel(converter, fake, tmp_path):
    check = load_script('ci', 'check_wheel.py')
    convert(converter, fake)
    project = fake[2]
    path = tmp_path / 'cuju_grf_media-0.3.0-py3-none-any.whl'
    media_wheel(path, project)
    assert check.media_problems(str(path)) == []
    media_wheel(path, project, extra='cuju_grf_media/data/stray.bmp')
    assert check.media_problems(str(path)) == [
        'expected only the listed files, found cuju_grf_media/data/stray.bmp and 0 more'
    ]
    (project / 'src' / 'cuju_grf_media' / 'data' / 'databases' / 'logo.bmp').write_bytes(b'BM')
    media_wheel(path, project)
    assert check.media_problems(str(path)) == [
        'data/databases/logo.bmp differs from its manifest entry'
    ]
    # A wheel of cuju holds none of the media.
    cuju_wheel = tmp_path / 'cuju-0.3.0-py3-none-manylinux_2_28_x86_64.whl'
    with zipfile.ZipFile(cuju_wheel, 'w') as wheel:
        for name in check.LICENSES:
            wheel.writestr(f'cuju-0.3.0.dist-info/licenses/{name}', '')
        wheel.writestr('cuju/libcuju_cpu.so', '')
        wheel.writestr('cuju/libcuju_cuda.so', '')
        wheel.writestr('cuju_grf_media/manifest.json', '')
    assert check.problems(str(cuju_wheel), cpu_only=False) == [
        'expected no media, found cuju_grf_media/manifest.json and 0 more'
    ]


def test_the_extra_takes_the_media_of_this_release():
    tomllib = pytest.importorskip('tomllib')
    with open(source_file('pyproject.toml'), 'rb') as handle:
        extras = tomllib.load(handle)['project']['optional-dependencies']
    with open(
        source_file('packaging', 'cuju-grf-media', 'src', 'cuju_grf_media', '__init__.py')
    ) as f:
        version = re.search(r"__version__ = '(\d+)\.(\d+)\.\d+'", f.read()).groups()
    major, minor = map(int, version)
    requirement = f'cuju-grf-media>={major}.{minor}.0,<{major}.{minor + 1}'
    assert extras['pixels'] == [requirement]
    assert requirement in extras['gfootball-video']


def test_the_installed_media_match_their_lists():
    try:
        folder = _media.root()
    except cuju.MissingFileError:
        pytest.skip('cuju-grf-media is not installed')
    manifest = _media.manifest(folder)
    assert _media.problems(folder) == []
    assert len(manifest['adboards']) == 16
    sources = os.path.join(ROOT, 'packaging', 'cuju-grf-media', 'sources.txt')
    if os.path.isfile(sources):
        converter = load_script('ci', 'make_render_media.py')
        pinned = {entry['path']: entry['sha256'] for entry in converter.read_sources(sources)}
        assert {entry['path']: entry['sha256'] for entry in manifest['files']} == pinned


def test_release_uploads_the_media_only_where_pypi_lacks_their_version():
    path = os.path.join(ROOT, '.github', 'workflows', 'release.yml')
    if not os.path.isfile(path):
        pytest.skip('needs the source tree, which lacks the release workflow')
    with open(path, encoding='utf-8') as handle:
        workflow = handle.read()
    job = workflow[workflow.index('\n    publish-grf-media:') :]
    assert 'environment: pypi-grf-media' in job and 'id-token: write' in job
    assert 'https://pypi.org/pypi/cuju-grf-media/$version/json' in job
    assert "if: steps.pypi.outputs.upload == 'true'" in job
    assert '*) ' in job and 'exit 1' in job


def test_the_renderer_ports_ship_their_licenses():
    pyproject = source_file('pyproject.toml')
    check = load_script('ci', 'check_wheel.py')
    text = open(pyproject, encoding='utf-8').read()
    notice = open(source_file('NOTICE'), encoding='utf-8').read()
    assert re.search(r'^license = ".* AND Zlib"$', text, re.MULTILINE)
    for name, line in (
        ('LICENSES/MIT-Mesa.txt', 'Permission is hereby granted, free of charge'),
        ('LICENSES/Zlib.txt', 'Altered source versions must be plainly marked'),
    ):
        assert line in open(source_file(*name.split('/')), encoding='utf-8').read()
        assert f'"{name}"' in text
        assert name in check.LICENSES
        assert name in notice
