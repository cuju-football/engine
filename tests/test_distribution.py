# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks the distributions (`cuju-engine`, `cuju-engine-cpu`, rebuilds, images) and their hosts."""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile

import pytest
import torch

import cuju
from conftest import play_digest
from cuju._batch import library
from cuju._library_path import library_path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


def test_cpu_only_distribution_drops_only_the_cuda_backend(tmp_path):
    tomllib = pytest.importorskip('tomllib')
    original = source_file('pyproject.toml')
    with open(original, 'rb') as handle:
        before = tomllib.load(handle)
    if before['project']['name'] != 'cuju-engine':
        pytest.skip('the source tree is already the cuju-engine-cpu project')
    copy = tmp_path / 'pyproject.toml'
    shutil.copyfile(original, copy)
    subprocess.run([sys.executable, source_file('ci', 'cpu_only.py'), str(copy)], check=True)
    with open(copy, 'rb') as handle:
        after = tomllib.load(handle)
    assert after['project']['name'] == 'cuju-engine-cpu'
    defines = after['tool']['scikit-build']['cmake']['define']
    assert defines['CUJU_CUDA'] == 'OFF'
    assert 'CMAKE_CUDA_ARCHITECTURES' not in defines
    assert not [line for line in after['project']['classifiers'] if 'CUDA' in line]
    for table in (before, after):
        project = table['project']
        del project['name'], project['description']
        project['classifiers'] = [line for line in project['classifiers'] if 'CUDA' not in line]
        table['tool']['scikit-build']['cmake']['define'].pop('CMAKE_CUDA_ARCHITECTURES', None)
        table['tool']['scikit-build']['cmake']['define'].pop('CUJU_CUDA', None)
    assert before == after


def test_cpu_only_refuses_an_unknown_project():
    script = load_script('ci', 'cpu_only.py')
    with pytest.raises(ValueError, match='expected one match'):
        script.cpu_only('[project]\nname = "other"\n')


def test_missing_cuda_library_points_to_the_cpu(monkeypatch):
    monkeypatch.delenv('CUJU_CUDA_LIBRARY', raising=False)
    monkeypatch.setitem(library._LIBRARIES, 'cuda', ('no_such_library', 'CUJU_CUDA_LIBRARY'))
    with pytest.raises(FileNotFoundError, match="device='cpu'"):
        library._load_library(device='cuda')


def test_wheel_check_tells_the_distributions_apart(tmp_path):
    script = load_script('ci', 'check_wheel.py')
    path = tmp_path / 'cuju_engine_cpu-0.1.0-py3-none-manylinux_2_28_x86_64.whl'
    with zipfile.ZipFile(path, 'w') as wheel:
        for name in script.LICENSES:
            wheel.writestr(f'cuju_engine_cpu-0.1.0.dist-info/licenses/{name}', '')
        wheel.writestr('cuju/libcuju_cpu.so', '')
    assert script.problems(path, cpu_only=True) == []
    assert script.problems(path, cpu_only=False) == ['missing the CUDA library']


def write_wheel(path, members, date_time=(2026, 1, 1, 0, 0, 0)):
    with zipfile.ZipFile(path, 'w') as wheel:
        for name, data in members.items():
            wheel.writestr(zipfile.ZipInfo(name, date_time), data)


def test_build_comparison_finds_what_differs(tmp_path, capsys):
    script = load_script('ci', 'compare_builds.py')
    members = {'cuju/__init__.py': b'', 'cuju/libcuju_cpu.so': b'\x7fELF library'}
    for build in ('first', 'second', 'later', 'other'):
        (tmp_path / build).mkdir()
    write_wheel(tmp_path / 'first' / 'cuju.whl', members)
    write_wheel(tmp_path / 'second' / 'cuju.whl', members)
    assert script.compare(str(tmp_path / 'first'), str(tmp_path / 'second'))
    write_wheel(tmp_path / 'later' / 'cuju.whl', members, date_time=(2026, 1, 2, 0, 0, 0))
    write_wheel(tmp_path / 'other' / 'cuju.whl', {**members, 'cuju/__init__.py': b'#'})
    capsys.readouterr()
    assert not script.compare(str(tmp_path / 'first'), str(tmp_path / 'later'))
    assert 'cuju/__init__.py: the metadata differs' in capsys.readouterr().out
    assert not script.compare(str(tmp_path / 'first'), str(tmp_path / 'other'))
    assert 'cuju/__init__.py: the bytes differ' in capsys.readouterr().out


def test_image_builds_the_architectures_of_the_wheel():
    with open(source_file('Dockerfile'), encoding='utf-8') as handle:
        image = re.search(r'ARG CUDA_ARCHITECTURES="([^"]+)"', handle.read()).group(1)
    with open(source_file('.github', 'workflows', 'release.yml'), encoding='utf-8') as handle:
        wheel = re.search(r'CMAKE_CUDA_ARCHITECTURES="([^"]+)"', handle.read()).group(1)
    assert image == wheel


def test_wheel_holds_only_checked_architectures():
    with open(source_file('CMakeLists.txt'), encoding='utf-8') as handle:
        checked = re.search(r'set\(CUJU_CHECKED_CUDA_ARCHITECTURES ([0-9 ]+)\)', handle.read())
    with open(source_file('.github', 'workflows', 'release.yml'), encoding='utf-8') as handle:
        wheel = re.search(r'CMAKE_CUDA_ARCHITECTURES="([^"]+)"', handle.read()).group(1)
    assert wheel.split(';') == [f'{capability}-real' for capability in checked.group(1).split()]


def test_publishing_needs_a_tag_that_names_the_version():
    with open(source_file('.github', 'workflows', 'release.yml'), encoding='utf-8') as handle:
        workflow = handle.read()
    publish = workflow[workflow.index('\n    publish:') :]
    condition = publish[: publish.index('runs-on:')]
    assert 'inputs.publish' in condition
    assert "startsWith(github.ref, 'refs/tags/v')" in condition
    check = publish.index('"$GITHUB_REF_NAME" != "v$version"')
    assert publish.index("python/cuju/_version.py')['__version__']") < check
    assert check < publish.index('gh-action-pypi-publish')


def test_each_trusted_publishing_job_checks_the_release_tag():
    with open(source_file('.github', 'workflows', 'release.yml'), encoding='utf-8') as handle:
        workflow = handle.read()
    job_sections = re.split(r'(?m)^    ([\w-]+):[ \t]*$', workflow.split('\njobs:\n', 1)[1])
    jobs = {
        name: [step.strip() for step in re.split(r'(?m)^            - ', section)]
        for name, section in zip(job_sections[1::2], job_sections[2::2], strict=True)
    }
    tag_check = next(
        step
        for step in jobs['publish'][1:]
        if step.startswith('name: Check that the tag names the version of the package\n')
    )
    upload_action = 'uses: pypa/gh-action-pypi-publish@'
    missing = []
    for name, (settings, *steps) in jobs.items():
        if not re.search(r'(?m)^            id-token: write[ \t]*$', settings):
            continue
        first_upload = next(
            (index for index, step in enumerate(steps) if step.startswith(upload_action)),
            None,
        )
        if first_upload is None or tag_check not in steps[:first_upload]:
            missing.append(name)
    assert not missing, f'Tag check missing before PyPI publishing in {", ".join(missing)}'


def test_each_published_project_has_its_own_environment():
    with open(source_file('.github', 'workflows', 'release.yml'), encoding='utf-8') as handle:
        workflow = handle.read()
    environments = {}
    for job in ('publish', 'publish-gfootball', 'publish-grf-media'):
        text = workflow[workflow.index(f'\n    {job}:') + 1 :]
        text = text[: text.index('steps:')]
        environments[job] = text.split('environment: ', 1)[1].split()[0]
    assert environments == {
        'publish': 'pypi',
        'publish-gfootball': 'pypi-gfootball',
        'publish-grf-media': 'pypi-grf-media',
    }


def test_apptainer_takes_the_image_of_this_version():
    with open(source_file('apptainer.def'), encoding='utf-8') as handle:
        image = re.search(r'(?m)^ +IMAGE=(\S+)$', handle.read()).group(1)
    assert image == f'cuju:{cuju.__version__}'


@pytest.mark.parametrize('threads', [1, 4])
def test_a_caller_that_flushes_subnormals_changes_nothing(threads):
    digests = os.path.join(os.path.dirname(os.path.abspath(cuju.__file__)), 'data', 'digests.json')
    with open(digests, encoding='utf-8') as handle:
        expected = json.load(handle)
    case = expected['cases']['5_vs_5/simple115v2']
    if not torch.set_flush_denormal(True):
        pytest.skip('PyTorch cannot flush subnormals on this processor')
    try:
        found = play_digest(
            case['assets'],
            'cpu',
            expected['steps'],
            seed=expected['seed'],
            environments=expected['environments'],
            threads=threads,
        )
    finally:
        torch.set_flush_denormal(False)
    assert found == case['digest']


def test_library_names_follow_the_system(monkeypatch):
    monkeypatch.delenv('CUJU_CPU_LIBRARY', raising=False)
    monkeypatch.delenv('CUJU_CUDA_LIBRARY', raising=False)
    if os.name == 'nt':
        pattern = '{}.dll'
    elif sys.platform == 'darwin':
        pattern = 'lib{}.dylib'
    else:
        pattern = 'lib{}.so'
    assert os.path.basename(library_path('cpu')) == pattern.format('cuju_cpu')
    assert os.path.basename(library_path('cuda')) == pattern.format('cuju_cuda')


def test_windows_exports_every_function_that_the_package_calls():
    directory = os.path.dirname(source_file('src', 'environment_api.h'))
    exports = set()
    for name in sorted(os.listdir(directory)):
        if name.endswith('_api.h'):
            with open(os.path.join(directory, name), encoding='utf-8') as handle:
                for line in handle:
                    match = re.match(r'[A-Za-z][^(]*[ *](cuju_[a-z0-9_]+)\(', line)
                    if match:
                        exports.add(match.group(1))
    called = set()
    for name in ('_batch/environment.py', '_batch/library.py', '_batch/memory.py', 'scenarios.py'):
        with open(os.path.join(os.path.dirname(cuju.__file__), name), encoding='utf-8') as handle:
            called.update(re.findall(r'\.(cuju_[a-z0-9_]+)\b', handle.read()))
    assert called and called <= exports
