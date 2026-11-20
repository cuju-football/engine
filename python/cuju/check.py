# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks that this installation reproduces the expected outputs, on the CPU and on every GPU.

    cuju check                # every case on the CPU and on every visible GPU
    cuju check --long         # also the long cases, later episodes and the second half
    cuju check --device cuda  # one backend, which must then be present
    cuju check --list         # the cases, without playing them

It prints the setup, checks the package data against `data/SHA256SUMS`, and plays each case of
`data/digests.json`, four environments with fixed actions whose outputs and final states hash into
one SHA-256, against the digest of a build that matches GRF. The exit status is 0 when everything
that ran matched, else 1. An absent backend is reported and skipped unless `--device` names it,
when the status is 3.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import struct
import sys
import textwrap
import time

import torch

import cuju

from . import _checksums, _cli_parsers
from . import _cli_contract as contract
from ._batch import library as _libraries
from .errors import Error

DIGESTS = os.path.join(_checksums.DATA_DIRECTORY, 'digests.json')
# The processor features that the packaged CPU library needs.
_FEATURES = ('avx2', 'bmi2', 'fma')


def load_cases(long: bool = False) -> dict[str, dict]:
    """Returns the cases of `data/digests.json` with their steps, and the long ones if `long`."""
    with open(DIGESTS, encoding='utf-8') as handle:
        expected = json.load(handle)
    cases = {
        name: {**entry, 'steps': expected['steps']} for name, entry in expected['cases'].items()
    }
    if long:
        cases.update(expected.get('long_cases', {}))
    for entry in cases.values():
        entry.setdefault('seed', expected['seed'])
        entry.setdefault('environments', expected['environments'])
    return cases


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Returns the bytes of a tensor, read without numpy."""
    tensor = tensor.detach().to('cpu').contiguous()
    return ctypes.string_at(tensor.data_ptr(), tensor.numel() * tensor.element_size())


def actions_at(step: int, environments: int, agents: int, action_count: int) -> torch.Tensor:
    """Returns the actions of a step from integer arithmetic alone, the same on every machine."""
    values = [
        [(step * 7 + environment * 3 + agent * 5) % action_count for agent in range(agents)]
        for environment in range(environments)
    ]
    return torch.tensor(values, dtype=torch.int32)


def play_digest(assets, device, steps, seed=7, environments=4, **options) -> str:
    """Plays a batch with the actions of `actions_at` and returns the SHA-256 of its outputs.

    The digest covers every observation, reward and done flag of every step, then the state of
    every environment at the end unless the batch stacks its observations. A GPU batch runs on
    the current CUDA device.
    """
    batch = cuju.BatchEnvironment(
        assets, environment_count=environments, device=device, seed=seed, **options
    )
    digest = hashlib.sha256()

    def add(observation):
        if isinstance(observation, dict):
            for name in sorted(observation):
                digest.update(tensor_bytes(observation[name]))
        else:
            digest.update(tensor_bytes(observation))

    try:
        add(batch.reset())
        for step in range(steps):
            actions = actions_at(step, environments, batch.agents, batch.action_count)
            observation, reward, done, _ = batch.step(actions.to(device))
            add(observation)
            digest.update(tensor_bytes(reward))
            digest.update(tensor_bytes(done))
        if not batch.stacked:
            for index in range(environments):
                digest.update(batch.get_state(index))
    finally:
        batch.close()
    return digest.hexdigest()


def _fatbin_sections(data: bytes) -> list[tuple[int, int]] | None:
    """Returns the offset and size of each section that holds fat binaries, or `None`.

    A 64-bit little-endian ELF file names it `.nv_fatbin`, a Windows DLL `.nv_fatb`.
    """
    if data[:6] == b'\x7fELF\x02\x01':
        (table,) = struct.unpack_from('<Q', data, 0x28)
        entry_size, count, names = struct.unpack_from('<HHH', data, 0x3A)

        def section(index):
            name, _, _, _, offset, size = struct.unpack_from(
                '<IIQQQQ', data, table + index * entry_size
            )
            return name, offset, size

        names_offset = section(names)[1]
        found = []
        for index in range(count):
            name, offset, size = section(index)
            start = names_offset + name
            if data[start : data.index(b'\0', start)] == b'.nv_fatbin':
                found.append((offset, size))
        return found
    if data[:2] == b'MZ':
        (header,) = struct.unpack_from('<I', data, 0x3C)
        if data[header : header + 4] != b'PE\0\0':
            return None
        _, count, _, _, _, optional_size, _ = struct.unpack_from('<HHIIIHH', data, header + 4)
        table = header + 24 + optional_size
        found = []
        for index in range(count):
            name, _, _, size, offset = struct.unpack_from('<8sIIII', data, table + index * 40)
            if name.rstrip(b'\0') == b'.nv_fatb':
                found.append((offset, size))
        return found
    return None


def cuda_architectures(path: str) -> list[str] | None:
    """Returns the GPU code in a CUDA library's fat binary, or `None` for a file without one.

    Machine code reads as `sm_86` and PTX as `compute_86`.
    """
    try:
        with open(path, 'rb') as handle:
            data = handle.read()
        sections = _fatbin_sections(data)
        if sections is None:
            return None
        found = []
        for offset, size in sections:
            position = offset
            while position + 16 <= offset + size:
                magic, _, header, length = struct.unpack_from('<IHHQ', data, position)
                if magic != 0xBA55ED50:
                    position += 8
                    continue
                entry = position + header
                while entry < position + header + length:
                    # Each entry holds its kind, 1 for PTX and 2 for machine code, its header size,
                    # its payload size and, at byte 28, the architecture.
                    kind, _, entry_header, payload = struct.unpack_from('<HHIQ', data, entry)
                    (architecture,) = struct.unpack_from('<I', data, entry + 0x1C)
                    prefix = {1: 'compute', 2: 'sm'}.get(kind)
                    if prefix and f'{prefix}_{architecture}' not in found:
                        found.append(f'{prefix}_{architecture}')
                    entry += entry_header + payload
                position += header + length
        return sorted(found, key=lambda code: (code.startswith('compute'), int(code.split('_')[1])))
    except (OSError, struct.error, ValueError):
        return None


def code_for(capability: tuple[int, int], architectures: list[str]) -> str:
    """Returns which code of the CUDA library a GPU of `capability` runs, as the driver picks it."""
    major, minor = capability
    machine = [int(code[3:]) for code in architectures if code.startswith('sm_')]
    ptx = [int(code[8:]) for code in architectures if code.startswith('compute_')]
    # Machine code runs on its own major version from its minor version up.
    native = [code for code in machine if code // 10 == major and code % 10 <= minor]
    if native:
        return f'runs the machine code of sm_{max(native)}'
    compiled = [code for code in ptx if code <= major * 10 + minor]
    if compiled:
        return (
            f'runs the PTX of compute_{max(compiled)}, compiled by the driver for this GPU, which '
            'the digests below must then confirm'
        )
    return (
        f'holds no code for compute capability {major}.{minor}: build cuju from the repository '
        'for this GPU'
    )


def _processor():
    """Returns the model and the flags of the processor, from `/proc/cpuinfo`."""
    model, flags = platform.processor() or platform.machine(), set()
    try:
        with open('/proc/cpuinfo', encoding='ascii', errors='replace') as handle:
            for line in handle:
                key, _, value = line.partition(':')
                if key.strip() == 'model name' and model in ('', 'x86_64'):
                    model = value.strip()
                elif key.strip() == 'flags':
                    flags = set(value.split())
                    break
    except OSError:
        pass
    return model, flags


def _cpu_quota():
    """Returns the CPU quota of the cgroup of the process, such as `200000 100000`, or `None`."""
    try:
        with open('/sys/fs/cgroup/cpu.max', encoding='ascii') as handle:
            quota = handle.read().strip()
    except OSError:
        return None
    return None if quota.startswith('max') else quota


def _driver():
    """Returns the NVIDIA driver's version and the CUDA version it supports, where they load."""
    parts = []
    try:
        nvml = ctypes.CDLL('nvml.dll' if os.name == 'nt' else 'libnvidia-ml.so.1')
        if nvml.nvmlInit_v2() == 0:
            version = ctypes.create_string_buffer(96)
            if nvml.nvmlSystemGetDriverVersion(version, 96) == 0:
                parts.append(f'driver {version.value.decode()}')
            nvml.nvmlShutdown()
    except (OSError, AttributeError):
        pass
    try:
        version = ctypes.c_int()
        driver = 'nvcuda.dll' if os.name == 'nt' else 'libcuda.so.1'
        if ctypes.CDLL(driver).cuDriverGetVersion(ctypes.byref(version)) == 0:
            parts.append(f'CUDA {version.value // 1000}.{version.value % 1000 // 10}')
    except (OSError, AttributeError):
        pass
    return ', '.join(parts) or 'not found'


def _version(module):
    try:
        return importlib.metadata.version(module)
    except importlib.metadata.PackageNotFoundError:
        return None


def _library(device):
    """Returns the lines that describe the library of `device`, and its GPU code for `cuda`."""
    path = cuju.library_path(device)
    variable = _libraries._LIBRARIES[device][1]
    source = f'from {variable}' if os.environ.get(variable) else 'of the package'
    if not os.path.isfile(path):
        return [f'{device} library: missing at {path} ({source})'], None
    size = os.path.getsize(path) / 2**20
    lines = [
        f'{device} library: {path} ({source}), {size:.1f} MiB, '
        f'SHA-256 {_checksums.file_sha256(path)}'
    ]
    architectures = cuda_architectures(path) if device == 'cuda' else None
    if device == 'cuda':
        machine = [code for code in architectures or [] if code.startswith('sm_')]
        ptx = [code for code in architectures or [] if code.startswith('compute_')]
        lines.append(
            f'  GPU code: machine code for {" ".join(machine) or "none"}; '
            f'PTX for {" ".join(ptx) or "none"}'
            if architectures is not None
            else '  GPU code: unreadable'
        )
    return lines, architectures


def diagnostics() -> list[str]:
    """Returns lines on this installation's versions, processor, libraries and GPUs."""
    version = getattr(cuju, '__version__', None) or _version('cuju-engine') or 'from a source tree'
    lines = [
        f'cuju {version} at {os.path.dirname(os.path.abspath(cuju.__file__))}',
        f'Python {platform.python_version()}, torch {torch.__version__} '
        f'(CUDA {torch.version.cuda or "none"}), numpy {_version("numpy") or "none"}, '
        f'gymnasium {_version("gymnasium") or "none"}',
    ]
    build_info = getattr(cuju, 'build_info', None)
    if callable(build_info):
        lines.append(f'build: {build_info()}')
    libc = ' '.join(platform.libc_ver()).strip()
    lines.append(f'{platform.platform()}{f", {libc}" if libc else ""}')
    model, flags = _processor()
    usable = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    quota = _cpu_quota()
    lines.append(
        f'CPU: {model}, {os.cpu_count()} logical processors, {usable} usable by this process'
        + (f', cgroup quota {quota}' if quota else '')
    )
    if platform.machine().lower() not in ('x86_64', 'amd64'):
        missing = []
    elif flags:
        missing = [name.upper() for name in _FEATURES if name not in flags]
    else:
        # Without `/proc/cpuinfo`, the system answers for what it can, Windows for AVX2 alone.
        missing = _libraries._missing_processor_features()
    lines.append(
        '  flags: nothing that the packaged CPU library needs is missing'
        if not missing
        else f'  flags: {", ".join(missing)} missing, so the packaged CPU library cannot run here'
    )
    cpu_lines, _ = _library('cpu')
    cuda_lines, architectures = _library('cuda')
    lines += cpu_lines + cuda_lines
    if torch.cuda.is_available():
        lines.append(f'NVIDIA: {_driver()}')
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            capability = (properties.major, properties.minor)
            line = (
                f'GPU {index}: {properties.name}, compute capability {capability[0]}.'
                f'{capability[1]}, {properties.total_memory / 2**30:.1f} GiB'
            )
            if architectures is not None:
                line += f'; the CUDA library {code_for(capability, architectures)}'
            lines.append(line)
    else:
        lines.append('GPU: none that PyTorch can use')
    names = sorted(name for name in os.environ if name.startswith('CUJU_'))
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
        names.append('CUDA_VISIBLE_DEVICES')
    lines.append(
        'environment: '
        + (', '.join(f'{name}={os.environ[name]}' for name in names) or 'no CUJU_ variable set')
    )
    if not os.environ.get('CUJU_LOG'):
        lines.append('  CUJU_LOG=1 logs the creation of every batch')
    return lines


def check_data() -> list[str]:
    """Returns the files of the package data that differ from `data/SHA256SUMS` or that it lacks."""
    try:
        listed = _checksums.read_checksums()
    except (OSError, ValueError) as error:
        return [f'the list cannot be read: {error}']
    problems = []
    for name, digest in listed.items():
        path = os.path.join(_checksums.DATA_DIRECTORY, name)
        if not os.path.isfile(path):
            problems.append(f'{name} is missing')
        elif _checksums.file_sha256(path) != digest:
            problems.append(f'{name} differs')
    problems += [f'{name} is not listed' for name in _checksums.data_files() if name not in listed]
    return problems


def _loads(device):
    """Returns why the library of `device` does not load here, or `None` if it does."""
    try:
        _libraries._load_library(None, device)
    except (OSError, RuntimeError) as error:
        return str(error)
    return None


def _devices(requested):
    """Returns the devices to play as `(label, device, CUDA index)`, and the notes on the absent."""
    devices, notes = [], []
    if requested in ('all', 'cpu'):
        problem = _loads('cpu')
        if problem is None:
            devices.append(('cpu', 'cpu', None))
        else:
            notes.append(f'the CPU cases did not run, {problem}')
    if requested in ('all', 'cuda'):
        problem = _loads('cuda')
        if problem is not None:
            notes.append(f'the GPU cases did not run, {problem}')
        elif not torch.cuda.is_available():
            notes.append('the GPU cases did not run, no GPU that PyTorch can use')
        else:
            devices += [
                (f'cuda:{index}', 'cuda', index) for index in range(torch.cuda.device_count())
            ]
    return devices, notes


def run_case(entry, device, index):
    """Plays one case on `device`, on CUDA device `index` if given, and returns its digest."""
    arguments = (entry['assets'], device, entry['steps'])
    options = {'seed': entry['seed'], 'environments': entry['environments'], **entry['options']}
    if index is None:
        return play_digest(*arguments, **options)
    with torch.cuda.device(index):
        return play_digest(*arguments, **options)


def _list_cases(cases, arguments):
    if arguments.json:
        contract.emit(
            'engine.check.list/1',
            {
                'cases': [
                    {
                        'name': name,
                        'assets': entry['assets'],
                        'steps': entry['steps'],
                        'options': entry['options'],
                    }
                    for name, entry in cases.items()
                ]
            },
        )
    elif cases:
        rows = [
            (name, str(entry['assets']), f'{entry["steps"]} steps', str(entry['options'] or ''))
            for name, entry in cases.items()
        ]
        widths = [max(map(len, column)) for column in zip(*rows, strict=True)]
        if sys.stdout.isatty():
            columns = shutil.get_terminal_size().columns
            widths[0] = min(widths[0], max(1, (columns - widths[2] - 6) // 2))
            widths[1] = min(widths[1], max(1, columns - widths[0] - widths[2] - 6))
            widths[3] = min(widths[3], max(0, columns - sum(widths[:3]) - 6))
        for row in rows:
            cells = []
            for value, width in zip(row, widths, strict=True):
                if width and len(value) > width:
                    value = value[: width - 1] + '…'
                cells.append(f'{value:<{width}}' if width else '')
            print('  '.join(cells).rstrip())
    if not cases and not getattr(arguments, 'quiet', False):
        print(
            'No cases are listed. Run cuju check --list --long to include the long cases.',
            file=sys.stderr,
        )
    return 0


def _run(arguments: argparse.Namespace) -> int:
    cases = load_cases(arguments.long or bool(arguments.case))
    if arguments.case:
        for name in arguments.case:
            if name not in cases:
                arguments._command_parser.error(
                    f'expected a named check case, found {name!r}',
                    hint='run cuju check --list --long to list the cases.',
                )
        cases = {name: cases[name] for name in arguments.case}

    if arguments.list:
        return _list_cases(cases, arguments)

    quiet = getattr(arguments, 'quiet', False)
    interactive = not quiet and sys.stderr.isatty()
    columns = shutil.get_terminal_size().columns if interactive else 0
    line_width = 0

    def progress(text: str, *, update: bool = False) -> None:
        nonlocal line_width
        if quiet:
            return
        if interactive and update:
            text = text[: max(1, columns - 1)]
            print(f'\r{text:<{line_width}}', end='', file=sys.stderr, flush=True)
            line_width = max(line_width, len(text))
        else:
            if line_width:
                print(f'\r{" " * line_width}\r', end='', file=sys.stderr)
                line_width = 0
            print(text, file=sys.stderr)

    setup = diagnostics()
    for line in setup:
        if interactive:
            line = textwrap.fill(
                line,
                columns,
                subsequent_indent='  ',
                break_long_words=False,
                break_on_hyphens=False,
                replace_whitespace=False,
            )
        progress(line)

    problems = check_data()
    if not arguments.json:
        detail = (
            ', '.join(problems)
            if problems
            else f'{len(_checksums.data_files())} files match data/SHA256SUMS'
        )
        print(f'package data: {detail}')

    failures = len(problems)
    devices, notes = _devices(arguments.device)
    results = []
    for label, device, index in devices:
        matched, started = 0, time.perf_counter()
        for name, entry in cases.items():
            begun = time.perf_counter()
            found, error = None, None
            try:
                found = run_case(entry, device, index)
            except Error as problem:
                error = str(problem)
            expected = entry['digest']
            equal = error is None and expected is not None and found == expected
            matched += equal
            results.append(
                {
                    'device': label,
                    'case': name,
                    'expected': expected,
                    'found': found,
                    'matched': equal,
                    'error': error,
                }
            )
            if error is not None:
                progress(f'{label:8} {name}: FAILED, {error}')
                if device == 'cuda':
                    progress(f'{label:8} the other cases of this GPU did not run')
                    break
            elif equal:
                progress(f'{label:8} {name}: ok ({time.perf_counter() - begun:.1f} s)', update=True)
            elif expected is None:
                progress(f'{label:8} {name}: no expected digest yet, this build gives {found}')
            else:
                progress(f'{label:8} {name}: DIFFERENT, {found} instead of {expected}')
        failures += len(cases) - matched
        progress(
            f'{label:8} {matched} of {len(cases)} cases match '
            f'({time.perf_counter() - started:.0f} s)'
        )

    unavailable = not devices
    if unavailable and not failures:
        expected = (
            'a usable backend'
            if arguments.device == 'all'
            else (f'an available {arguments.device} backend')
        )
        return contract.fail(
            'cuju check',
            f'expected {expected}, found none. ' + ', '.join(notes),
            hint='run cuju check --device all to check the available backends.'
            if arguments.device != 'all'
            else 'run python -m pip show cuju-engine to check the installation.',
            as_json=arguments.json,
            status=contract.EXIT_ENVIRONMENT,
            error_id='check.backend_unavailable',
            values={'device': arguments.device},
        )

    status = contract.EXIT_NO if failures else 0
    if unavailable:
        contract.fail('cuju check', ', '.join(notes), status=status)
    else:
        for note in notes:
            progress(note)

    if arguments.json:
        contract.emit(
            'engine.check/1',
            {
                'diagnostics': setup,
                'data_problems': problems,
                'results': results,
                'backend_notes': notes,
                'failures': failures,
                'exit': status,
            },
        )
    elif failures:
        print(f'FAILED: {failures} checks differ or did not run')
    else:
        print('PASSED: every case that ran gave the expected digest')
    return status


def main(argv: list[str] | None = None) -> int:
    """Runs the self-check and returns its exit status."""
    return contract.run_command(_cli_parsers.check(), _run, argv)


if __name__ == '__main__':
    sys.exit(main())
