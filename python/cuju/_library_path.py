# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import os
import platform
import sys
from typing import Any

from .errors import ChoiceError

_PACKAGE_DIRECTORY = os.path.dirname(__file__)
if os.name == 'nt':
    _LIBRARY_FILE = '{}.dll'
elif sys.platform == 'darwin':
    _LIBRARY_FILE = 'lib{}.dylib'
else:
    _LIBRARY_FILE = 'lib{}.so'
_LIBRARIES = {
    'cuda': (_LIBRARY_FILE.format('cuju_cuda'), 'CUJU_CUDA_LIBRARY'),
    'cpu': (_LIBRARY_FILE.format('cuju_cpu'), 'CUJU_CPU_LIBRARY'),
}


def _backend(device: Any) -> str:
    if isinstance(device, str) and device in _LIBRARIES:
        return device
    import torch

    try:
        kind = torch.device(device).type
    except (RuntimeError, TypeError):
        kind = None
    if kind not in _LIBRARIES:
        raise ChoiceError(
            f"expected the device 'cuda', 'cuda:N', 'cpu' or a torch.device, found {device!r}"
        )
    return kind


def _automatic(device: Any) -> Any:
    if not (isinstance(device, str) and device == 'auto'):
        return device
    name, variable = _LIBRARIES['cuda']
    library = os.environ.get(variable) or os.path.join(_PACKAGE_DIRECTORY, name)
    present = bool(os.environ.get(variable)) or os.path.isfile(library)
    if not present:
        return 'cpu'
    import torch

    if torch.cuda.is_available():
        return f'cuda:{torch.cuda.current_device()}'
    return 'cpu'


def library_path(device: Any = 'cuda') -> str:
    name, variable = _LIBRARIES[_backend(_automatic(device))]
    return os.environ.get(variable) or os.path.join(_PACKAGE_DIRECTORY, name)


_MACOS_FEATURES = (
    ('AVX2', 'hw.optional.avx2_0'),
    ('BMI2', 'hw.optional.bmi2'),
    ('FMA', 'hw.optional.fma'),
)


def _macos_feature(key):
    try:
        sysctl = ctypes.CDLL(None).sysctlbyname
    except (OSError, AttributeError):
        return True
    value = ctypes.c_int(0)
    size = ctypes.c_size_t(ctypes.sizeof(value))
    if sysctl(key.encode(), ctypes.byref(value), ctypes.byref(size), None, ctypes.c_size_t(0)):
        return True
    return value.value != 0


def _missing_processor_features():
    """Returns the features among AVX2, BMI2 and FMA that the processor lacks, as far as known."""
    if os.name == 'nt':
        # Windows answers for AVX2 (`PF_AVX2_INSTRUCTIONS_AVAILABLE`, 40), not for BMI2 or FMA.
        return [] if ctypes.windll.kernel32.IsProcessorFeaturePresent(40) else ['AVX2']
    if sys.platform == 'darwin':
        return [name for name, key in _MACOS_FEATURES if not _macos_feature(key)]
    try:
        with open('/proc/cpuinfo', encoding='ascii', errors='replace') as handle:
            for line in handle:
                if line.startswith('flags'):
                    flags = set(line.partition(':')[2].split())
                    return [name.upper() for name in ('avx2', 'bmi2', 'fma') if name not in flags]
    except OSError:
        pass
    return []


def _check_processor(path):
    """Raises `DeviceError` for the packaged CPU library on a CPU without AVX2, BMI2 or FMA."""
    from .errors import DeviceError

    if platform.machine().lower() not in ('x86_64', 'amd64'):
        return
    if os.path.abspath(path) != os.path.join(
        os.path.abspath(_PACKAGE_DIRECTORY), _LIBRARIES['cpu'][0]
    ):
        return
    missing = _missing_processor_features()
    if missing:
        raise DeviceError(
            'the CPU library of cuju needs a processor with AVX2, BMI2 and FMA, and this one lacks '
            + ', '.join(missing)
        )
