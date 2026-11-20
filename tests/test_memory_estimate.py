# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks that `cuju.memory_estimate` counts what a batch requests at its creation."""

import os
import re
import subprocess
import sys
import warnings

import pytest
import torch

import cuju
from conftest import CUDA
from cuju._batch import memory

CASES = [
    ('5_vs_5_l4_r0', {}),
    ('5_vs_5_l1_r1', {'representation': 'raw'}),
    ('5_vs_5_l2_r0', {'representation': 'extracted', 'stacked': True}),
    ('11_vs_11_stochastic_l1_r0', {'representation': 'simple115', 'agent_counts': [(3, 0)]}),
    (
        ['5_vs_5_l4_r4', '11_vs_11_stochastic_l1_r0'],
        {'assignment': [0, 1, 0, 1], 'difficulty': [(None, None), (None, 0.5)], 'with_raw': True},
    ),
]

needs_cuda = pytest.mark.skipif(not CUDA, reason='no CUDA device or library')

# The most local memory that a first GPU batch reserves for each thread that the GPU runs at once.
LOCAL_BYTES_PER_THREAD = 8 * 1024


# A child process creates a CPU batch and prints the resident memory that the creation added. A
# fresh process holds no freed memory that the batch could reuse without new pages.
RESIDENT = """
import cuju

# The package loads the engine and PyTorch at the first use of a batch, so the memory they take
# stays out of the measurement.
cuju.BatchEnvironment


def resident():
    with open('/proc/self/status', encoding='ascii') as status:
        for line in status:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024


before = resident()
batch = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=1024, device='cpu', threads=4)
print(resident() - before)
"""

# The start of each child process that measures a GPU. It defines `used`, the memory that the
# whole GPU holds, from NVML, and keeps in `least_free` the least free memory that it saw. Under the
# WDDM driver of Windows and WSL the free memory that CUDA reports follows a budget of the process,
# which moves with the memory of other processes, the pytest process among them once it created a
# batch, and it then counts about twice the local memory of a first batch. A child without NVML
# prints `no NVML` and ends.
DEVICE_MEMORY = """
import ctypes
import sys

import torch
import cuju


class Memory(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in ('total', 'free', 'used')]


try:
    nvml = ctypes.CDLL('nvml.dll' if sys.platform == 'win32' else 'libnvidia-ml.so.1')
except OSError:
    nvml = None
handle = ctypes.c_void_p()
uuid = f'GPU-{torch.cuda.get_device_properties(torch.cuda.current_device()).uuid}'.encode()
if nvml is None or nvml.nvmlInit_v2() or nvml.nvmlDeviceGetHandleByUUID(uuid, ctypes.byref(handle)):
    print('no NVML')
    sys.exit()
least_free = 2**64


def used():
    global least_free
    torch.cuda.synchronize()
    memory = Memory()
    assert nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) == 0
    least_free = min(least_free, memory.free)
    return memory.used
"""

# A child process creates a first batch of its GPU, which makes the driver reserve the local memory
# of the kernels, once per process, and prints the estimate of a larger batch, the memory that its
# creation took and the least free memory.
GPU_MEMORY = """
cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=1, device='cuda').close()
estimate = cuju.memory_estimate('5_vs_5_l4_r0', environment_count=1024, device='cuda')
before = used()
batch = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=1024, device='cuda')
print(estimate, used() - before, least_free)
"""

# A child process creates the first batches of its GPU and prints the local memory that the
# estimate gives, the memory that the first creation took beyond the second, the local memory that
# an estimate gives once a batch exists and the least free memory.
LOCAL_MEMORY = """
estimate = cuju.memory_estimate('5_vs_5_l4_r0', environment_count=64, device='cuda')
before = used()
first = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=64, device='cuda')
between = used()
second = cuju.BatchEnvironment('5_vs_5_l4_r0', environment_count=64, device='cuda')
after = used()
later = cuju.memory_estimate('5_vs_5_l4_r0', environment_count=64, device='cuda')
taken = (between - before) - (after - between)
print(estimate.local_memory_bytes, taken, later.local_memory_bytes, least_free)
"""

# A child process prints the reservation that a first batch of its GPU adds for the kernels and the
# threads that the GPU runs at once.
RESERVATION = """
import torch
import cuju

properties = torch.cuda.get_device_properties(torch.cuda.current_device())
threads = properties.multi_processor_count * getattr(
    properties, 'max_threads_per_multi_processor', 2048
)
estimate = cuju.memory_estimate('5_vs_5_l4_r0', environment_count=1, device='cuda')
print(estimate.local_memory_bytes, threads)
"""

# A child process fills its GPU but for a first batch and half the reservation for the kernels, and
# prints the error of the creation. It prints `created` where the WDDM driver of Windows and WSL
# places the rest in shared system memory, and `full` where the GPU holds no such filler.
RESERVATION_ERROR = """
import torch
import cuju

estimate = cuju.memory_estimate('5_vs_5_l1_r0', environment_count=2, device='cuda')
torch.empty(1, device='cuda')
room = torch.cuda.mem_get_info()[0] - estimate - estimate.local_memory_bytes // 2
try:
    filler = torch.empty(room, dtype=torch.uint8, device='cuda')
except (RuntimeError, ValueError):
    print('full')
else:
    try:
        cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=2, device='cuda')
        print('created')
    except RuntimeError as error:
        print(error)
"""

# A child process creates two batches of its GPU with a free memory that fits the batch alone, and
# prints the warnings of each creation.
FIRST_BATCH_WARNING = """
import warnings
import torch
import cuju

estimate = cuju.memory_estimate('5_vs_5_l1_r0', environment_count=2, device='cuda')
free = estimate + estimate.local_memory_bytes // 2
torch.cuda.mem_get_info = lambda device=None: (free, 2**40)
for _ in range(2):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=2, device='cuda').close()
    print(len(caught), *(str(warning.message) for warning in caught))
"""


def run_child(script):
    """Runs `script` in a fresh process and returns its standard output."""
    child = subprocess.run(
        [sys.executable, '-c', script], capture_output=True, text=True, timeout=300
    )
    assert child.returncode == 0, child.stderr[-2000:]
    return child.stdout


def measure_gpu(script):
    """Runs `script` after `DEVICE_MEMORY` in a fresh process and returns the integers it prints."""
    output = run_child(DEVICE_MEMORY + script)
    if output.strip() == 'no NVML':
        pytest.skip('no NVML to count the memory of the GPU')
    *values, least_free = (int(value) for value in output.split())
    # A full GPU raises an error on Linux, while WDDM keeps a few tens of megabytes free and places
    # the rest of the batch in shared system memory, which the count of the GPU leaves out.
    if least_free < 256 * 2**20:
        pytest.skip('the GPU was full, as other processes held most of its memory')
    return values


@pytest.mark.parametrize('assets, options', CASES)
def test_the_estimate_is_the_batch(device, assets, options):
    estimate = cuju.memory_estimate(assets, environment_count=4, device=device, **options)
    with cuju.BatchEnvironment(assets, environment_count=4, device=device, **options) as batch:
        assert batch.memory_bytes == estimate


def test_a_cpu_estimate_holds_no_local_memory():
    estimate = cuju.memory_estimate('5_vs_5_l4_r0', environment_count=2, device='cpu')
    assert isinstance(estimate, int) and estimate.local_memory_bytes == 0


def test_the_estimate_grows_by_one_world_and_its_buffers_per_environment(device):
    one, two, three = (
        cuju.memory_estimate('5_vs_5_l4_r0', environment_count=count, device=device)
        for count in (1, 2, 3)
    )
    # Each environment holds a world of a few hundred kilobytes, and the tables stay shared.
    assert three - two == two - one > 300_000


@needs_cuda
def test_a_gpu_environment_holds_a_work_area():
    one, two = (
        {
            device: cuju.memory_estimate('5_vs_5_l4_r0', environment_count=count, device=device)
            for device in ('cpu', 'cuda')
        }
        for count in (1, 2)
    )
    # Beyond the memory of a CPU environment, a GPU environment holds the work area of the kernels,
    # 12 KB, and the actions of its 4 agents.
    assert (two['cuda'] - one['cuda']) - (two['cpu'] - one['cpu']) == 12 * 1024 + 4 * 4


def test_the_estimate_raises_as_the_batch():
    with pytest.raises(ValueError, match='first index'):
        cuju.memory_estimate('5_vs_5_l1_r0', environment_count=2, first_index=-1, device='cpu')
    with pytest.raises(RuntimeError, match='agents'):
        cuju.memory_estimate('5_vs_5', agent_counts=[(9, 0)], environment_count=1, device='cpu')


@pytest.mark.skipif(not os.path.exists('/proc/self/status'), reason='no /proc')
def test_a_cpu_batch_takes_its_estimate_in_resident_memory():
    estimate = cuju.memory_estimate('5_vs_5_l4_r0', environment_count=1024, device='cpu', threads=4)
    taken = int(run_child(RESIDENT).split()[-1])
    # The CPU zeroes every buffer at its creation, so all of it is resident. The host copies of the
    # bundles and animations, the library and the alignment of each buffer add the rest. A fixed
    # thread count keeps the stacks and allocator arenas of the pool the same on every machine, and
    # the rest reaches about 90 megabytes on Arm.
    assert 0.98 * estimate <= taken <= estimate * 1.05 + 128 * 2**20


@needs_cuda
def test_torch_counts_the_estimate_of_a_gpu_batch():
    estimate = cuju.memory_estimate('5_vs_5_l4_r0', environment_count=64, device='cuda')
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    batch = cuju.BatchEnvironment(
        '5_vs_5_l4_r0', environment_count=64, device='cuda', allocator='torch'
    )
    taken = torch.cuda.memory_allocated() - before
    batch.close()
    # PyTorch rounds each request up to a multiple of 512 bytes, and the batch makes fewer than 200.
    assert estimate <= taken <= estimate + 200 * 512


@needs_cuda
def test_a_gpu_batch_takes_its_estimate_of_the_free_memory():
    estimate, taken = measure_gpu(GPU_MEMORY)
    # The driver rounds each allocation up to its pages, and other users of the GPU blur the count.
    assert abs(taken - estimate) <= 0.02 * estimate + 64 * 2**20


@needs_cuda
def test_a_first_gpu_batch_takes_the_local_memory_of_the_estimate():
    local, taken, later = measure_gpu(LOCAL_MEMORY)
    # A few kilobytes for each thread that the GPU runs at once, which the second batch finds held.
    assert local > 2**20 and later == 0
    # The driver rounds the reservation and each allocation up to its pages.
    assert abs(taken - local) <= 0.01 * local + 16 * 2**20


@needs_cuda
def test_the_kernels_reserve_little_local_memory_per_thread():
    local, threads = (int(value) for value in run_child(RESERVATION).split())
    # The lists that every lane of a warp holds alike lie in the work area of the environment, so
    # each thread of the GPU needs a few kilobytes beyond the stack that the driver keeps, where it
    # needed about 50 KB before.
    assert 0 < local <= LOCAL_BYTES_PER_THREAD * threads


@needs_cuda
def test_a_failed_reservation_names_the_local_memory_of_a_thread():
    message = run_child(RESERVATION_ERROR).strip()
    if message in ('created', 'full'):
        pytest.skip('the driver gave the reservation shared system memory, or the GPU was full')
    if 'the local memory of the kernels' not in message:
        pytest.skip('the GPU ran out of memory before the reservation')
    per_thread = int(re.search(r'kernels, [^(]*\((\d+) bytes\) for each of', message).group(1))
    # The figure is the local memory of one thread of the kernels, above the stack of 1 KB.
    assert 1024 < per_thread <= LOCAL_BYTES_PER_THREAD + 1024


@needs_cuda
def test_the_warning_counts_the_local_memory_of_a_first_gpu_batch():
    first, second = run_child(FIRST_BATCH_WARNING).splitlines()
    assert first.startswith('1 ') and 'of it for the local memory of the kernels' in first
    assert second == '0'


def test_the_warning_names_both_amounts():
    with pytest.warns(UserWarning, match=r'requested 2\.0 MiB of cuda:0.*1\.0 MiB free.*WDDM'):
        memory._warn_memory(2 * 2**20, 2**20, torch.device('cuda', 0))
    with pytest.warns(UserWarning, match=r'3\.0 MiB of cuda:0, 2\.0 MiB of it for the local'):
        memory._warn_memory(3 * 2**20, 2**20, torch.device('cuda', 0), 2 * 2**20)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        memory._warn_memory(2**20, 2**20, torch.device('cuda', 0))


@needs_cuda
def test_a_gpu_batch_beyond_the_free_memory_warns(monkeypatch):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=2, device='cuda').close()
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device=None: (2**20, 2**30))
    with pytest.warns(UserWarning, match='1.0 MiB free before its creation'):
        cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=2, device='cuda').close()


def test_a_cpu_batch_never_warns(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device=None: (0, 2**30))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        cuju.BatchEnvironment('5_vs_5_l1_r0', environment_count=2, device='cpu').close()
