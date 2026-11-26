# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""Ends a child process and everything that it started, on Linux, macOS and Windows."""

import contextlib
import os
import signal
import subprocess
import sys
from typing import Any

WINDOWS = os.name == 'nt'
_CREATE_SUSPENDED = 0x00000004


def popen_options() -> dict[str, Any]:
    """Returns how to start a process that the tree will end."""
    if WINDOWS:
        # The process waits suspended until it is in its job, so it starts nothing outside it.
        return {'creationflags': _CREATE_SUSPENDED}
    return {'start_new_session': True}


class _Job:
    """A job object of Windows whose close ends every process in it."""

    _KILL_ON_CLOSE = 0x2000

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        class Basic(ctypes.Structure):
            _fields_ = (
                ('per_process_time', ctypes.c_int64),
                ('per_job_time', ctypes.c_int64),
                ('flags', wintypes.DWORD),
                ('minimum_working_set', ctypes.c_size_t),
                ('maximum_working_set', ctypes.c_size_t),
                ('active_processes', wintypes.DWORD),
                ('affinity', ctypes.c_size_t),
                ('priority', wintypes.DWORD),
                ('scheduling', wintypes.DWORD),
            )

        class Counters(ctypes.Structure):
            _fields_ = tuple((name, ctypes.c_uint64) for name in 'abcdef')

        class Extended(ctypes.Structure):
            _fields_ = (
                ('basic', Basic),
                ('io', Counters),
                ('process_memory', ctypes.c_size_t),
                ('job_memory', ctypes.c_size_t),
                ('peak_process_memory', ctypes.c_size_t),
                ('peak_job_memory', ctypes.c_size_t),
            )

        self._wintypes = wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        self._kernel = kernel
        self.handle = kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise OSError('cannot make a job object')
        info = Extended()
        info.basic.flags = self._KILL_ON_CLOSE
        if not kernel.SetInformationJobObject(
            self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)
        ):
            self.close()
            raise OSError('cannot limit a job object')

    def assign(self, process_handle: int) -> None:
        if not self._kernel.AssignProcessToJobObject(self.handle, self._wintypes.HANDLE(process_handle)):
            raise OSError('cannot put the process in a job object')

    def terminate(self) -> None:
        if self.handle:
            self._kernel.TerminateJobObject(self.handle, 1)

    def close(self) -> None:
        if self.handle:
            self._kernel.CloseHandle(self.handle)
            self.handle = None


def _resume(handle: int) -> None:
    import ctypes

    ctypes.windll.ntdll.NtResumeProcess(ctypes.c_void_p(handle))


class Tree:
    """The tree of one started process, ended with `kill` or when the host ends."""

    def __init__(self, process: subprocess.Popen) -> None:
        """Takes a process that `popen_options` started."""
        self.process = process
        self._job: _Job | None = None
        if WINDOWS:
            try:
                self._job = _Job()
                self._job.assign(int(process._handle))
            except OSError:
                if self._job is not None:
                    self._job.close()
                self._job = None
            finally:
                _resume(int(process._handle))

    def kill(self) -> None:
        """Ends the process and every process of its session or job."""
        if self._job is not None:
            self._job.terminate()
        if not WINDOWS:
            # The session of the process holds whatever it started and did not move.
            with contextlib.suppress(OSError):
                os.killpg(self.process.pid, signal.SIGKILL)
        if self.process.poll() is None:
            with contextlib.suppress(OSError):
                self.process.kill()

    def close(self) -> None:
        """Ends the tree and lets go of the job."""
        self.kill()
        if self._job is not None:
            self._job.close()
            self._job = None


def base_interpreter(python: str) -> tuple[str, dict[str, str]]:
    """Returns the interpreter to start and the variables it needs, which keep a venv on Windows.

    Windows starts a virtual environment's interpreter through a launcher process that holds the
    handles of the process and the tree. The base interpreter starts directly instead, with
    `__PYVENV_LAUNCHER__` naming the environment's, so the same packages load.
    """
    if not WINDOWS:
        return python, {}
    if os.path.normcase(os.path.abspath(python)) == os.path.normcase(os.path.abspath(sys.executable)):
        base = getattr(sys, '_base_executable', python)
        if os.path.normcase(base) != os.path.normcase(python):
            return base, {'__PYVENV_LAUNCHER__': python}
        return python, {}
    folder = os.path.dirname(os.path.abspath(python))
    for configuration in (os.path.join(folder, 'pyvenv.cfg'), os.path.join(os.path.dirname(folder), 'pyvenv.cfg')):
        if os.path.isfile(configuration):
            with open(configuration, encoding='utf-8') as handle:
                for line in handle:
                    key, _, value = line.partition('=')
                    if key.strip().lower() == 'home':
                        base = os.path.join(value.strip(), os.path.basename(python))
                        if os.path.isfile(base):
                            return base, {'__PYVENV_LAUNCHER__': os.path.abspath(python)}
            break
    return python, {}
