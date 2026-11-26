# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""Runs a tree of controllers in another process, which speaks the protocol of `client.py`."""

import contextlib
import os
import queue
import struct
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any


from ..errors import ConfigurationError
from . import _tree, client, protocol
from ._contract import Actions, Controller, ControllerError, Observation, Spec, normalize

LOAD_SECONDS = 300.0
CLOSE_SECONDS = 5.0
_FIELDS = ('data', 'first', 'active', 'left_players', 'right_players', 'previous', 'step', 'side')


def source() -> bytes:
    """Returns the bytes of `client.py`, which `cuju agents client` prints."""
    return Path(client.__file__).read_bytes()


class Process(Controller):
    """Runs its inner controller in a new process, with any interpreter or command."""

    asynchronous = True

    def __init__(
        self,
        inner: Any,
        python: str | None = None,
        command: list[str] | None = None,
        prefix: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float = LOAD_SECONDS,
        stderr: Any = None,
        control: dict[str, Any] | None = None,
    ) -> None:
        """Holds what to run and how, and starts the process when the first use needs it.

        Args:
            inner: The unbuilt tree below, a node, a path or an entry, which the child builds.
            python: The interpreter, by default this one.
            command: The whole command, in which `{stub}` stands for the code that starts the
                client. A command without it runs as it is and speaks the protocol itself.
            prefix: A command put in front, such as `prlimit` or `docker run -i`.
            cwd: The working folder of the child.
            env: Variables added to the child's environment.
            timeout: Seconds to wait for the load and for each answer.
            stderr: Where the child's standard error goes: `None` for this process's, a file
                or a descriptor, a path, or a function of each line as bytes.
            control: The control that the host reads, by default the inner's.
        """
        if isinstance(inner, Controller):
            raise ConfigurationError('expected an unbuilt inner tree, found a built controller')
        from ._load import node_of

        self.inner = node_of(inner)
        if not isinstance(self.inner, dict):
            raise ConfigurationError(f'expected an unbuilt inner tree, found {inner!r}')
        if command is not None and python is not None:
            raise ConfigurationError('expected python or command, found both')
        self.python = python
        self.command = None if command is None else list(command)
        self.prefix = [] if prefix is None else list(prefix)
        self.cwd = cwd
        self.env = {} if env is None else dict(env)
        self.timeout = float(timeout)
        self.stderr = stderr
        self._control = control
        self._ready: dict[str, Any] | None = None
        self.process: subprocess.Popen | None = None
        self._tree: _tree.Tree | None = None
        self._frames: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._side = 0
        self._step = 0

    # What the child declared.

    @property
    def representation(self) -> str:  # type: ignore[override]
        """The representation of the inner controller, which the child reports."""
        return self._load()['representation']

    @property
    def control(self) -> dict[str, Any] | None:  # type: ignore[override]
        """The control of this node, else the inner's."""
        if self._control is not None:
            return self._control
        return self._load()['control']

    @control.setter
    def control(self, value: dict[str, Any] | None) -> None:
        self._control = value

    @property
    def pid(self) -> int | None:
        """The identifier of the child, for user parts that limit it."""
        return None if self.process is None else self.process.pid

    # Starting.

    def _argv(self) -> tuple[list[str], dict[str, str], bool]:
        stub = client.STUB
        extra: dict[str, str] = {}
        if self.command is not None:
            argv = [part.replace('{stub}', stub) for part in self.command]
            return [*self.prefix, *argv], extra, any('{stub}' in part for part in self.command)
        python, extra = _tree.base_interpreter(self.python or sys.executable)
        return [*self.prefix, python, '-X', 'utf8', '-c', stub], extra, True

    def _spawn(self) -> None:
        argv, extra, stubbed = self._argv()
        sink = self.stderr
        target: Any = None
        callback = None
        if callable(sink) and not hasattr(sink, 'fileno'):
            target, callback = subprocess.PIPE, sink
        elif isinstance(sink, (str, os.PathLike)):
            target = open(sink, 'ab')  # noqa: SIM115, closed after the spawn
        elif sink is not None:
            target = sink
        environment = {**os.environ, **extra, **self.env}
        try:
            self.process = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=target,
                close_fds=True,
                cwd=self.cwd,
                env=environment,
                **_tree.popen_options(),
            )
        except OSError as error:
            raise ControllerError(f'expected to start {argv[0]!r}, found {error}', 'start') from None
        finally:
            if isinstance(sink, (str, os.PathLike)) and target is not None:
                target.close()
        self._tree = _tree.Tree(self.process)
        threading.Thread(target=self._read, daemon=True, name='cuju-agents-pipe').start()
        if callback is not None:
            threading.Thread(
                target=self._copy, args=(callback,), daemon=True, name='cuju-agents-stderr'
            ).start()
        if stubbed:
            code = source()
            self._write(struct.pack('<Q', len(code)) + code)

    def _copy(self, callback: Any) -> None:
        stream = self.process.stderr  # type: ignore[union-attr]
        for line in iter(stream.readline, b''):
            with contextlib.suppress(Exception):
                callback(line)

    def _read(self) -> None:
        stream = self.process.stdout  # type: ignore[union-attr]
        while True:
            try:
                self._frames.put(('frame', protocol.read(stream)))
            except EOFError as error:
                with contextlib.suppress(subprocess.TimeoutExpired):
                    self.process.wait(2)  # type: ignore[union-attr]
                self._frames.put(('end', str(error)))
                return
            except (OSError, ValueError) as error:
                self._frames.put(('end', f'the answers did not read, {error}'))
                return

    def _write(self, data: bytes) -> None:
        try:
            stream = self.process.stdin  # type: ignore[union-attr]
            stream.write(data)
            stream.flush()
        except (OSError, ValueError) as error:
            raise ControllerError(f'the process ended, {error}', 'closed') from None

    def _send(self, kind: str, value: Any = None) -> None:
        self._write(protocol.encode(kind, value))

    def _next(self, timeout: float | None) -> dict[str, Any] | None:
        try:
            what, value = self._frames.get(timeout=timeout)
        except queue.Empty:
            return None
        if what == 'end':
            self._frames.put((what, value))
            code = None if self.process is None else self.process.poll()
            raise ControllerError(
                f'the process ended before it answered, with exit code {code}, {value}', 'closed'
            )
        if value['kind'] == 'error':
            raise ControllerError(
                f'the child raised {value.get("message")}\n{value.get("traceback", "")}'.rstrip(),
                value.get('reason', 'crash'),
            )
        return value

    def _load(self) -> dict[str, Any]:
        if self._ready is not None:
            return self._ready
        self._spawn()
        self._send('hello', {'inner': self.inner})
        frame = self._next(self.timeout)
        if frame is None:
            self.close()
            raise ControllerError(f'no answer to hello within {self.timeout:g} s', 'timeout')
        if frame['kind'] != 'ready':
            raise ControllerError(f'expected ready, found {frame["kind"]}', 'protocol')
        value = frame['value']
        value['control'] = None if value['control'] is None else normalize(value['control'])
        self._ready = value
        return value

    # The contract.

    def start(self, spec: Spec) -> None:
        """Starts the session of the inner controller in the child."""
        self._load()
        self._side = spec.side
        self._step = 0
        self._send('start', spec.to_dict())
        frame = self._next(self.timeout)
        if frame is None:
            self.close()
            raise ControllerError(f'no answer to start within {self.timeout:g} s', 'timeout')

    def send(self, observation: Observation) -> None:
        """Sends the observation to the child, as numpy arrays, and returns."""
        value = {name: getattr(observation, name) for name in _FIELDS}
        value['meta'] = observation.meta
        value['environments'] = observation.environments
        try:
            self._send('act', value)
        except protocol.FrameError as error:
            raise ControllerError(str(error), 'shape', self._side) from None
        self._step += 1

    def receive(self, timeout: float) -> Actions | None:
        """Returns the child's answer, or `None` if it is not there in `timeout` seconds."""
        frame = self._next(timeout)
        if frame is None:
            return None
        if frame['kind'] != 'actions':
            raise ControllerError(f'expected actions, found {frame["kind"]}', 'protocol')
        value = frame['value']
        return Actions(value['actions'], value['extras'])

    def act(self, observation: Observation) -> Actions:
        """Sends and waits for the answer, up to `timeout` seconds."""
        self.send(observation)
        answer = self.receive(self.timeout)
        if answer is None:
            self.terminate()
            raise ControllerError(
                f'no answer within {self.timeout:g} s', 'timeout', self._side, self._step
            )
        return answer

    def terminate(self) -> None:
        """Ends the child and everything it started."""
        if self._tree is not None:
            self._tree.kill()

    def close(self) -> None:
        """Asks the child to end, then ends its whole tree."""
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            with contextlib.suppress(ControllerError, OSError, ValueError):
                self._send('close')
                process.stdin.close()  # type: ignore[union-attr]
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(CLOSE_SECONDS)
        if self._tree is not None:
            self._tree.close()
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(10)
        for stream in (process.stdin, process.stdout, process.stderr):
            with contextlib.suppress(OSError, ValueError, AttributeError):
                stream.close()  # type: ignore[union-attr]
        self.process = None
        self._tree = None
        self._ready = None
        self._frames = queue.Queue()


__all__ = ['Process', 'source']
