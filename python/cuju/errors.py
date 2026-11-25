# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""The exceptions that cuju raises, each also a subclass of the built-in type it stands for."""

from __future__ import annotations

__all__ = [
    'ArgumentError',
    'ChoiceError',
    'ClosedError',
    'ConfigurationError',
    'DeviceError',
    'Error',
    'LibraryError',
    'MissingFileError',
    'StateError',
    'UndefinedPathError',
]


class Error(Exception):
    """The base of every exception that cuju raises for a failure of its own."""

    __str__ = Exception.__str__


class ConfigurationError(Error, ValueError):
    """An option of a batch or of a scenario that cuju refuses."""


class ChoiceError(ConfigurationError, KeyError):
    """A name outside the valid ones, such as a representation, an action set or a device."""


class ArgumentError(Error, ValueError, IndexError):
    """An argument of a call that cuju refuses."""


class StateError(Error, ValueError, RuntimeError):
    """A state that does not load into an environment, or a batch that has no GRF state."""


class ClosedError(Error, RuntimeError):
    """A call on a batch after `close`."""


class DeviceError(Error, RuntimeError):
    """A device that cannot run the batch, or a failure of CUDA."""


class UndefinedPathError(Error, RuntimeError):
    """An environment that reached a path that GRF leaves undefined."""


class LibraryError(Error, RuntimeError):
    """A failure that the library reports, or a call that has no output to give."""


class MissingFileError(Error, FileNotFoundError):
    """A scenario file, a bundle, a library or another file that does not exist."""


_CONFIGURATION_MARKERS = (
    ' agents, the players of the ',
    ' agents, all the controllable players of a side',
    ' agents in ',
)


def _from_library(message: str) -> Error:
    if 'reached a path that GRF leaves undefined' in message:
        return UndefinedPathError(message)
    if message.startswith('cuda') or 'CUDA kernel' in message:
        return DeviceError(message)
    if any(marker in message for marker in _CONFIGURATION_MARKERS):
        return ConfigurationError(message)
    return LibraryError(message)
