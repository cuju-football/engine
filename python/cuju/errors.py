# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""The exceptions that cuju raises, each also a subclass of the built-in type it stands for.

Every class derives from `Error` and from a built-in exception, so a clause such as
`except ValueError:` catches it as well. `ChoiceError` is also a `KeyError`, and `ArgumentError`
is both a `ValueError` and an `IndexError`. `SignatureError` and `TrackerError`, which only
`cuju.gfootball_engine` raises, live there.
"""

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

    # `KeyError` quotes its message, so every class here prints the message as it is.
    __str__ = Exception.__str__


class ConfigurationError(Error, ValueError):
    """An option of a batch or of a scenario that cuju refuses."""


class ChoiceError(ConfigurationError, KeyError):
    """A name outside the valid ones, such as a representation, an action set or a device."""


class ArgumentError(Error, ValueError, IndexError):
    """An argument of a call that cuju refuses.

    Such an argument is actions of the wrong shape, an action outside the action set, or an
    environment index outside the batch.
    """


class StateError(Error, ValueError, RuntimeError):
    """A state that does not load into an environment, or a batch that has no GRF state."""


class ClosedError(Error, RuntimeError):
    """A call on a batch after `close`."""


class DeviceError(Error, RuntimeError):
    """A device that cannot run the batch, or a failure of CUDA.

    Such a device is a missing CUDA GPU, a GPU index past the last, or a processor without AVX2,
    BMI2 or FMA.
    """


class UndefinedPathError(Error, RuntimeError):
    """An environment that reached a path that GRF leaves undefined."""


class LibraryError(Error, RuntimeError):
    """A failure that the library reports, or a call that has no output to give.

    Such a call is a step before the first reset, or `raw()` and `frame()` on a batch that writes
    no raw observation or renders no frames.
    """


class MissingFileError(Error, FileNotFoundError):
    """A scenario file, a bundle, a library or another file that does not exist."""


# The parts of the library's messages that name an option of the batch that it refuses.
_CONFIGURATION_MARKERS = (
    ' agents, the players of the ',
    ' agents, all the controllable players of a side',
    ' agents in ',
)


def _from_library(message: str) -> Error:
    """Returns the exception for a failure message of the library."""
    if 'reached a path that GRF leaves undefined' in message:
        return UndefinedPathError(message)
    # A failed CUDA call starts the message with its name, such as `cudaMalloc`, and a GPU without
    # a kernel is a device fault.
    if message.startswith('cuda') or 'CUDA kernel' in message:
        return DeviceError(message)
    # The library refuses more agents than a team has players, and a scenario that controls all
    # its players with other counts, as options of the batch.
    if any(marker in message for marker in _CONFIGURATION_MARKERS):
        return ConfigurationError(message)
    return LibraryError(message)
