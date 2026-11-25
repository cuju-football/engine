# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""The two exceptions that only `cuju.gfootball_engine` raises."""

from __future__ import annotations

from ..errors import Error

__all__ = ['SignatureError', 'TrackerError']


class SignatureError(Error, TypeError):
    """Arguments that a function of `cuju.gfootball_engine` does not take.

    Its message is the one of GRF's `Boost.Python.ArgumentError`, which is a `TypeError` as well.
    """


class TrackerError(Error, RuntimeError):
    """Two engines that GRF's tracker plays in lockstep and whose states differ.

    GRF ends the process there, and cuju raises this in both threads.
    """
