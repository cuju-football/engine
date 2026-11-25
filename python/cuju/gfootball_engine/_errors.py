# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ..errors import Error

__all__ = ['SignatureError', 'TrackerError']


class SignatureError(Error, TypeError):
    """Arguments that a function of `cuju.gfootball_engine` does not take."""


class TrackerError(Error, RuntimeError):
    """Two engines that GRF's tracker plays in lockstep and whose states differ."""
