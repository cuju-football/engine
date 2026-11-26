# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Imports cuju's port of GRF under GRF's own names, `gfootball` and `gfootball_engine`.

Python's `site` activates the alias at startup through `cuju_gfootball.pth`, and an import of this
package activates it as well, for an interpreter started without `site`.
"""

from cuju.gfootball_alias import _startup
from cuju.gfootball_alias._version import __version__

__all__ = ['__version__', 'active', 'status']


def active() -> bool:
    """Returns whether GRF's names import cuju's port."""
    return not _startup.reason()


def status() -> str:
    """Returns a sentence that says whether the alias is active, and why it is not."""
    found = _startup.reason()
    if not found:
        return (
            'cuju-gfootball is active: gfootball, gfootball_engine and _gameplayfootball import '
            'cuju.gfootball and cuju.gfootball_engine.'
        )
    return f'cuju-gfootball is inactive: {found}.'
