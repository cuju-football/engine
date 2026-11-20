# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Prints whether the alias is active, `python -m cuju_gfootball`, with exit status 1 where not."""

import sys

import cuju_gfootball

print(cuju_gfootball.status())
sys.exit(0 if cuju_gfootball.active() else 1)
