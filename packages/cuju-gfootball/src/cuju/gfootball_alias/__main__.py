# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Prints whether the alias is active, with exit status 1 where not."""

import sys

import cuju.gfootball_alias

print(cuju.gfootball_alias.status())
sys.exit(0 if cuju.gfootball_alias.active() else 1)
