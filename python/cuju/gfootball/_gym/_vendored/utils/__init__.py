# Copyright (c) 2016 OpenAI (https://openai.com)
# Modified by the cuju Authors in 2026, in the lines that ci/port_gfootball.py lists.
# SPDX-License-Identifier: MIT

"""A set of common utilities used within the environments. These are
not intended as API functions, and will not remain stable over time.
"""

# These submodules should not have any import-time dependencies.
# We want this since we use `utils` during our import-time sanity checks
# that verify that our dependencies are actually present.
from .colorize import colorize
from .ezpickle import EzPickle
