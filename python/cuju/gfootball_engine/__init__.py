# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""GRF's package `gfootball_engine`, whose module `_gameplayfootball` cuju's engine implements.

`cuju.gfootball` imports it as GRF's Python imports `gfootball_engine`. The module keeps GRF's
names and the environment variables that GRF's package sets, and leaves `sys.path` as it is, so
that GRF's own `_gameplayfootball` is never shadowed where both are installed.
"""

from __future__ import absolute_import, division, print_function  # noqa: UP010, GRF's names

import logging  # noqa: F401, a name of GRF's package
import os
import sys  # noqa: F401, a name of GRF's package

game_path = os.path.dirname(os.path.abspath(__file__))
gfootball_dir = os.path.dirname(os.path.abspath(__file__))
font_file = os.path.join(gfootball_dir, 'fonts/AlegreyaSansSC-ExtraBold.ttf')
if 'MESA_GL_VERSION_OVERRIDE' not in os.environ:
    os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.2'
if 'MESA_GLSL_VERSION_OVERRIDE' not in os.environ:
    os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '150'
if 'GFOOTBALL_FONT' not in os.environ:
    os.environ['GFOOTBALL_FONT'] = font_file
data_dir = os.path.join(gfootball_dir, 'data')
if 'GFOOTBALL_DATA_DIR' not in os.environ:
    os.environ['GFOOTBALL_DATA_DIR'] = data_dir

from ._errors import SignatureError, TrackerError  # noqa: E402, F401
from ._gameplayfootball import *  # noqa: E402, F403
