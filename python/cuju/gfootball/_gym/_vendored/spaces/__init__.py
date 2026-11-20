# Copyright (c) 2016 OpenAI (https://openai.com)
# Modified by the cuju Authors in 2026, in the lines that ci/port_gfootball.py lists.
# SPDX-License-Identifier: MIT

from cuju.gfootball._gym._vendored.spaces.space import Space
from cuju.gfootball._gym._vendored.spaces.box import Box
from cuju.gfootball._gym._vendored.spaces.discrete import Discrete
from cuju.gfootball._gym._vendored.spaces.multi_discrete import MultiDiscrete
from cuju.gfootball._gym._vendored.spaces.multi_binary import MultiBinary
from cuju.gfootball._gym._vendored.spaces.tuple import Tuple
from cuju.gfootball._gym._vendored.spaces.dict import Dict

from cuju.gfootball._gym._vendored.spaces.utils import flatdim
from cuju.gfootball._gym._vendored.spaces.utils import flatten_space
from cuju.gfootball._gym._vendored.spaces.utils import flatten
from cuju.gfootball._gym._vendored.spaces.utils import unflatten

__all__ = [
    "Space",
    "Box",
    "Discrete",
    "MultiDiscrete",
    "MultiBinary",
    "Tuple",
    "Dict",
    "flatdim",
    "flatten_space",
    "flatten",
    "unflatten",
]
