# Copyright (c) 2016 OpenAI (https://openai.com)
# Modified by the cuju Authors in 2026, in the lines that ci/make/port_gfootball.py lists.
# SPDX-License-Identifier: MIT

import numpy as np


def json_encode_np(obj):
    """
    JSON can't serialize numpy types, so convert to pure python
    """
    if isinstance(obj, np.ndarray):
        return list(obj)
    elif isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.int8):
        return int(obj)
    elif isinstance(obj, np.int16):
        return int(obj)
    elif isinstance(obj, np.int32):
        return int(obj)
    elif isinstance(obj, np.int64):
        return int(obj)
    else:
        return obj
