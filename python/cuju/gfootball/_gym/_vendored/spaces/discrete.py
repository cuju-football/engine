# Copyright (c) 2016 OpenAI (https://openai.com)
# Modified by the cuju Authors in 2026, in the lines that ci/port_gfootball.py lists.
# SPDX-License-Identifier: MIT

import numpy as np
from .space import Space


class Discrete(Space):
    r"""A discrete space in :math:`\{ 0, 1, \\dots, n-1 \}`.

    Example::

        >>> Discrete(2)

    """

    def __init__(self, n, seed=None):
        assert n >= 0
        self.n = n
        super(Discrete, self).__init__((), np.int64, seed)

    def sample(self):
        return self.np_random.randint(self.n)

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (
            x.dtype.char in np.typecodes["AllInteger"] and x.shape == ()
        ):
            as_int = int(x)
        else:
            return False
        return as_int >= 0 and as_int < self.n

    def __repr__(self):
        return "Discrete(%d)" % self.n

    def __eq__(self, other):
        return isinstance(other, Discrete) and self.n == other.n
