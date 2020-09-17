# -*- coding: utf-8 -*-

"""_extnumba.py

Utility functions for Numba.
"""

import numba
import numpy as np


def set_random_seed(seed):
    """Sets python & numba seed to same value.
    """
    np.random.seed(seed)
    _numba_set_random_seed(seed)


@numba.njit
def _numba_set_random_seed(seed):
    np.random.seed(seed)
