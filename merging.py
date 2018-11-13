#!/usr/bin/env python3

import numpy as np
import numpy.ma as ma


def nz_percentile(A, q):
    """calculate np.percentile(...,q) on an array's nonzero elements only"""

    if ma.is_masked(A):
        A = A.filled(0)

    return np.percentile(A[A > 0], q)
