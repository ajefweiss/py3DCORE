# -*- coding: utf-8 -*-

"""_extmath.py

Extra math functions.
"""

import numba.cuda as cuda
import numpy as np

from numba import guvectorize


def cholesky(mat, use_gpu=False):
    """Cholesky Decomposition.

    Uses the LDL variant of the Cholesky decomposition to compute the lower triangular matrix of a
    positiv-semidefinite matrix.

    Parameters
    ----------
    mat : np.ndarray
        Single or multiple positive semi-definite matrix/matrices.
    use_gpu : bool
        GPU flag, by default False.

    Returns
    -------
    np.ndarray
        Lower triangular matrix/matrices.
    """
    if use_gpu and cuda.is_available():
        raise NotImplementedError
    else:
        return _numba_ldl_lsqrtd(mat)


@guvectorize([
    "void(float32[:, :], float32[:, :])",
    "void(float64[:, :], float64[:, :])"],
    '(n, n) -> (n, n)')
def _numba_ldl_lsqrtd(mat, res):
    n = mat.shape[0]

    _lmat = np.identity(n)
    _dmat = np.zeros((n, n))

    for i in range(n):
        _dmat[i, i] = mat[i, i] - np.sum(_lmat[i, :i]**2 * np.diag(_dmat)[:i])

        for j in range(i + 1, n):
            if _dmat[i, i] == 0:
                _lmat[i, i] = 0
            else:
                _lmat[j, i] = mat[j, i] - np.sum(_lmat[j, :i] * _lmat[i, :i] * np.diag(_dmat)[:i])
                _lmat[j, i] /= _dmat[i, i]

    res[:] = np.dot(_lmat, np.sqrt(_dmat))[:]
