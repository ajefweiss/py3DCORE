# -*- coding: utf-8 -*-

"""util.py

3DCORE utility functions.
"""

import datetime
import numba
import numpy as np
import py3dcore
import sys

from numba import guvectorize
from scipy.signal import detrend, welch
from typing import Sequence, Tuple


@guvectorize([
    "void(float32[:, :], float32[:, :])",
    "void(float64[:, :], float64[:, :])"],
    '(n, n) -> (n, n)')
def ldl_decomp(mat: np.ndarray, res: np.ndarray) -> None:
    """Computes the LDL decomposition, returns L.sqrt(D).
    """
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


def mag_fft(dt: Sequence[datetime.datetime], bdt: np.ndarray, sampling_freq: int) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the mean power spectrum distribution from a magnetic field measurements over all three vector components.

    Note: Assumes that P(k) is the same for all three vector components.
    """
    n_s = int(((dt[-1] - dt[0]).total_seconds() / 3600) - 1)
    n_perseg = np.min([len(bdt), 256])

    p_bX = detrend(bdt[:, 0], type="linear", bp=n_s)
    p_bY = detrend(bdt[:, 1], type="linear", bp=n_s)
    p_bZ = detrend(bdt[:, 2], type="linear", bp=n_s)

    _,  wX = welch(p_bX, fs=1 / sampling_freq, nperseg=n_perseg)
    _,  wY = welch(p_bY, fs=1 / sampling_freq, nperseg=n_perseg)
    wF, wZ = welch(p_bZ, fs=1 / sampling_freq, nperseg=n_perseg)

    wS = (wX + wY + wZ) / 3

    # convert P(k) into suitable form for fft
    fF = np.fft.fftfreq(len(p_bX), d=sampling_freq)
    fS = np.zeros((len(fF)))

    for i in range(len(fF)):
        k = np.abs(fF[i])
        fS[i] = np.sqrt(wS[np.argmin(np.abs(k - wF))])

    return fF, fS


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    _numba_set_random_seed(seed)


@numba.njit
def _numba_set_random_seed(seed: int) -> None:
    np.random.seed(seed)
