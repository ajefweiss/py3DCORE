# -*- coding: utf-8 -*-

"""rmse.py

RMSE summary statistics for 3DCORE magnetic field signatures.
"""

import numba
import numpy as np


def rmse(values, reference, mask=None, use_gpu=False):
    """Compute RMSE between numerous generated 3DCORE profiles and a reference profile. If a
    mask is given, profiles that are masked if their values are not non-zero where the filter is
    set to non-zero.

    Parameters
    ----------
    values : Union[list[np.ndarray], list[numba.cuda.cudadrv.devicearray.DeviceNDArray]]
        List of magnetic field outputs.
    reference : Union[np.ndarray, numba.cuda.cudadrv.devicearray.DeviceNDArray]
        Reference magnetic field measurements.
    mask : np.ndarray, optional
        Mask array, by default None
    use_gpu : bool, optional
        GPU flag, by default False
    """
    if use_gpu:
        raise NotImplementedError
    else:
        rmse = np.zeros(len(values[0]))

        if mask is not None:
            for i in range(len(reference)):
                _error_rmse(values[i], reference[i], mask[i], rmse)

            rmse = np.sqrt(rmse / len(values))

            mask_arr = np.copy(rmse)

            for i in range(len(reference)):
                _error_mask(values[i], mask[i], mask_arr)

            return mask_arr
        else:
            for i in range(len(reference)):
                _error_rmse(values[i], reference[i], 1, rmse)

            rmse = np.sqrt(rmse / len(values))

            return rmse


@numba.njit
def _error_mask(values_t, mask, rmse):
    for i in numba.prange(len(values_t)):
        _v = np.sum(values_t[i]**2)
        if (_v > 0 and mask == 0) or (_v == 0 and mask != 0):
            rmse[i] = np.inf


@numba.njit
def _error_rmse(values_t, ref_t, mask, rmse):
    for i in numba.prange(len(values_t)):
        if mask == 1:
            rmse[i] += np.sum((values_t[i] - ref_t)**2)
