# -*- coding: utf-8 -*-

"""summary.py

Implements functions for ABC-SMC summary statistics
"""

import numpy as np
import numba


def sumstat(values, reference, stype="rmse", **kwargs):
    """Summary statistic.
    Compatiable with multi-point
    """
    if stype == "rmse":
        return rmse(values, reference, **kwargs)
    elif stype == "rmse_t":
        return rmse_t(values, reference, **kwargs)
    elif stype == "norm_rmse_all":
        obsc = kwargs.pop("obsc")
        dls = np.array(kwargs.pop("dls"))
        mask = kwargs.pop("mask", None)

        varr = np.array(values)

        rmse_all = np.zeros((obsc, varr.shape[1]))

        _dl = 0
        for i in range(obsc):
            slc = slice(_dl, _dl + dls[i] + 2)
            values_i = varr[slc]
            reference_i = np.array(reference)[slc]

            normfac = np.mean(np.sqrt(np.sum(reference_i**2, axis=1)))

            if mask is not None:
                mask_i = np.array(mask)[slc]
            else:
                mask_i = None

            rmse_all[i] = rmse(values_i, reference_i, mask=mask_i) / normfac

            _dl += dls[i] + 2

        return rmse_all.T

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


def rmse_t(values, reference, mask=None, use_gpu=False):
    """Compute RMSE with magnitude between numerous generated 3DCORE profiles and a reference profile. If a
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
                _error_rmse_t(values[i], reference[i], mask[i], rmse)

            rmse = np.sqrt(rmse / len(values))

            mask_arr = np.copy(rmse)

            for i in range(len(reference)):
                _error_mask(values[i], mask[i], mask_arr)

            return mask_arr
        else:
            for i in range(len(reference)):
                _error_rmse_t(values[i], reference[i], 1, rmse)

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


@numba.njit
def _error_rmse_t(values_t, ref_t, mask, rmse):
    for i in numba.prange(len(values_t)):
        if mask == 1:           
            rmse[i] += np.sum((values_t[i] - ref_t)**2) + (np.linalg.norm(values_t[i]) - np.linalg.norm(ref_t))**2
