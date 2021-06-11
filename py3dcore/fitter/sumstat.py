# -*- coding: utf-8 -*-

"""summary.py

Implements functions for ABC-SMC summary statistics
"""

import numpy as np
import numba

from typing import Any, Optional


def sumstat(values: np.ndarray, reference: np.ndarray, stype: str = "norm_rmse", **kwargs: Any) -> np.ndarray:
    if stype == "norm_rmse":
        data_l = np.array(kwargs.pop("data_l"))        
        length = kwargs.pop("length")
        mask = kwargs.pop("mask", None)

        varr = np.array(values)

        rmse_all = np.zeros((length, varr.shape[1]))

        _dl = 0
        for i in range(length):
            slc = slice(_dl, _dl + data_l[i] + 2)
            values_i = varr[slc]
            reference_i = np.array(reference)[slc]

            normfac = np.mean(np.sqrt(np.sum(reference_i**2, axis=1)))

            if mask is not None:
                mask_i = np.array(mask)[slc]
            else:
                mask_i = None

            rmse_all[i] = rmse(values_i, reference_i, mask=mask_i) / normfac

            _dl += data_l[i] + 2

        return rmse_all.T
    else:
        raise NotImplementedError


def rmse(values: np.ndarray, reference: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
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
def _error_mask(values_t: np.ndarray, mask: np.ndarray, rmse: np.ndarray) -> None:
    for i in numba.prange(len(values_t)):
        _v = np.sum(values_t[i]**2)
        if (_v > 0 and mask == 0) or (_v == 0 and mask != 0):
            rmse[i] = np.inf


@numba.njit
def _error_rmse(values_t: np.ndarray, ref_t: np.ndarray, mask: np.ndarray, rmse: np.ndarray) -> None:
    for i in numba.prange(len(values_t)):
        if mask == 1:
            rmse[i] += np.sum((values_t[i] - ref_t)**2)
