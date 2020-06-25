# -*- coding: utf-8 -*-

import numba
import numpy as np


# Wrappers

def p(t, iparams, sparams, use_gpu=False):
    """Wrapper function for propagating the flux rope state parameters.

    Parameters
    ----------
    t : float
        Datetime timestamp,
    iparams : Union[np.ndarray, numba.cuda.cudadrv.devicearray.DeviceNDArray]
        Initial parameters array.
    sparams : Union[np.ndarray, numba.cuda.cudadrv.devicearray.DeviceNDArray]
        State parameters array.
    use_gpu : bool
        GPU flag, by default False.
    """
    if use_gpu:
        raise NotImplementedError
    else:
        _numba_p(t, iparams, sparams)


@numba.njit
def _numba_p(t, iparams, sparams):
    for i in range(0, len(iparams)):
        (t_i, _, _, _, d, _, r, v, _, _, b_i, bg_d, bg_v, _) = iparams[i]

        # rescale parameters
        bg_d = bg_d * 1e-7
        r = r * 695510

        dt = t - t_i
        dv = v - bg_v

        bg_sgn = int(-1 + 2 * int(v > bg_v))

        rt = (bg_sgn / bg_d * np.log1p(bg_sgn *
                                       bg_d * dv * dt) + bg_v * dt + r) / 1.496e8
        vt = dv / (1 + bg_sgn * bg_d * dv * dt) + bg_v

        rho_1 = d * (rt ** 1.14) / 2
        rho_0 = (rt - rho_1) / 2
        b_t = b_i * (2 * rho_0) ** (-1.64)

        sparams[i, 0] = t
        sparams[i, 1] = vt
        sparams[i, 2] = rho_0
        sparams[i, 3] = rho_1
        sparams[i, 4] = b_t
