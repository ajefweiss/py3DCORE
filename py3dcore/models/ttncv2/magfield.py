# -*- coding: utf-8 -*-

import numpy as np

from numba import guvectorize
from py3dcore.models.ttncv2.coordinates import _numba_jac
from py3dcore.rotqs import _numba_quaternion_rotate


# Wrappers

def h(q, b, iparams, sparams, q_xs, **kwargs):
    """Wrapper functions for calculating magnetic field vector at (q) coordinates in (s) according
    to the gold hoyle solution using the tapered torus model.

    Parameters
    ----------
    q : Union[np.ndarray, numba.cuda.cudadrv.devicearray.DeviceNDArray]
        The state parameter array.
    b : Union[np.ndarray, numba.cuda.cudadrv.devicearray.DeviceNDArray]
        Magnetic field output array.
    iparams : Union[np.ndarray, numba.cuda.cudadrv.devicearray.DeviceNDArray]
        Initial parameter array.
    sparams : Union[np.ndarray, numba.cuda.cudadrv.devicearray.DeviceNDArray]
        State parameter array.
    q_xs : Union[np.ndarray, numba.cuda.cudadrv.devicearray.DeviceNDArray]
        Array for (x) -> (s) rotational quaternions.

    Other Parameters
    ----------------
    bounded : bool
        Return zero values if measurement is taken outside of the flux rope structure.
    rng_states : Union[np.ndarray, numba.cuda.cudadrv.devicearray.DeviceNDArray]
        CUDA rng state array.
    use_cuda : bool, optional
        CUDA flag, by default False.

    Raises
    ------
    RuntimeError
        If oo method is called with gpu flag set.
    """
    if kwargs.get("use_cuda", False):
        raise NotImplementedError("CUDA functionality is not available yet")
    else:
        bounded = kwargs.get("bounded", True)

        _numba_h(q, iparams, sparams, q_xs, bounded, b)


@guvectorize([
    "void(float32[:], float32[:], float32[:], float32[:], boolean, float32[:])",
    "void(float64[:], float64[:], float64[:], float64[:], boolean, float64[:])"],
    '(i), (j), (k), (l), () -> (i)')
def _numba_h(q, iparams, sparams, q_xs, bounded, b):
    bsnp = np.empty((3,))

    (q0, q1, q2) = (q[0], q[1], q[2])

    if q0 <= 1 or bounded is False:
        (t_i, _, _, _, _, w, delta, _, _, turns, _, _, _, _, _, noise) = iparams
        (_, _, rho_0, rho_1, b_t) = sparams

        # get normal vectors
        (dr, dpsi, dphi) = _numba_jac(q0, q1, q2, rho_0, rho_1, delta, w)

        # unit normal vectors
        dr = dr / np.linalg.norm(dr)
        dpsi_norm = np.linalg.norm(dpsi)
        dpsi = dpsi / dpsi_norm
        dphi_norm = np.linalg.norm(dphi)
        dphi = dphi / dphi_norm

        br = 0

        fluxfactor = 1 / np.sin(q1 / 2)**2

        hand = turns / np.abs(turns)
        turns = np.abs(turns)
        dinv = 1 / delta
        b_10 = b_t * delta

        #h = (delta - 1)**2 / (1 + delta)**2
        #E = np.pi * (1 + delta) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))

        #t = turns * rho_1 / rho_0 * E / 2 / np.pi * np.sin(q1 / 2)**2

        bpsi = dinv * b_10 * (2 - q0**2) * fluxfactor
        bphi = 2 * dinv * hand * b_10 * q0 / (dinv**2 + 1) / turns * fluxfactor

        # magnetic field in (x)
        bsnp[0] = dr[0] * br + dpsi[0] * bpsi + dphi[0] * bphi
        bsnp[1] = dr[1] * br + dpsi[1] * bpsi + dphi[1] * bphi
        bsnp[2] = dr[2] * br + dpsi[2] * bpsi + dphi[2] * bphi

        # magnetic field in (s)
        bss = _numba_quaternion_rotate(np.array([0, bsnp[0], bsnp[1], bsnp[2]]), q_xs)

        b[0] = bss[0] + np.random.normal(0, noise)
        b[1] = bss[1] + np.random.normal(0, noise)
        b[2] = bss[2] + np.random.normal(0, noise)
    else:
        b[0] = 0
        b[1] = 0
        b[2] = 0
