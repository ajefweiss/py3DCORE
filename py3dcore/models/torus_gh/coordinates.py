# -*- coding: utf-8 -*-

import math
import numba
import numpy as np

from numba import guvectorize
from py3dcore.quaternions import _numba_quaternion_rotate


# Wrappers

def g(q, s, iparams, sparams, q_xs, use_gpu=False):
    """Wrapper function for transforming (q) coordinates to (s) coordinates using the tapered torus
    global shape.

    Parameters
    ----------
    q : Union[np.ndarray, numba.cuda.cudadrv.devicearray.DeviceNDArray]
        The (q) coordinate array.
    s : Union[np.ndarray, numba.cuda.cudadrv.devicearray.DeviceNDArray]
        The (s) coordinate array.
    sparams : Union[np.ndarray, numba.cuda.cudadrv.devicearray.DeviceNDArray]
        The state parameter array.
    qs_xs : Union[np.ndarray, numba.cuda.cudadrv.devicearray.DeviceNDArray]
        Array for (x) -> (s) rotational quaternions.
    use_gpu : bool
        GPU flag, by default False.

    Raises
    ------
    RuntimeError
        If oo method is called with gpu flag set.
    """
    if use_gpu:
        raise NotImplementedError
    else:
        _numba_g(q, iparams, sparams, q_xs, s)


def f(s, q, iparams, sparams, q_sx, use_gpu=False):
    """Wrapper function for transforming (s) coordinates to (q) coordinates using the tapered torus
    global shape.

    Parameters
    ----------
    s : Union[np.ndarray, numba.cuda.cudadrv.devicearray.DeviceNDArray]
        The (s) coordinate array.
    q : Union[np.ndarray, numba.cuda.cudadrv.devicearray.DeviceNDArray]
        The (q) coordinate array.
    sparams : Union[np.ndarray, numba.cuda.cudadrv.devicearray.DeviceNDArray]
        The state parameter array.
    qs_sx : Union[np.ndarray, numba.cuda.cudadrv.devicearray.DeviceNDArray]
        Array for (s) -> (x) rotational quaternions.
    use_gpu : bool
        GPU flag, by default False.

    Raises
    ------
    RuntimeError
        if oo method is called in gpu mode
    """
    if use_gpu:
        raise NotImplementedError
    else:
        _numba_f(s, iparams, sparams, q_sx, q)


@guvectorize([
    "void(float32[:], float32[:], float32[:], float32[:], float32[:])",
    "void(float64[:], float64[:], float64[:], float64[:], float64[:])"],
    '(i), (j), (k), (l) -> (i)')
def _numba_g(q, iparams, sparams, q_xs, s):
    (q0, q1, q2) = q
    delta = iparams[5]
    (_, _, rho_0, rho_1, _) = sparams

    x = np.array([
        0,
        -(rho_0 + q0 * rho_1 * np.cos(q2)) * np.cos(q1) + rho_0,
        (rho_0 + q0 * rho_1 * np.cos(q2)) * np.sin(q1),
        q0 * rho_1 * np.sin(q2) * delta]
    )

    s[:] = _numba_quaternion_rotate(x, q_xs)


@guvectorize([
    "void(float32[:], float32[:], float32[:], float32[:], float32[:])",
    "void(float64[:], float64[:], float64[:], float64[:], float64[:])"],
    '(i), (j), (k), (l) -> (i)')
def _numba_f(s, iparams, sparams, q_sx, q):
    delta = iparams[5]
    (_, _, rho_0, rho_1, _) = sparams
    _s = np.array([0, s[0], s[1], s[2]]).astype(s.dtype)

    xs = _numba_quaternion_rotate(_s, q_sx)

    (x0, x1, x2) = xs

    rd = np.sqrt((rho_0 - x0) ** 2 + x1 ** 2) - rho_0

    if x0 == rho_0:
        if x1 >= 0:
            psi = np.pi / 2
        else:
            psi = 3 * np.pi / 2
    else:
        psi = np.arctan2(-x1, x0 - rho_0) + np.pi

    if rd == 0:
        if x2 >= 0:
            phi = np.pi / 2
        else:
            phi = 3 * np.pi / 2
    else:
        if rd > 0:
            phi = np.arctan(x2 / rd / delta)
        else:
            phi = -np.pi + np.arctan(x2 / rd / delta)

        if phi < 0:
            phi += 2 * np.pi

    if phi == np.pi / 2 or phi == 3 * np.pi / 2:
        r = x2 / delta / rho_1 / np.sin(phi)
    else:
        r = np.abs((np.sqrt((rho_0 - x0) ** 2 + x1 ** 2) - rho_0) / np.cos(phi) / rho_1)

    q[0] = r
    q[1] = psi
    q[2] = phi


@numba.njit
def _numba_jac(q0, q1, q2, rho_0, rho_1, delta):
    dr = np.array([
        -rho_1 * np.cos(q2) * np.cos(q1),
        rho_1 * np.cos(q2) * np.sin(q1),
        rho_1 * np.sin(q2) * delta
    ])

    dpsi = np.array([
        rho_0 * np.sin(q1) + q0 * rho_1 * np.cos(q2) * np.sin(q1),
        rho_0 * np.cos(q1) + q0 * rho_1 * np.cos(q2) * np.cos(q1),
        0,
    ])

    dphi = np.array([
        q0 * rho_1 * np.sin(q2) * np.cos(q1),
        -q0 * rho_1 * np.sin(q2) * np.sin(q1),
        q0 * rho_1 * np.cos(q2) * delta
    ])

    return dr, dpsi, dphi
