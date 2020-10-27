# -*- coding: utf-8 -*-

import numba
import numpy as np

from numba import guvectorize
from py3dcore.rotqs import _numba_quaternion_rotate


# Wrappers

def g(q, s, iparams, sparams, q_xs, **kwargs):
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

    Other Parameters
    ----------------
    use_cuda : bool
        CUDA flag, by default False.
    """
    if kwargs.get("use_cuda", False):
        raise NotImplementedError("CUDA functionality is not available yet")
    else:
        _numba_g(q, iparams, sparams, q_xs, s)


def f(s, q, iparams, sparams, q_sx, **kwargs):
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

    Other Parameters
    ----------------
    use_cuda : bool
        CUDA flag, by default False.
    """
    if kwargs.get("use_cuda", False):
        raise NotImplementedError("CUDA functionality is not available yet")
    else:
        _numba_f(s, iparams, sparams, q_sx, q)


@guvectorize([
    "void(float32[:], float32[:], float32[:], float32[:], float32[:])",
    "void(float64[:], float64[:], float64[:], float64[:], float64[:])"],
    '(i), (j), (k), (l) -> (i)')
def _numba_g(q, iparams, sparams, q_xs, s):
    (q0, q1, q2) = q
    w = iparams[5]
    (_, _, rho_0, rho_1, delta, _) = sparams

    x = np.array([
        0,
        -(rho_0 + q0 * rho_1 * np.sin(q1 / 2)**2 * np.cos(q2)) * np.cos(q1) + rho_0,
        w * np.sin(q1 / 2)**2 * (rho_0 + q0 * rho_1 * np.sin(q1 / 2)**2 * np.cos(q2)) * np.sin(q1),
        q0 * rho_1 * np.sin(q1 / 2)**2 * np.sin(q2) * delta]
    )

    s[:] = _numba_quaternion_rotate(x, q_xs)


@numba.njit(["f4(f4, f4, f4, f4)", "f8(f8, f8, f8, f8)"])
def _numba_solve_psi(x0, x1, w, rho_0):
    """Search for psi within range [pi/2,  3 * pi /2].

    Uses a simple damped newton method.
    """
    func = lambda psi: x1 / (x0 - rho_0) + w * np.sin(psi / 2)**2 * np.tan(psi)
    dfunc = lambda psi:  w * np.sin(psi / 2) * np.cos(psi / 2) * np.tan(psi) + w * np.sin(psi / 2)**2 / np.cos(psi)**2

    t2 = np.pi
    alpha = 1

    # max iter
    N = 1e3
    Nc = 0

    while True:
        if Nc >= N:
            return np.nan

        t1 = t2 - alpha * func(t2) / dfunc(t2)

        if t1 > 3 * np.pi / 2 or t1 < np.pi / 2:
            alpha /= 2
            continue

        if np.abs(func(t2)) < 1e-10:
            break

        t2 = t1

        Nc += 1

    return t2


@guvectorize([
    "void(float32[:], float32[:], float32[:], float32[:], float32[:])",
    "void(float64[:], float64[:], float64[:], float64[:], float64[:])"],
    '(i), (j), (k), (l) -> (i)')
def _numba_f(s, iparams, sparams, q_sx, q):
    w = iparams[5]
    (_, _, rho_0, rho_1, delta, _) = sparams
    _s = np.array([0, s[0], s[1], s[2]]).astype(s.dtype)

    xs = _numba_quaternion_rotate(_s, q_sx)

    (x0, x1, x2) = xs

    if x0 == rho_0:
        if x1 >= 0:
            psi = np.pi / 2
        else:
            psi = 3 * np.pi / 2
    else:
        psi = _numba_solve_psi(x0, x1, w, rho_0)

    # abort if no solution for psi is found
    if np.isnan(psi):
        q[0] = 1e9
        q[1] = 1e-9
        q[2] = 1e-9
    else:
        g1 = np.cos(psi)**2 + w**2 * np.sin(psi / 2)**4 * np.sin(psi)**2
        rd = np.sqrt(((rho_0 - x0) ** 2 + x1 ** 2) / g1) - rho_0

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
            r = x2 / delta / rho_1 / np.sin(phi) / np.sin(psi / 2)**2
        else:
            r = np.abs(rd / np.cos(phi) / np.sin(psi / 2)**2 / rho_1)

        q[0] = r
        q[1] = psi
        q[2] = phi


@numba.njit
def _numba_jac(q0, q1, q2, rho_0, rho_1, delta, w):
    dr = np.array([
        -rho_1 * np.sin(q1 / 2)**2 * np.cos(q2) * np.cos(q1),
        w * np.sin(q1 / 2)**2 * rho_1 * np.sin(q1 / 2)**2 * np.cos(q2) * np.sin(q1),
        rho_1 * np.sin(q1 / 2)**2 * np.sin(q2) * delta
    ])

    dpsi = np.array([
        rho_0 * np.sin(q1) + q0 * rho_1 * np.sin(q1 / 2)**2 *
        np.cos(q2) * np.sin(q1) - q0
        * rho_1 * np.cos(q1 / 2) * np.sin(q1 / 2) * np.cos(q2) * np.cos(q1),
        w * (rho_0 * np.cos(q1) + q0 * rho_1 * np.sin(q1 / 2)**2 * np.cos(q2) * np.cos(q1)
             + q0 * rho_1 * np.cos(q1 / 2) * np.sin(q1 / 2) * np.cos(q2) * np.sin(q1))
        * np.sin(q1 / 2)**2 + np.sin(q1 / 2) * np.cos(q1 / 2) * w
        * (rho_0 + q0 * rho_1 * np.sin(q1 / 2)**2 * np.cos(q2)) * np.sin(q1),
        q0 * rho_1 * delta * np.cos(q1 / 2) * np.sin(q1 / 2) * np.sin(q2)
    ])

    dphi = np.array([
        q0 * rho_1 * np.sin(q1 / 2)**2 * np.sin(q2) * np.cos(q1),
        -w * np.sin(q1 / 2)**2 * q0 * rho_1 * np.sin(q1 / 2)**2 * np.sin(q2) * np.sin(q1),
        q0 * rho_1 * np.sin(q1 / 2)**2 * np.cos(q2) * delta
    ])

    return dr, dpsi, dphi
