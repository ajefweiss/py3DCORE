# -*- coding: utf-8 -*-

"""quaternions.py

Implements functions for using quaternions to rotate vectors.
"""

import math
import numba
import numba.cuda as cuda
import numpy as np

from numba import guvectorize


def generate_quaternions(iparams_arr, qs_sx, qs_xs, use_gpu=False, indices=None):
    """Wrapper function for generating rotational quaternions from iparams array.

    Parameters
    ----------
    iparams_arr : Union[np.ndarray, numba.cuda.cudadrv.devicearray.DeviceNDArray]
        Initial parameters array.
    qs_sx : Union[np.ndarray, numba.cuda.cudadrv.devicearray.DeviceNDArray]
        Array for (s) -> (x) rotational quaternions.
    qs_xs : Union[np.ndarray, numba.cuda.cudadrv.devicearray.DeviceNDArray]
        Array for (x) -> (s) rotational quaternions.
    use_gpu : bool
        GPU flag, by default False.
    indices : np.ndarray, optional
        Lon/Lat/Inc indices in params, by default None.
    """
    if indices is None:
        indices = np.array([1, 2, 3])
    else:
        indices = np.array(indices)

    if use_gpu and cuda.is_available():
        (i1, i2, i3) = (indices[0], indices[1], indices[2])

        # WIP
    else:
        (i1, i2, i3) = (indices[0], indices[1], indices[2])
        (lon, lat, inc) = (iparams_arr[:, i1], iparams_arr[:, i2], iparams_arr[:, i3])

        ux = np.array([0, 1.0, 0, 0])
        uy = np.array([0, 0, 1.0, 0])
        uz = np.array([0, 0, 0, 1.0])

        rlon = _numba_quaternion_create(lon, uz)
        rlat = _numba_quaternion_create(-lat, quaternion_rotate(uy, rlon))
        rinc = _numba_quaternion_create(
            inc, quaternion_rotate(ux, _numba_quaternion_multiply(rlat, rlon)))

        _numba_quaternion_multiply(rinc, _numba_quaternion_multiply(rlat, rlon), qs_xs)
        _numba_quaternion_conjugate(qs_xs, qs_sx)


def quaternion_rotate(vec, q, use_gpu=False):
    if use_gpu and cuda.is_available():
        pass
    else:
        return _numba_quaternion_multiply(
            q, _numba_quaternion_multiply(vec, _numba_quaternion_conjugate(q)))


@numba.njit
def _numba_quaternion_rotate(vec, q):
    qh = (q[0], -q[1], -q[2], -q[3])

    # mul 1
    ma = - vec[1] * qh[1] - vec[2] * qh[2] - vec[3] * qh[3]
    mb = + vec[1] * qh[0] + vec[2] * qh[3] - vec[3] * qh[2]
    mc = - vec[1] * qh[3] + vec[2] * qh[0] + vec[3] * qh[1]
    md = + vec[1] * qh[2] - vec[2] * qh[1] + vec[3] * qh[0]

    # mul 2
    rb = q[0] * mb + q[1] * ma + q[2] * md - q[3] * mc
    rc = q[0] * mc - q[1] * md + q[2] * ma + q[3] * mb
    rd = q[0] * md + q[1] * mc - q[2] * mb + q[3] * ma

    return np.array([rb, rc, rd])


@guvectorize([
    "void(float32, float32[:], float32[:])",
    "void(float64, float64[:], float64[:])"],
    '(), (n) -> (n)')
def _numba_quaternion_create(rot, vec, res):
    argument = np.radians(rot / 2)

    res[0] = np.cos(argument)

    # due to ufunc broadcasting rules vec must be of length 4, first entry is not used
    res[1] = vec[1] * np.sin(argument)
    res[2] = vec[2] * np.sin(argument)
    res[3] = vec[3] * np.sin(argument)


@guvectorize([
    "void(float32[:], float32[:])",
    "void(float64[:], float64[:])"],
    '(n) -> (n)')
def _numba_quaternion_conjugate(q, res):
    res[0] = q[0]
    res[1:] = -q[1:]


@guvectorize([
    "void(float32[:], float32[:], float32[:])",
    "void(float64[:], float64[:], float64[:])"],
    '(n), (n) -> (n)', fastmath=True)
def _numba_quaternion_multiply(q1, q2, res):
    (q1a, q1b, q1c, q1d) = (q1[0], q1[1], q1[2], q1[3])
    (q2a, q2b, q2c, q2d) = (q2[0], q2[1], q2[2], q2[3])

    res[0] = q1a * q2a - q1b * q2b - q1c * q2c - q1d * q2d
    res[1] = q1a * q2b + q1b * q2a + q1c * q2d - q1d * q2c
    res[2] = q1a * q2c - q1b * q2d + q1c * q2a + q1d * q2b
    res[3] = q1a * q2d + q1b * q2c - q1c * q2b + q1d * q2a


if cuda.is_available():
    @cuda.jit(device=True, inline=True)
    def _cuda_quaternion_rotate(vec, q):
        qh = (q[0], -q[1], -q[2], -q[3])

        # mul 1
        ma = - vec[1] * qh[1] - vec[2] * qh[2] - vec[3] * qh[3]
        mb = + vec[1] * qh[0] + vec[2] * qh[3] - vec[3] * qh[2]
        mc = - vec[1] * qh[3] + vec[2] * qh[0] + vec[3] * qh[1]
        md = + vec[1] * qh[2] - vec[2] * qh[1] + vec[3] * qh[0]

        # mul 2
        rb = q[0] * mb + q[1] * ma + q[2] * md - q[3] * mc
        rc = q[0] * mc - q[1] * md + q[2] * ma + q[3] * mb
        rd = q[0] * md + q[1] * mc - q[2] * mb + q[3] * ma

        return (rb, rc, rd)

    @guvectorize(
        "void(float32, float32[:], float32[:])",
        '(), (n) -> (n)', target="cuda")
    def _cuda_quaternion_create(rot, vec, res):
        argument = math.pi * rot / numba.float32(360)

        res[0] = math.cos(argument)

        # due to ufunc broadcasting rules vec must be of length 4, first entry is not used
        res[1] = vec[1] * math.sin(argument)
        res[2] = vec[2] * math.sin(argument)
        res[3] = vec[3] * math.sin(argument)

    @guvectorize(
        "void(float32[:], float32[:])",
        '(n) -> (n)', target="cuda")
    def _cuda_quaternion_conjugate(q, res):
        res[0] = q[0]
        res[1:] = -q[1:]

    @guvectorize(
        "void(float32[:], float32[:], float32[:])",
        '(n), (n) -> (n)', target="cuda")
    def _cuda_quaternion_multiply(q1, q2, res):
        (q1a, q1b, q1c, q1d) = (q1[0], q1[1], q1[2], q1[3])
        (q2a, q2b, q2c, q2d) = (q2[0], q2[1], q2[2], q2[3])

        res[0] = q1a * q2a - q1b * q2b - q1c * q2c - q1d * q2d
        res[1] = q1a * q2b + q1b * q2a + q1c * q2d - q1d * q2c
        res[2] = q1a * q2c - q1b * q2d + q1c * q2a + q1d * q2b
        res[3] = q1a * q2d + q1b * q2c - q1c * q2b + q1d * q2a
