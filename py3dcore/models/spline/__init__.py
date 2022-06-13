# -*- coding: utf-8 -*-

import datetime
import json
import numba
import numpy as np
import os
import py3dcore

from ...model import SimulationBlackBox
from ...rotqs import _numba_quaternion_rotate
from ...swbg import SolarWindBG, _numba_get_sw_vr
from ...util import ldl_decomp
from heliosat.util import sanitize_dt
from itertools import product
from numba import guvectorize
from scipy.optimize import least_squares
from typing import Any, Callable, Optional, Sequence, Tuple, Union

from .csplines import csplines_qs, csplines_sq, csplines_gh, csplines_gh_ub


class SplineModel(SimulationBlackBox):
    """Implements the spline model.

    Model Parameters
    ================
        For this specific model there are a total of 13 initial parameters which are as follows:
        0: t_i          time offset
        1: lon          longitude
        2: lat          latitude
        3: inc          inclination

        4: dia          cross section diameter at 1 AU
        5: delta        cross section aspect ratio

        6: r0           initial cme radius
        7: v0           initial cme velocity
        8: T            T factor (related to tau)

        9: n_a          expansion rate
        10: n_b         magnetic field decay rate

        11: b           magnetic field strength at center at 1AU
        12: bg_d        solar wind background drag coefficient

        There are Nx6 state parameters which are as follows:
        0: p_xt         position x
        1: p_yt         position y
        2: p_zt         position z
        3: v_xt         velocity x
        4: v_yt         velocity y
        5: v_zt         velocity z
    """
    mag_model: str
    particles: int
    shape_model: str
    swbg: np.ndarray

    cscoeff: np.ndarray
    cscoeff_v: np.ndarray

    def __init__(self, dt_0: Union[str, datetime.datetime], ensemble_size: int, swbg: SolarWindBG, particles: int = 11, iparams: dict = {}, shape_model: str = "torus", mag_model: str = "gh", dtype: type = np.float32) -> None:
        with open(os.path.join(os.path.dirname(py3dcore.__file__), "models/spline/parameters.json")) as fh:
            iparams_dict = json.load(fh)

        for k, v in iparams.items():
            if k in iparams_dict:
                iparams_dict[k].update(v)
            else:
                raise KeyError("key \"{0!s}\" not defined in parameters.json".format(k))

        super(SplineModel, self).__init__(dt_0, iparams=iparams_dict, sparams=(particles, 6), ensemble_size=ensemble_size, dtype=dtype)

        self.mag_model = mag_model
        self.particles = particles
        self.shape_model = shape_model
        self.swbg = swbg

        self.cscoeff = np.empty((ensemble_size, particles, 3), dtype=dtype)
        self.cscoeff_v = np.zeros((ensemble_size, particles, 3), dtype=dtype)

    def generator(self, *args: Any, **kwargs: Any) -> None:
        super(SplineModel, self).generator(*args, **kwargs)

        # init sparams
        if self.shape_model == "cylinder":
            init_particles(_numba_pfunc_cylinder, self.iparams_arr, self.sparams_arr, self.qs_xs, self.particles)
        elif self.shape_model == "circle":
            init_particles(_numba_pfunc_circle, self.iparams_arr, self.sparams_arr, self.qs_xs, self.particles)
        elif self.shape_model == "gcslike":
            init_particles(_numba_pfunc_gcslike, self.iparams_arr, self.sparams_arr, self.qs_xs, self.particles)
        elif self.shape_model == "gcslike_curved":
            init_particles(_numba_pfunc_gcslike_curved, self.iparams_arr, self.sparams_arr, self.qs_xs, self.particles)
        elif self.shape_model == "torus":
            init_particles(_numba_pfunc_torus, self.iparams_arr, self.sparams_arr, self.qs_xs, self.particles)
        else:
            raise NotImplementedError
    
    def propagator(self, dt_to: Union[str, datetime.datetime], time_resolution: int = 5 * 3600) -> None:
        offset = self.dtype(sanitize_dt(dt_to).timestamp() - self.dt_t.timestamp())  # type: ignore

        if offset < 0:
            raise NotImplementedError("cannot simulate backwards in time")

        while offset > time_resolution:
            _numba_propagator(time_resolution, self.iparams_arr, self.sparams_arr, self.particles, self.swbg.sw_data, self.swbg.rmin, self.swbg.rmax)

            offset -= time_resolution

        if offset > 0:
            _numba_propagator(offset, self.iparams_arr, self.sparams_arr, self.particles, self.swbg.sw_data, self.swbg.rmin, self.swbg.rmax)

        self.dt_t = sanitize_dt(dt_to)

        lmatrix = _numba_generate_csplines_lmatrix(self.particles)
        _numba_generate_csplines(self.sparams_arr, lmatrix, self.particles, self.cscoeff, self.cscoeff_v)

    def simulator_mag(self, pos: np.ndarray, out: np.ndarray) -> None:
        _q_tmp = np.zeros((len(self.iparams_arr), 3 * 4))
        
        csplines_sq(pos, self.iparams_arr, self.sparams_arr, self.cscoeff, self.cscoeff_v, _q_tmp, _q_tmp)

        if self.mag_model == "gh":
            csplines_gh(_q_tmp, self.iparams_arr, self.sparams_arr, self.cscoeff, self.cscoeff_v, out, out)
        else:
            raise NotImplementedError

    def update_iparams(self, *args: Any, **kwargs: Any) -> None:
        super(SplineModel, self).update_iparams(*args, **kwargs)

        # init sparams
        if self.shape_model == "cylinder":
            init_particles(_numba_pfunc_cylinder, self.iparams_arr, self.sparams_arr, self.qs_xs, self.particles)
        elif self.shape_model == "circle":
            init_particles(_numba_pfunc_circle, self.iparams_arr, self.sparams_arr, self.qs_xs, self.particles)
        elif self.shape_model == "gcslike":
            init_particles(_numba_pfunc_gcslike, self.iparams_arr, self.sparams_arr, self.qs_xs, self.particles)
        elif self.shape_model == "gcslike_curved":
            init_particles(_numba_pfunc_gcslike_curved, self.iparams_arr, self.sparams_arr, self.qs_xs, self.particles)
        elif self.shape_model == "torus":
            init_particles(_numba_pfunc_torus, self.iparams_arr, self.sparams_arr, self.qs_xs, self.particles)
        else:
            raise NotImplementedError

    def perturb_iparams(self, *args: Any, **kwargs: Any) -> None:
        super(SplineModel, self).perturb_iparams(*args, **kwargs)

        # init sparams
        if self.shape_model == "cylinder":
            init_particles(_numba_pfunc_cylinder, self.iparams_arr, self.sparams_arr, self.qs_xs, self.particles)
        elif self.shape_model == "circle":
            init_particles(_numba_pfunc_circle, self.iparams_arr, self.sparams_arr, self.qs_xs, self.particles)
        elif self.shape_model == "gcslike":
            init_particles(_numba_pfunc_gcslike, self.iparams_arr, self.sparams_arr, self.qs_xs, self.particles)
        elif self.shape_model == "gcslike_curved":
            init_particles(_numba_pfunc_gcslike_curved, self.iparams_arr, self.sparams_arr, self.qs_xs, self.particles)
        elif self.shape_model == "torus":
            init_particles(_numba_pfunc_torus, self.iparams_arr, self.sparams_arr, self.qs_xs, self.particles)
        else:
            raise NotImplementedError

    def visualize_fieldlines(self, q0: np.ndarray, iparam_index: int = 0, step_size: float = 0.001) -> np.ndarray:
        _q = np.empty((12,))
        _s = np.empty((3,))
        _b = np.empty((3,))

        h = 1 / (self.particles - 1)

        if q0[1] < h:
            q0[1] = h

        csplines_qs(q0, self.iparams_arr[iparam_index], self.sparams_arr[iparam_index],
                    self.cscoeff[iparam_index], self.cscoeff_v[iparam_index], _s, _s)

        fll = [np.array(_s)]

        def iter(s):
            csplines_sq(s, self.iparams_arr[iparam_index], self.sparams_arr[iparam_index],
                        self.cscoeff[iparam_index], self.cscoeff_v[iparam_index], _q, _q)
            csplines_gh_ub(_q, self.iparams_arr[iparam_index], self.sparams_arr[iparam_index],
                        self.cscoeff[iparam_index], self.cscoeff_v[iparam_index], _b, _b)
            return _b / np.linalg.norm(_b)

        while True:
            try:
                sol = getattr(least_squares(
                    lambda x: x - fll[-1] - step_size *
                    iter((x.astype(self.dtype) + fll[-1]) / 2),
                    fll[-1], method="trf"), "x")

                fll.append(np.array(sol.astype(self.dtype)))

                csplines_sq(fll[-1], self.iparams_arr[iparam_index], self.sparams_arr[iparam_index],
                            self.cscoeff[iparam_index], self.cscoeff_v[iparam_index], _q)
            except Exception as err:
                return np.array(fll[:-1]).astype(self.dtype)

            if _q[1] > 1 - h:
                return np.array(fll[:-1]).astype(self.dtype)

    def visualize_shape(self, iparam_index: int = 0, resolution: int = 10) -> np.ndarray:
        r = np.array([1.0], dtype=self.dtype)

        h = 1 / (self.particles - 1)

        c = 360 // resolution + 1
        u = np.linspace(h, 1 - h, int(360 // resolution) + 1)
        v = np.radians(np.r_[0:360 + resolution:resolution])

        # combination of wireframe points in (q)
        arr = np.array(list(product(r, u, v)), dtype=self.dtype).reshape(c ** 2, 3)
        arr2 = np.zeros_like(arr)

        for i in range(0, len(arr)):
            csplines_qs(arr[i], self.iparams_arr[iparam_index], self.sparams_arr[iparam_index],
                        self.cscoeff[iparam_index], self.cscoeff_v[iparam_index], arr2[i], arr2[i])

        return arr2.reshape((c, c, 3))


def init_particles(pfunc: Callable, iparams_arr: np.ndarray, sparams_arr: np.ndarray, qs_xs:np.ndarray, pn: int) -> None:
    for i in range(len(iparams_arr)):
        (_, lon, lat, inc, _, _, r_0, v_0, _, _, _, _, _) = iparams_arr[i]

        r_0 *= .00465047
        phi_0 = 1
        phi_arr = np.linspace(phi_0, 2 * np.pi - phi_0, pn, endpoint=True)

        xs_1 = np.array([pfunc(r_0, phi) for phi in phi_arr])
        xs_2 = np.array([pfunc(r_0 * 1.1, phi) for phi in phi_arr])

        # rotate into (s)
        xs_1 = np.array([_numba_quaternion_rotate(_, qs_xs[i]) for _ in xs_1])   
        xs_2 = np.array([_numba_quaternion_rotate(_, qs_xs[i]) for _ in xs_2])  

        sparams_arr[i, :, :3] = xs_1

        dxs = xs_2 - xs_1
        dxs *= v_0 / np.linalg.norm(dxs[(pn - 1) // 2])

        sparams_arr[i, :, 3:] = dxs


@numba.njit(parallel=False)
def _numba_generate_csplines(sparams_arr: np.ndarray, Lmat: np.ndarray, pn: int, cscoeff_x: np.ndarray, cscoeff_v: np.ndarray) -> None:
    h = 1 / (pn - 1)

    for i in numba.prange(len(sparams_arr)):
        d, x, y = np.zeros((pn,)), np.empty((pn,)), np.empty((pn,))

        for j in range(3):
            d[0] = 0
            d[pn - 1] = 0

            pos = sparams_arr[i, :, j]

            for k in range(1, pn - 1):
                d[k] = 3 * (pos[k + 1] + pos[k - 1] - 2 * pos[k]) / h**2

            y[0] = 0
            for k in range(1, pn):
                y[k] = 1 / Lmat[k, k] * (d[k] - Lmat[k, k - 1] * y[k - 1])
            
            x[pn - 1] = y[pn - 1] / Lmat[k, k]

            for k in range(0, pn - 1):
                k_ = pn - 2 - k
                x[k_] = 1 / Lmat[k_, k_] * (y[k_] - Lmat[k_ + 1,k_] * x[k_ + 1])

            cscoeff_x[i, :, j] = x

        d, x, y = np.zeros((pn,)), np.empty((pn,)), np.empty((pn,))

        for j in range(3, 6):
            d[0] = 0
            d[pn - 1] = 0

            vel = sparams_arr[i, :, j]

            for k in range(1, pn - 1):
                d[k] = 3 * (vel[k + 1] + vel[k - 1] - 2 * vel[k]) / h**2

            y[0] = 0
            for k in range(1, pn):
                y[k] = 1 / Lmat[k, k] * (d[k] - Lmat[k, k - 1] * y[k - 1])
            
            x[pn - 1] = y[pn - 1] / Lmat[k, k]

            for k in range(0, pn - 1):
                k_ = pn - 2 - k
                x[k_] = 1 / Lmat[k_, k_] * (y[k_] - Lmat[k_ + 1,k_] * x[k_ + 1])

            cscoeff_v[i, :, j - 3] = x


def _numba_generate_csplines_lmatrix(pn: int) -> np.ndarray:
    A = np.zeros((pn, pn))

    for k in range(pn):
        A[k, k] = 2
    
    for k in range(pn - 1):
        A[k, k + 1] = 1 / 2
        A[k + 1, k] = 1 / 2
    
    A[0, 1] = .5
    A[k-1, k-2] = .5

    return ldl_decomp(A)


#@numba.njit
def _numba_propagator(t_offset: float, iparams_arr: np.ndarray, sparams_arr: np.ndarray, pn: int, sw_data: np.ndarray, sw_rmin: float, sw_rmax: float) -> None:
    for i in numba.prange(len(iparams_arr)):
        gamma = iparams_arr[i, 12] * 1e-7

        a_fac = np.empty((pn,))

        for k in range(pn):
            # 3d drag
            #p_v_abs = np.linalg.norm(sparams_arr[i, k, 3:])

            #p_xe = sparams_arr[i, k, :3] / np.linalg.norm(sparams_arr[i, k, :3])
            #p_ve = sparams_arr[i, k, 3:] / p_v_abs

            #v_sw_vec = (_numba_get_sw_vr(t_offset, sparams_arr[i, k, :3], sw_data, sw_rmax, sw_rmin) * p_xe).astype(p_ve.dtype)
            #v_sw_eff = np.linalg.norm(np.dot(p_ve, v_sw_vec))

            # 1d drag
            p_v_abs = np.linalg.norm(sparams_arr[i, k, 3:6])
            v_sw_eff = _numba_get_sw_vr(t_offset, sparams_arr[i, k, :3], sw_data, sw_rmax, sw_rmin)

            if v_sw_eff == 0 or np.isnan(v_sw_eff) or p_v_abs == 0 or np.isnan(p_v_abs):
                a_fac[k] = 1
            else:
                dv = p_v_abs - v_sw_eff
                a = -gamma * dv * np.abs(dv) * t_offset
                #print(a)
                #print(p_v_abs)
                a_fac[k] = (p_v_abs + a) / p_v_abs

        sparams_arr[i, :, 0] += sparams_arr[i, :, 3] * (1 + a_fac) / 2 / 1.496e+8 * t_offset
        sparams_arr[i, :, 1] += sparams_arr[i, :, 4] * (1 + a_fac) / 2 / 1.496e+8 * t_offset
        sparams_arr[i, :, 2] += sparams_arr[i, :, 5] * (1 + a_fac) / 2 / 1.496e+8 * t_offset
        sparams_arr[i, :, 3] *= a_fac  
        sparams_arr[i, :, 4] *= a_fac  
        sparams_arr[i, :, 5] *= a_fac


@numba.njit
def _numba_pfunc_cylinder(r: float, p: float) -> np.ndarray:
    return np.array([0, r, r * (p - np.pi) / np.pi, 0])


@numba.njit
def _numba_pfunc_gcslike(r: float, p: float) -> np.ndarray:
    return np.array([0, -r / 2 * np.cos(p) + r / 2, r / 2 * np.sin(p) * np.abs(np.sin(p / 2)), 0])

@numba.njit
def _numba_pfunc_circle(r: float, p: float) -> np.ndarray:
    p = ((p - 1) / (2 * np.pi - 2) - 0.5) * np.pi / 3
    return np.array([0, r * np.cos(p), r * np.sin(p), 0])

@numba.njit
def _numba_pfunc_gcslike_curved(r: float, p: float) -> np.ndarray:
    d = 2
    if p < d or p > 2 * np.pi - d:
        pred = 0
    else:
        pred = 2 * np.pi * (p - d) / (2 * np.pi - 2 * d)

    return np.array([0, -r / 2 * np.cos(p) + r / 2, r / 2 * np.sin(p) * np.abs(np.sin(p / 2)), r / 25 * np.sin(pred)])

@numba.njit
def _numba_pfunc_torus(r: float, p: float) -> np.ndarray:
    return np.array([0, -r / 2 * np.cos(p) + r / 2, r / 2 * np.sin(p), 0])

