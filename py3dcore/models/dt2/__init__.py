# -*- coding: utf-8 -*-

import datetime
import json
import os
from itertools import product
from typing import Union

import numpy as np
from heliosat.util import sanitize_dt
from numba import guvectorize
from scipy.optimize import least_squares

import py3dcore

from ...model import SimulationBlackBox
from .distorted_shape import dgds, distorted_qs, distorted_sq, distorted_sq_gh

print("importing dt2model")


class DT2Model(SimulationBlackBox):
    """Implements the distorted model.

    Model Parameters
    ================
        For this specific model there are a total of 14 initial parameters which are as follows:
        0: t_i          time offset
        1: lon          longitude
        2: lat          latitude
        3: inc          inclination

        4: dia          cross section diameter at 1 AU
        5: delta        cross section aspect ratio

        6: r0           initial cme radius
        7: v0           initial cme velocity
        8: T            T factor (related to the twist)

        9: n_a          expansion rate
        10: n_b         magnetic field decay rate

        11: b           magnetic field strength at center at 1AU
        12: bg_d        solar wind background drag coefficient
        13: bg_v        solar wind background speed

        14: n
        15: beta
        16: lambda
        17: epsilon
        18: kappa
        19: phihw       half width angle

        There are 5 state parameters which are as follows:
        0: v_t          current velocity
        1: rho_0        torus major radius
        2: rho_1        torus minor radius
        3: b_t          magnetic field strength at center
        4: gamma_l      axis length
        5: vel_05       curve velocity at s=0.5
    """

    mag_model: str
    shape_model: str

    def __init__(
        self,
        dt_0: Union[str, datetime.datetime],
        ensemble_size: int,
        iparams: dict = {},
        shape_model: str = "distorted",
        mag_model: str = "gh",
        dtype: type = np.float32,
    ) -> None:
        with open(
            os.path.join(
                os.path.dirname(py3dcore.__file__), "models/dt2/parameters.json"
            )
        ) as fh:
            iparams_dict = json.load(fh)

        for k, v in iparams.items():
            if k in iparams_dict:
                iparams_dict[k].update(v)
            else:
                raise KeyError('key "%s" not defined in parameters.json', k)

        super(DT2Model, self).__init__(
            dt_0,
            iparams=iparams_dict,
            sparams=6,
            ensemble_size=ensemble_size,
            dtype=dtype,
        )

        self.mag_model = mag_model
        self.shape_model = shape_model

    def propagator(self, dt_to: Union[str, datetime.datetime]) -> None:
        _numba_propagator(
            self.dtype(sanitize_dt(dt_to).timestamp() - self.dt_0.timestamp()),
            self.iparams_arr,
            self.sparams_arr,
            self.sparams_arr,
        )

        self.dt_t = dt_to

    def simulator_mag(self, pos: np.ndarray, out: np.ndarray) -> None:
        if self.shape_model == "distorted":
            distorted_sq_gh(
                pos, self.iparams_arr, self.sparams_arr, self.qs_sq, self.qs_qs, out
            )
        else:
            raise NotImplementedError

    def visualize_gamma(
        self,
        iparam_index: int = 0,
        resolution: int = 20,
        s_max: float = 1,
        s_min: float = 0,
    ) -> np.ndarray:
        sv = np.linspace(s_min, s_max, resolution)

        arr = np.array([[0, 0, _] for _ in sv])
        arr2 = np.zeros_like(arr)

        for i in range(0, len(sv)):
            distorted_qs(
                arr[i],
                self.iparams_arr[iparam_index],
                self.sparams_arr[iparam_index],
                self.qs_qs[iparam_index],
                arr2[i],
            )

        return arr2

    def visualize_shape(
        self,
        iparam_index: int = 0,
        resolution: int = 20,
        s_max: float = 0.1,
        s_min: float = 0.9,
    ) -> np.ndarray:
        r = np.array([1.0], dtype=self.dtype)
        u = np.linspace(0, 1, resolution)
        v = np.linspace(s_min, s_max, resolution)
        c1 = len(u)
        c2 = len(v)
        # combination of wireframe points in (q)
        arr = np.array(list(product(r, u, v)), dtype=self.dtype).reshape(c1 * c2, 3)
        arr2 = np.zeros_like(arr)

        for i in range(0, len(arr)):
            distorted_qs(
                arr[i],
                self.iparams_arr[iparam_index],
                self.sparams_arr[iparam_index],
                self.qs_qs[iparam_index],
                arr2[i],
            )

        return arr2.reshape((c1, c2, 3))

    def visualize_fieldline(
        self,
        q0: np.ndarray,
        index: int = 0,
        steps: int = 1000,
        step_size: float = 0.01,
        return_phi: bool = False,
        s_max: float = 1,
        s_min: float = 0,
    ):
        """Integrates along the magnetic field lines starting at a point q0 in (q) coordinates and
        returns the field lines in (s) coordinates.

        Parameters
        ----------
        q0 : np.ndarray
            Starting point in (q) coordinates.
        index : int, optional
            Model run index, by default 0.
        steps : int, optional
            Number of integration steps, by default 1000.
        step_size : float, optional
            Integration step size, by default 0.01.

        Returns
        -------
        np.ndarray
            Integrated magnetic field lines in (s) coordinates.
        """

        _tva = np.empty((3,), dtype=self.dtype)
        _tvb = np.empty((3,), dtype=self.dtype)
        _tvc = np.empty((3,), dtype=self.dtype)

        distorted_qs(
            q0,
            self.iparams_arr[index],
            self.sparams_arr[index],
            self.qs_qs[index],
            _tva,
        )

        r0 = q0[0]

        fl = [np.array(_tva, dtype=self.dtype)]
        rc = [q0[0]]
        qc = [q0[1]]
        sc = [q0[2]]

        def iterate(s: float) -> float:
            distorted_sq_gh(
                s,
                self.iparams_arr[index],
                self.sparams_arr[index],
                self.qs_sq[index],
                self.qs_qs[index],
                _tva,
            )

            return _tva / np.linalg.norm(_tva)

        while len(fl) < steps:

            try:
                # use implicit method and least squares for calculating the next step
                # sol = getattr(
                #     least_squares(
                #         lambda x: x
                #         - fl[-1]
                #         - step_size * iterate((x.astype(self.dtype) + fl[-1]) / 2),
                #         fl[-1],
                #     ),
                #     "x",
                # )

                # rk4

                k1 = iterate(fl[-1])
                k2 = iterate(fl[-1] + step_size * k1 / 2)
                k3 = iterate(fl[-1] + step_size * k2 / 2)
                k4 = iterate(fl[-1] + step_size * k3)

                sol = fl[-1] + step_size / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

                # remap to same q
                distorted_sq(
                    sol,
                    self.iparams_arr[index],
                    self.sparams_arr[index],
                    self.qs_sq[index],
                    _tvb,
                )

                _tvb[0] = r0

                distorted_qs(
                    _tvb,
                    self.iparams_arr[index],
                    self.sparams_arr[index],
                    self.qs_qs[index],
                    _tvc,
                )

                fl.append(np.array(_tvc.astype(self.dtype)))

                if _tvb[2] > s_max or _tvb[2] < s_min:
                    break
                rc.append(_tvb[0])
                qc.append(_tvb[1])
                sc.append(_tvb[2])
            except Exception as e:
                print(e)
                break

        fl = np.array(fl, dtype=self.dtype)
        qc = np.array(qc, dtype=self.dtype)
        sc = np.array(sc, dtype=self.dtype)
        rc = np.array(rc, dtype=self.dtype)

        if return_phi:
            return fl, qc, sc, rc
        else:
            return fl


@guvectorize(
    [
        "void(float32, float32[:], float32[:], float32[:])",
        "void(float64, float64[:], float64[:], float64[:])",
    ],
    "(), (j), (k) -> (k)",
    target="parallel",
)
def _numba_propagator(
    t_offset: float, iparams: np.ndarray, _: np.ndarray, sparams: np.ndarray
) -> None:
    (
        t_i,
        _,
        _,
        _,
        d,
        _,
        r,
        v,
        _,
        n_a,
        n_b,
        b_i,
        bg_d,
        bg_v,
        alpha,
        beta,
        lambda_v,
        epsilon,
        kappa,
        phihw,
        _,
        _,
    ) = iparams

    # rescale parameters
    bg_d = bg_d * 1e-7
    r = r * 695510

    dt = t_offset - t_i
    dv = v - bg_v

    bg_sgn = int(-1 + 2 * int(v > bg_v))

    rt = (bg_sgn / bg_d * np.log1p(bg_sgn * bg_d * dv * dt) + bg_v * dt + r) / 1.496e8

    vt = dv / (1 + bg_sgn * bg_d * dv * dt) + bg_v

    rho_1 = d * (rt**n_a) / 2
    rho_0 = (rt - rho_1) / 2
    b_t = b_i * (2 * rho_0) ** (-n_b)

    # compute flux rope length, rough estimate based on 20 points
    s_r = [
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
    ]

    gamma_l = 0

    for s_v in s_r:
        dv = rho_0 * np.linalg.norm(
            (dgds(s_v, phihw, alpha, beta, lambda_v, epsilon, kappa))
        )

        if not np.isnan(dv):
            gamma_l += dv

        if s_v == 0.5:
            vel05 = dv

    gamma_l /= len(s_r)

    sparams[0] = vt
    sparams[1] = rho_0
    sparams[2] = rho_1
    sparams[3] = b_t
    sparams[4] = gamma_l
    sparams[5] = vel05
