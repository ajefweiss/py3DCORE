# -*- coding: utf-8 -*-

"""thux.py
"""

import datetime
import numba
import numpy as np
import os

from scipy.io import loadmat


class SolarWindBG(object):
    dt_0: datetime.datetime
    sw_data: np.ndarray

    ftype: str

    rmin: float
    rmax: float

    def __init__(self, dt_0: datetime.datetime, path: str, file_type: str = "radial_txt_2d", rmin: float = 0.0232523, rmax: float = 1.5) -> None:
        self.dt_0 = dt_0
        self.rmin = rmin
        self.rmax = rmax
        self.ftype = file_type

        if self.ftype in ["radial_txt", "radial_txt_2d"]:
            self.sw_data = np.loadtxt(path)
        elif self.ftype in ["matlab_3d"]:
            fname = os.path.basename(path).split(".")[0]
            self.sw_data = loadmat(path)[fname]
        else:
            raise NotImplementedError

    def get_sw_vr(self, dt: datetime.datetime, pos:np.ndarray) -> float:
        return _numba_get_sw_vr((dt - self.dt_0).total_seconds(), pos, self.sw_data, self.rmax, self.rmin)

    def visualize_xy(self, dt: datetime.datetime, z: float = 0) -> np.ndarray:
        rs = np.linspace(self.rmin, self.rmax, self.sw_data.shape[0])
        phis = np.linspace(0, 2 * np.pi, self.sw_data.shape[1])

        rr, pp = np.meshgrid(rs, phis)
        XX = (rr * np.sin(pp)).ravel()
        YY = (rr * np.cos(pp)).ravel()
        ZZ = np.zeros_like(XX)

        for i in range(len(XX)):
            ZZ[i] = self.get_sw_vr(dt, np.array([XX[i], YY[i], z]))

        shape = (self.sw_data.shape[1], self.sw_data.shape[0])

        XX = XX.reshape(shape)
        YY = YY.reshape(shape)
        ZZ = ZZ.reshape(shape)

        return XX, YY, ZZ

    def visualize_xz(self, dt: datetime.datetime) -> np.ndarray:
        rs = np.linspace(self.rmin, self.rmax, self.sw_data.shape[0])
        thetas = np.linspace(-np.pi / 2, np.pi / 2, self.sw_data.shape[2])

        rr, tt = np.meshgrid(rs, thetas)
        XX = (rr * np.cos(tt)).ravel()
        ZZ = (rr * np.sin(tt)).ravel()
        YY = np.zeros_like(XX)

        for i in range(len(XX)):
            YY[i] = self.get_sw_vr(dt, np.array([XX[i], 0, ZZ[i]]))

        shape = (self.sw_data.shape[2], self.sw_data.shape[0])

        XX = XX.reshape(shape)
        YY = YY.reshape(shape)
        ZZ = ZZ.reshape(shape)

        return XX, YY, ZZ


@numba.njit
def _numba_get_sw_vr(dt_offset: float, pos: np.ndarray, sw_data: np.ndarray, rmax: float, rmin: float) -> float:
    rota = 360 / (27 * 24 * 3600) * dt_offset

    r = np.linalg.norm(pos)

    lon = np.degrees(np.arctan2(-pos[1], pos[0]))

    if r < rmin:
        rx = 0
    elif r >= rmax:
        rx = sw_data.shape[0] - 1
    else:
        rx = int((r - rmin) * 1 / ((rmax - rmin) / sw_data.shape[0]))
        if rx >= sw_data.shape[0]:
            rx = sw_data.shape[0] - 1

    lx = int((lon + rota) / 2)

    if sw_data.ndim == 2:
        return sw_data[rx, lx]

    rv = np.sqrt(pos[1]**2 + pos[0]**2)

    if rv == 0:
        lat = 0
    else:
        lat = np.degrees(np.arctan(pos[2] / rv))

    ly = int(lat // 2 + 45)

    if ly == 90:
        ly = 89

    return sw_data[rx, lx, ly]
