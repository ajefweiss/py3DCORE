# -*- coding: utf-8 -*-

import numba
import numpy as np

from numba import guvectorize

#np.seterr(all = "raise") 


@numba.njit
def _numba_linalg_norm_axis1(arr: np.ndarray) -> np.ndarray:
    result = np.zeros((len(arr),))

    for i in range(len(arr)):
        result[i] = np.linalg.norm(arr[i])

    return result


@numba.njit
def csplines_C0(t: float, cscoeff: np.ndarray, sparams: np.ndarray, pn: int) -> np.ndarray:
    h = 1 / (pn - 1)
    si = int(np.floor(t * (pn - 1) + 1))
    tsi = si / (pn - 1)

    tsimt = tsi - t
    tmtsimh = (t - tsi + h)

    return (
        cscoeff[si - 1, 0] * tsimt**3 / 6 / h + cscoeff[si, 0] * tmtsimh**3 / 6 / h + (sparams[si - 1, 0] - (cscoeff[si - 1, 0] * h**2) / 6) * tsimt / h + (sparams[si, 0] - (cscoeff[si, 0] * h**2) / 6) * tmtsimh / h,
        cscoeff[si - 1, 1] * tsimt**3 / 6 / h + cscoeff[si, 1] * tmtsimh**3 / 6 / h + (sparams[si - 1, 1] - (cscoeff[si - 1, 1] * h**2) / 6) * tsimt / h + (sparams[si, 1] - (cscoeff[si, 1] * h**2) / 6) * tmtsimh / h,
        cscoeff[si - 1, 2] * tsimt**3 / 6 / h + cscoeff[si, 2] * tmtsimh**3 / 6 / h + (sparams[si - 1, 2] - (cscoeff[si - 1, 2] * h**2) / 6) * tsimt / h + (sparams[si, 2] - (cscoeff[si, 2] * h**2) / 6) * tmtsimh / h,
    )


@numba.njit
def csplines_C1(t: float, cscoeff: np.ndarray, sparams: np.ndarray, pn: int) -> np.ndarray:
    h = 1 / (pn - 1)
    si = int(np.floor(t * (pn - 1) + 1))
    tsi = si / (pn - 1)

    tsimt = tsi - t
    tmtsimh = (t - tsi + h)

    return (
        -cscoeff[si - 1, 0] * tsimt**2 / 2 / h + cscoeff[si, 0] * tmtsimh**2 / 2 / h - (sparams[si - 1, 0] - (cscoeff[si - 1, 0] * h**2) / 6) / h + (sparams[si, 0] - (cscoeff[si, 0] * h**2) / 6) / h,
        -cscoeff[si - 1, 1] * tsimt**2 / 2 / h + cscoeff[si, 1] * tmtsimh**2 / 2 / h - (sparams[si - 1, 1] - (cscoeff[si - 1, 1] * h**2) / 6) / h + (sparams[si, 1] - (cscoeff[si, 1] * h**2) / 6) / h,
        -cscoeff[si - 1, 2] * tsimt**2 / 2 / h + cscoeff[si, 2] * tmtsimh**2 / 2 / h - (sparams[si - 1, 2] - (cscoeff[si - 1, 2] * h**2) / 6) / h + (sparams[si, 2] - (cscoeff[si, 2] * h**2) / 6) / h,
    )


@numba.njit
def csplines_C2(t: float, cscoeff: np.ndarray, sparams: np.ndarray, pn: int) -> np.ndarray:
    h = 1 / (pn - 1)
    si = int(np.floor(t * (pn - 1) + 1))
    tsi = si / (pn - 1)

    return (
        cscoeff[si - 1, 0] * (tsi - t) / h + cscoeff[si, 0] * (t - tsi + h) / h,
        cscoeff[si - 1, 1] * (tsi - t) / h + cscoeff[si, 1] * (t - tsi + h) / h,
        cscoeff[si - 1, 2] * (tsi - t) / h + cscoeff[si, 2] * (t - tsi + h) / h,
    )


@numba.njit
def _numba_csplines_newton_step(t: float, x: np.ndarray, cscoeff: np.ndarray, sparams: np.ndarray, pn: int) -> float:
    c0v = np.array(csplines_C0(t, cscoeff, sparams, pn))
    c1v = np.array(csplines_C1(t, cscoeff, sparams, pn))
    c2v = np.array(csplines_C2(t, cscoeff, sparams, pn))

    f1 = np.sum((c0v - x) * c1v)
    f2 = np.sum(c1v**2 + (c0v - x)*c2v)

    if np.abs(f1 / f2) > 0.05:
        fac = 0.025 / np.abs(f1 / f2)
    else:
        fac = 1
    return float(t - fac * f1 / f2)


@guvectorize([
    "void(float32[:], float32[:], float32[:, :], float32[:, :], float32[:, :], float32[:], float32[:])",
    "void(float64[:], float64[:], float64[:, :], float64[:, :], float64[:, :], float64[:], float64[:])"],
    '(i), (j), (k, l), (k, n), (k, n), (o) -> (o)')
def csplines_qs(q: np.ndarray, iparams: np.ndarray, sparams: np.ndarray, cscoeff: np.ndarray, cscoeff_v: np.ndarray, _: np.ndarray, s: np.ndarray) -> None:
    #print("qs")
    pn = len(sparams)
    ci = int(pn // 2)

    d_1au = iparams[4]
    delta = 1 #iparams[5]
    n_a = iparams[9]

    r_apex = np.linalg.norm(sparams[ci, :3])
    d_rau = d_1au * (r_apex ** n_a)

    tv = q[1]

    # compute angle
    local_v = np.array(csplines_C0(tv, cscoeff_v, sparams[:, 3:], pn))
    
    local_t = np.array(csplines_C1(tv, cscoeff, sparams[:, :3], pn))
    local_t = local_t / np.linalg.norm(local_t)

    local_n = -np.cross(local_v, local_t)
    local_n = local_n / np.linalg.norm(local_n)
    
    local_r = -np.cross(local_t, local_n)
    local_r = local_r / np.linalg.norm(local_r)

    local_s = np.array(csplines_C0(tv, cscoeff, sparams[:, :3], pn))
    local_s += q[0] * d_rau * np.cos(q[2]) * local_r / 2
    local_s += q[0] * d_rau * np.sin(q[2]) * local_n / 2 * delta

    s[0] = local_s[0]
    s[1] = local_s[1]
    s[2] = local_s[2]


@guvectorize([
    "void(float32[:], float32[:], float32[:, :], float32[:, :], float32[:, :], float32[:], float32[:])",
    "void(float64[:], float64[:], float64[:, :], float64[:, :], float64[:, :], float64[:], float64[:])"],
    '(i), (j), (k, l), (k, n), (k, n), (o) -> (o)')
def csplines_sq(s: np.ndarray, iparams: np.ndarray, sparams: np.ndarray, cscoeff: np.ndarray, cscoeff_v: np.ndarray, _: np.ndarray, q: np.ndarray) -> None:
    #print("sq")
    pdists = _numba_linalg_norm_axis1(sparams[:, :3] - s)
    pdists_min = np.argmin(pdists)

    # detect easy oob
    if pdists_min == 0 or pdists_min == len(sparams) - 1:
        q[:] = np.nan
        return

    pn = len(sparams)
    ci = int(pn // 2)

    d_1au = iparams[4]
    delta = 1 #iparams[5]
    n_a = iparams[9]

    r_apex = np.linalg.norm(sparams[ci, :3])
    d_rau = d_1au * (r_apex ** n_a)

    #inter_dist_max = np.max(_numba_linalg_norm_axis1(sparams[1:, :3] - sparams[:-1, :3]))
    #
    #if pdists[pdists_min]**2 > inter_dist_max**2 / 4 + 2 * d_rau:
    #    q[:] = np.nan
    #    return

    t0 = pdists_min / (pn - 1)
    tv = t0

    # perform 8 newton steps, more than sufficient
    for i in range(8):
        tv = _numba_csplines_newton_step(tv, s, cscoeff, sparams, pn)

    if np.isnan(tv) or tv > 1 or tv < 0:
        q[:] = np.nan 
        return

    local_s = s - np.array(csplines_C0(tv, cscoeff, sparams, pn))

    # compute angle
    local_v = np.array(csplines_C0(tv, cscoeff_v, sparams[:, 3:], pn))
    #print(local_v)

    local_t = np.array(csplines_C1(tv, cscoeff, sparams[:, :3], pn))
    local_t = local_t / np.linalg.norm(local_t)

    local_n = -np.cross(local_v, local_t)
    local_n = local_n / np.linalg.norm(local_n)
    
    local_r = -np.cross(local_t, local_n)
    local_r = local_r / np.linalg.norm(local_r)

    angle = np.arctan2(np.dot(local_s, local_n), np.dot(local_s, local_r))

    d_rau_delta = d_rau * (1 + np.sin(angle) * (delta - 1))

    q[0] = 2 * np.linalg.norm(local_s) / d_rau_delta
    q[1] = tv
    q[2] = angle

    q[3] = local_r[0]
    q[4] = local_r[1]
    q[5] = local_r[2]

    q[6] = local_t[0]
    q[7] = local_t[1]
    q[8] = local_t[2]

    q[9] = local_n[0]
    q[10] = local_n[1]
    q[11] = local_n[2]


@guvectorize([
    "void(float32[:], float32[:], float32[:, :], float32[:, :], float32[:, :], float32[:], float32[:])",
    "void(float64[:], float64[:], float64[:, :], float64[:, :], float64[:, :], float64[:], float64[:])"],
    '(i), (j), (k, l), (k, n), (k, n), (o) -> (o)')
def csplines_gh(q: np.ndarray, iparams: np.ndarray, sparams: np.ndarray, cscoeff: np.ndarray, cscoeff_v: np.ndarray, _: np.ndarray, b: np.ndarray) -> None:
    #print("gh")
    rv = q[0]
    tv = q[1]
    av = q[2]

    if not np.isnan(rv) and rv < 1:
        delta = 1 #iparams[5]
        Tfac = iparams[8]
        n_b = iparams[10]
        b_1au = iparams[11]

        local_r = q[3:6]
        local_t = q[6:9]
        local_n = q[9:12]

        # ellipse circumference
        h = (delta - 1)**2 / (1 + delta)**2
        E = np.pi * (1 + delta) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))

        t = Tfac  / 2 / np.pi

        pn = len(sparams)
        ci = int(pn // 2)

        r_apex = np.linalg.norm(sparams[ci, :3])
        b_t = b_1au * (r_apex ** (-n_b))

        denom = (1 + t**2 * rv**2)
        bpsi = b_t / denom
        bphi = b_t * t * rv / denom

        b[:] = bpsi * local_t + bphi * local_n * np.cos(av) - bphi * local_r * np.sin(av)
    else:
        b[0] = 0
        b[1] = 0
        b[2] = 0


@guvectorize([
    "void(float32[:], float32[:], float32[:, :], float32[:, :], float32[:, :], float32[:], float32[:])",
    "void(float64[:], float64[:], float64[:, :], float64[:, :], float64[:, :], float64[:], float64[:])"],
    '(i), (j), (k, l), (k, n), (k, n), (o) -> (o)')
def csplines_gh_ub(q: np.ndarray, iparams: np.ndarray, sparams: np.ndarray, cscoeff: np.ndarray, cscoeff_v: np.ndarray, _: np.ndarray, b: np.ndarray) -> None:
    rv = q[0]
    tv = q[1]
    av = q[2]

    if not np.isnan(rv):
        delta = 1 # iparams[5]
        Tfac = iparams[8]
        n_b = iparams[10]
        b_1au = iparams[11]

        local_r = q[3:6]
        local_t = q[6:9]
        local_n = q[9:12]

        # ellipse circumference
        h = (delta - 1)**2 / (1 + delta)**2
        E = np.pi * (1 + delta) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))

        t = Tfac  / 2 / np.pi

        pn = len(sparams)
        ci = int(pn // 2)

        r_apex = np.linalg.norm(sparams[ci, :3])
        b_t = b_1au * (r_apex ** (-n_b))

        denom = (1 + t**2 * rv**2)
        bpsi = b_t / denom
        bphi = b_t * t * rv / denom

        b[:] = bpsi * local_t + bphi * local_n * np.cos(av) - bphi * local_r * np.sin(av)
    else:
        b[0] = 0
        b[1] = 0
        b[2] = 0
