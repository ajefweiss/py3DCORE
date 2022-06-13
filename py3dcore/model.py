# -*- coding: utf-8 -*-

"""model.py
"""

import datetime
from matplotlib.pyplot import pie
import numba
import numpy as np
import scipy as sp

from .rotqs import generate_quaternions
from .util import ldl_decomp, set_random_seed
from heliosat.util import sanitize_dt
from typing import Any, Callable, Optional, Sequence, Tuple, Union


class SimulationBlackBox(object):
    """SimulationBlackBox class.
    """
    dt_0: datetime.datetime
    dt_t: Optional[datetime.datetime]

    iparams: dict
    sparams: Union[int, Sequence[int]]

    iparams_arr: np.ndarray
    sparams_arr: np.ndarray

    # extra iparam arrays
    iparams_kernel: np.ndarray
    iparams_kernel_decomp: np.ndarray
    iparams_meta: np.ndarray
    iparams_weight: np.ndarray

    ensemble_size: int

    dtype: type

    qs_sx: np.ndarray
    qs_xs: np.ndarray

    def __init__(self, dt_0: Union[str, datetime.datetime], iparams: dict, sparams: Union[int, Sequence[int]], ensemble_size: int, dtype: type) -> None:
        self.dt_0 = sanitize_dt(dt_0)
        self.dt_t = self.dt_0

        self.iparams = iparams
        self.sparams = sparams
        self.ensemble_size = ensemble_size
        self.dtype = dtype

        self.iparams_arr = np.empty((self.ensemble_size, len(self.iparams)), dtype=self.dtype)
        self.iparams_kernel = None
        self.iparams_weight = None
        self.iparams_kernel_decomp = None

        if isinstance(self.sparams, int):
            self.sparams_arr = np.empty((self.ensemble_size, self.sparams), dtype=self.dtype)
        elif isinstance(self.sparams, tuple):
            self.sparams_arr = np.empty((self.ensemble_size, *self.sparams), dtype=self.dtype)
        else:
            raise TypeError("sparams must be of type int or tuple, not {0!s}".format(type(self.sparams)))

        self.qs_sx = np.empty((self.ensemble_size, 4), dtype=self.dtype)
        self.qs_xs = np.empty((self.ensemble_size, 4), dtype=self.dtype)

        self.iparams_meta = np.empty((len(self.iparams), 7), dtype=self.dtype)
        self.update_iparams_meta()
 
    def generator(self, random_seed: int = 42) -> None:
        set_random_seed(random_seed)

        for k, iparam in self.iparams.items():
            ii = iparam["index"]
            dist = iparam["distribution"]

            def trunc_generator(func: Callable, max_v: float, min_v: float, size: int, **kwargs: Any) -> np.ndarray:
                numbers = func(size=size, **kwargs)
                for _ in range(100):
                    flt = ((numbers > max_v) | (numbers < min_v))
                    if np.sum(flt) == 0:
                        return numbers
                    numbers[flt] = func(size=len(flt), **kwargs)[flt]
                raise RuntimeError("drawing numbers inefficiently (%i/%i after 100 iterations)", len(flt), size)                

            # generate values according to given distribution
            if dist in ["fixed", "fixed_value"]:
                self.iparams_arr[:, ii] = iparam["default_value"]
            elif dist in ["constant", "uniform"]:
                self.iparams_arr[:, ii] = np.random.rand(self.ensemble_size) * (iparam["maximum"] - iparam["minimum"]) + iparam["minimum"]
            elif dist in ["gaussian", "normal"]:
                self.iparams_arr[:, ii] = trunc_generator(np.random.normal, iparam["maximum"], iparam["minimum"], self.ensemble_size, loc=iparam["mean"], scale=iparam["std"])
            else:
                raise NotImplementedError("parameter distribution \"{0!s}\" is not implemented".format(dist))

        generate_quaternions(self.iparams_arr, self.qs_sx, self.qs_xs)

        self.dt_t = self.dt_0

    def propagator(self, dt_to: Union[str, datetime.datetime]) -> None:
        raise NotImplementedError

    def simulator(self, dt: Union[str, datetime.datetime, Sequence[str], Sequence[datetime.datetime]], pos: Union[np.ndarray, Sequence[np.ndarray]], sparams: Optional[Sequence[int]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if isinstance(dt, datetime.datetime) or isinstance(dt, str):
            dt = [dt]  # type: ignore
            pos = [pos]
        elif (len(dt) > 1 and len(pos) == 1) or np.array(pos).ndim == 1:
            pos = [np.array(pos, dtype=self.dtype) for _ in range(len(dt))]

        b_out = [np.empty((self.ensemble_size, 3), dtype=self.dtype) for _ in range(len(dt))]  # type: ignore

        if sparams and len(sparams) > 0:
            s_out = [np.empty((self.ensemble_size, len(sparams)), dtype=self.dtype) for _ in range(len(dt))]  # type: ignore

        for i in range(len(dt)):  # type: ignore
            self.propagator(dt[i])  # type: ignore
            self.simulator_mag(pos[i], b_out[i])

            if sparams and len(sparams) > 0:
                s_out[i][:] = self.sparams_arr[i, sparams]
        if sparams and len(sparams) > 0:
            return b_out, s_out
        else:
            return b_out, None

    def simulator_mag(self, pos: np.ndarray, out: np.ndarray) -> None:
        raise NotImplementedError

    def update_iparams(self, iparams_arr: np.ndarray, update_weights_kernels: bool = False, kernel_mode: str = "cm") -> None:
        if update_weights_kernels:
            old_iparams = np.array(self.iparams_arr, dtype=self.dtype)
            old_weights = np.array(self.iparams_weight, dtype=self.dtype)

            self.iparams_arr = iparams_arr.astype(self.dtype)

            self.update_weights(old_iparams, old_weights, kernel_mode=kernel_mode)

            self.update_kernels(kernel_mode=kernel_mode)
        else:
            self.iparams_arr = iparams_arr.astype(self.dtype)

        generate_quaternions(self.iparams_arr, self.qs_sx, self.qs_xs)

        self.dt_t = self.dt_0

    def update_kernels(self, kernel_mode: str = "cm") -> None:
        if kernel_mode == "cm":
            self.iparams_kernel = 2 * np.cov(self.iparams_arr, rowvar=False)

            # due to aweights sometimes very small numbers are generated
            #self.iparams_kernel[np.where(self.iparams_kernel < 1e-12)] = 0
        elif kernel_mode == "lcm":
            raise NotImplementedError
        else:
            raise NotImplementedError

        # compute lower triangular matrices
        self.iparams_kernel_decomp = ldl_decomp(self.iparams_kernel)

    def update_weights(self, old_iparams: np.ndarray, old_weights: np.ndarray, kernel_mode: str = "cm") -> None:
        if kernel_mode == "cm":
            # update weights
            _numba_weight_kernel_cm(self.iparams_arr, old_iparams, self.iparams_weight, old_weights, self.iparams_kernel)
        elif kernel_mode == "lcm":
            raise NotImplementedError
        else:
            raise NotImplementedError

        self.iparams_weight /= np.sum(self.iparams_weight)

    def perturb_iparams(self, old_iparams: np.ndarray, old_weights: np.ndarray, old_kernels: np.ndarray, kernel_mode: str = "cm") -> None:
        if kernel_mode == "cm":
            # perturb particles
            _numba_perturb_kernel_cm(self.iparams_arr, old_iparams, old_weights, old_kernels, self.iparams_meta)
        elif kernel_mode == "lcm":
            raise NotImplementedError
        else:
            raise ValueError("iparams kernel(s) must be 2 or 3 dimensional")

        generate_quaternions(self.iparams_arr, self.qs_sx, self.qs_xs)

        self.dt_t = self.dt_0

    def update_iparams_meta(self) -> None:
        for k, iparam in self.iparams.items():
            ii = iparam["index"]
            dist = iparam["distribution"]
            bound = iparam["boundary"]

            self.iparams_meta[ii, 0] = iparam.get("active", 1)

            if iparam["maximum"] <= iparam["minimum"]:
                raise ValueError("invalid parameter range selected")

            if dist in ["fixed", "fixed_value"]:
                self.iparams_meta[ii, 1] = 0
                self.iparams_meta[ii, 3] = iparam["maximum"]
                self.iparams_meta[ii, 4] = iparam["minimum"]
            elif dist in ["constant", "uniform"]:
                self.iparams_meta[ii, 1] = 1
                self.iparams_meta[ii, 3] = iparam["maximum"]
                self.iparams_meta[ii, 4] = iparam["minimum"]
            elif dist in ["gaussian", "normal"]:
                self.iparams_meta[ii, 1] = 2
                self.iparams_meta[ii, 3] = iparam["maximum"]
                self.iparams_meta[ii, 4] = iparam["minimum"]
                self.iparams_meta[ii, 5] = iparam["mean"]
                self.iparams_meta[ii, 6] = iparam["std"]
            else:
                raise NotImplementedError("parameter distribution \"{0!s}\" is not implemented".format(dist))

            if bound == "continuous":
                self.iparams_meta[ii, 2] = 0
            elif bound == "periodic":
                self.iparams_meta[ii, 2] = 1

    def visualize_shape(self, *args: Any, **kwargs: Any) -> np.ndarray:
        raise NotImplementedError


@numba.njit(fastmath=True)
def _numba_perturb_select_weights(size: np.ndarray, weights_old: np.ndarray) -> np.ndarray:
    r = np.random.rand(size)
    si = -np.ones((size,), dtype=np.int64)

    for wi in range(len(weights_old)):
        r -= weights_old[wi]

        uflt = si == -1
        nflt = r < 0

        si[uflt & nflt] = wi

    # for badly normalized weights
    si[si < 0] = 0

    return si


@numba.njit(fastmath=True, parallel=False)
def _numba_perturb_kernel_cm(iparams_new: np.ndarray, iparams_old: np.ndarray, weights_old: np.ndarray, kernel_lower: np.ndarray, meta: np.ndarray) -> None:
    isel = _numba_perturb_select_weights(len(iparams_new), weights_old)
    #print("perturb start")
    for i in range(len(iparams_new)):
        si = isel[i]

        c = 0
        ct = 0
        Nc = 4

        offset = np.dot(kernel_lower, np.random.randn(len(meta), Nc))

        while True:
            acc = True
            candidate = iparams_old[si] + offset[:, c]

            for pi in range(len(meta)):
                is_oob = ((candidate[pi] > meta[pi, 3]) | (candidate[pi] < meta[pi, 4]))

                if is_oob:
                    # shift for continous variables
                    if meta[pi, 2] == 1:
                        while candidate[pi] > meta[pi, 3]:
                            candidate[pi] += meta[pi, 4] - meta[pi, 3]

                        while candidate[pi] < meta[pi, 4]:
                            candidate[pi] += meta[pi, 3] - meta[pi, 4]
                    else:
                        acc = False
                        #print("MISS")
                        break
            
            if acc:
                #print("HIT")
                break

            c += 1

            if c >= Nc - 1:
                c = 0
                ct += 1
                offset = np.dot(kernel_lower, np.random.randn(len(meta), Nc))
                
        
        iparams_new[i] = candidate
    #print("perturb end")

@numba.njit(fastmath=True)
def _numba_weight_kernel_cm(iparams_new: np.ndarray, iparams_old: np.ndarray, weights_new: np.ndarray, weights_old: np.ndarray, kernel: np.ndarray) -> None:
    inv_kernel = np.linalg.pinv(kernel).astype(iparams_old.dtype) / 2

    for i in numba.prange(len(iparams_new)):
        nw = 0

        for j in range(len(iparams_old)):
            v = _numba_cov_dist(iparams_new[i], iparams_old[j], inv_kernel)

            nw += np.exp(np.log(weights_old[j]) - v)
        
        weights_new[i] = 1 / nw


@numba.njit(fastmath=True, inline="always")
def _numba_cov_dist(x1: np.ndarray, x2: np.ndarray, cov: np.ndarray) -> np.ndarray:
    dx = (x1 - x2).astype(cov.dtype)
    return np.dot(dx, np.dot(cov, dx))
