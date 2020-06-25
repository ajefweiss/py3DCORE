# -*- coding: utf-8 -*-

"""parameters.py

Implements the base 3DCORE parameter classes.
"""

import logging
import numba
import numpy as np
import scipy


class Base3DCOREParameters(object):
    params_dict = None

    dtype = None
    qindices = None

    def __init__(self, params_dict, **kwargs):
        """Initialize Base3DCOREParameters class.

        Parameters
        ----------
        params_dict : dict
            Parameter dictionary.

        Other Parameters
        ----------------
        dtype: type
            Data type, by default np.float32.
        qindices: np.ndarray
            Quaternion array columns, by default [1, 2, 3].
        """
        self.params_dict = params_dict

        self.dtype = kwargs.get("dtype", np.float32)
        self.qindices = np.array(kwargs.get("qindices", np.array([1, 2, 3]))).ravel()

        # update arrays
        self._update_arr()

    def __getitem__(self, key):
        for dk in self.params_dict:
            if self.params_dict[dk]["index"] == key or self.params_dict[dk]["name"] == key:
                return self.params_dict[dk]

        if isinstance(key, int):
            raise KeyError("key %i is out of range", key)
        else:
            raise KeyError("key \"%s\" does not exist", key)

    def __iter__(self):
        self.iter_n = 0

        return self

    def __len__(self):
        return len(self.params_dict.keys())

    def __next__(self):
        if self.iter_n < len(self):
            result = self[self.iter_n]

            self.iter_n += 1

            return result
        else:
            raise StopIteration

    def generate(self, iparams_arr, use_gpu=False, **kwargs):
        """Generate initial parameters

        Parameters
        ----------
        iparams : Union[np.ndarray,
            List[numba.cuda.cudadrv.devicearray.DeviceNDArray]]
            Initial parameters array.
        use_gpu : bool
            GPU flag, by default False.

        Other Parameters
        ----------------
        rng_states : List[numba.cuda.cudadrv.devicearray.DeviceNDArray]
            CUDA rng state array.
        """
        size = len(iparams_arr)

        if use_gpu:
            raise NotImplementedError
        else:
            for param in self:
                index = param["index"]

                if param["distribution"] == "fixed":
                    iparams_arr[:, index] = param["fixed_value"]
                elif param["distribution"] == "uniform":
                    maxv = param["maximum"]
                    minv = param["minimum"]

                    iparams_arr[:, index] = np.random.rand(
                        size) * (maxv - minv) + minv
                elif param["distribution"] == "gaussian":
                    maxv = param["maximum"]
                    minv = param["minimum"]

                    kwargs = {
                        "loc": param["mean"],
                        "scale": param["std"]
                    }

                    iparams_arr[:, index] = draw_numbers(np.random.normal, maxv, minv, size,
                                                         **kwargs)
                elif param["distribution"] == "gamma":
                    maxv = param["maximum"]
                    minv = param["minimum"]

                    kwargs = {
                        "loc": param["shape"],
                        "scale": param["scale"]
                    }

                    iparams_arr[:, index] = draw_numbers(np.random.gamma, maxv, minv, size,
                                                         **kwargs)
                else:
                    raise NotImplementedError

        self._update_arr()

    def perturb(self, iparams_arr, particles, weights, kernels_lower, use_gpu=False, **kwargs):
        """Perburb particles.

        Parameters
        ----------
        iparams : Union[np.ndarray,
            List[numba.cuda.cudadrv.devicearray.DeviceNDArray]]
            Initial parameters array.
        particles : Union[np.ndarray,
            List[numba.cuda.cudadrv.devicearray.DeviceNDArray]]
            Particles.
        weights : Union[np.ndarray,
            List[numba.cuda.cudadrv.devicearray.DeviceNDArray]]
            Weight array.
        kernels_lower : Union[np.ndarray,
            List[numba.cuda.cudadrv.devicearray.DeviceNDArray]]
            Transition kernels in the form of a lower triangular matrix.
            Number of dimensions determines the type of method used.
        use_gpu : bool
            GPU flag, by default False.

        Other Parameters
        ----------------
        rng_states : List[numba.cuda.cudadrv.devicearray.DeviceNDArray]
            CUDA rng state array.
        """
        if use_gpu:
            raise NotImplementedError
        else:
            if kernels_lower.ndim == 1:
                raise DeprecationWarning
            elif kernels_lower.ndim == 2:
                # perturbation with one covariance matrix
                _numba_perturb_particles_kernel(iparams_arr, particles, weights, kernels_lower,
                                                self.type_arr, self.bound_arr, self.maxv_arr,
                                                self.minv_arr)
            elif kernels_lower.ndim == 3:
                # perturbation with local covariance matrices
                _numba_perturb_particles_kernels(iparams_arr, particles, weights, kernels_lower,
                                                 self.type_arr, self.bound_arr, self.maxv_arr,
                                                 self.minv_arr)
            else:
                raise ValueError("kernel array must be 3-dimensional or lower")

    def weight(self, particles, particles_old, weights, weights_old, kernels, use_gpu=False,
               **kwargs):
        """Update particle weights.

        Parameters
        ----------
        particles : Union[np.ndarray,
            List[numba.cuda.cudadrv.devicearray.DeviceNDArray]]
            Particle array.
        particles_old : Union[np.ndarray,
            List[numba.cuda.cudadrv.devicearray.DeviceNDArray]]
            Old particle array.
        weights : Union[np.ndarray,
            List[numba.cuda.cudadrv.devicearray.DeviceNDArray]]
            Paritcle weights array.
        weights_old : Union[np.ndarray,
            List[numba.cuda.cudadrv.devicearray.DeviceNDArray]]
            Old particle weights array.
        kernels : Union[np.ndarray,
            List[numba.cuda.cudadrv.devicearray.DeviceNDArray]]
            Transition kernels. Number of dimensions determines the type of method used.
        use_gpu : bool
            GPU flag, by default False.

        Other Parameters
        ----------------
        exclude_priors: bool
            Exclude prior distributions from weighting, by default False.
        """
        logger = logging.getLogger(__name__)

        if use_gpu:
            raise NotImplementedError
        else:
            if kernels.ndim == 1:
                raise DeprecationWarning
            elif kernels.ndim == 2:
                _numba_calculate_weights_kernel(particles, particles_old,
                                                weights, weights_old, kernels)
            elif kernels.ndim == 3:
                _numba_calculate_weights_kernels(particles, particles_old,
                                                 weights, weights_old, kernels)
            else:
                raise ValueError("kernel array must be 3-dimensional or lower")

            if not kwargs.get("exclude_priors", False):
                _numba_calculate_weights_priors(
                    particles, weights, self.type_arr, self.dp1_arr, self.dp2_arr)

            # a hack for big weights
            quant = np.quantile(weights, 0.99)

            for i in range(0, len(weights)):
                if weights[i] > quant:
                    weights[i] = 0

            nanc = np.sum(np.isnan(weights))
            infc = np.sum(weights == np.inf)

            if nanc + infc > 0:
                logger.warning("detected %i NaN/inf weights, setting to 0", nanc + infc)
                weights[np.isnan(weights)] = 0
                weights[weights == np.inf] = 0

            # renormalize
            wsum = np.sum(weights)

            for i in range(0, len(weights)):
                weights[i] /= wsum

    def _update_arr(self):
        """Convert params_dict to arrays.
        """
        self.active = np.zeros((len(self), ), dtype=self.dtype)
        self.bound_arr = np.zeros((len(self), ), dtype=self.dtype)
        self.maxv_arr = np.zeros((len(self), ), dtype=self.dtype)
        self.dp1_arr = np.zeros((len(self), ), dtype=self.dtype)
        self.dp2_arr = np.zeros((len(self), ), dtype=self.dtype)
        self.minv_arr = np.zeros((len(self), ), dtype=self.dtype)
        self.type_arr = np.zeros((len(self), ), dtype=self.dtype)

        for param in self:
            index = param["index"]

            self.active[index] = param.get("active", 1)

            if param["distribution"] == "fixed":
                self.type_arr[index] = 0
                self.maxv_arr[index] = param["fixed_value"]
            elif param["distribution"] == "uniform":
                self.type_arr[index] = 1
                self.maxv_arr[index] = param["maximum"]
                self.minv_arr[index] = param["minimum"]
            elif param["distribution"] == "gaussian":
                self.type_arr[index] = 2
                self.maxv_arr[index] = param["maximum"]
                self.minv_arr[index] = param["minimum"]
                self.dp1_arr[index] = param.get("mean", 0)
                self.dp2_arr[index] = param.get("std", 0)
            elif param["distribution"] == "gamma":
                self.type_arr[index] = 3
                self.maxv_arr[index] = param["maximum"]
                self.minv_arr[index] = param["minimum"]
                self.dp1_arr[index] = param.get("shape", 0)
                self.dp2_arr[index] = param.get("scale", 0)
            else:
                raise NotImplementedError

            if param["boundary"] == "continuous":
                self.bound_arr[index] = 0
            elif param["boundary"] == "periodic":
                self.bound_arr[index] = 1
            else:
                raise NotImplementedError


def draw_numbers(func, maxv, minv, size, **kwargs):
    """Draw n random numbers from func, within the interval [minv, maxv].
    """
    i = 0

    numbers = func(size=size, **kwargs)
    

    while True:
        filter = ((numbers > maxv) | (numbers < minv))

        if np.sum(filter) == 0:
            break

        numbers[filter] = func(size=len(filter), **kwargs)[filter]

        i += 1

        if i > 1000:
            raise RuntimeError("drawing numbers inefficiently (%i/%i after 1000 iterations)",
                               len(filter), size)

    return numbers


@numba.njit
def _numba_perturb_particles_kernel(iparams_arr, particles, weights, kernel_lower, type_arr,
                                    bound_arr, maxv_arr, minv_arr):
    for i in range(len(iparams_arr)):
        # particle selector si
        r = np.random.rand(1)[0]

        for j in range(len(weights)):
            r -= weights[j]

            if r <= 0:
                si = j
                break

        # draw candidates until a candidite is within the valid range
        c = 0
        Nc = 25

        perturbations = np.dot(kernel_lower, np.random.randn(len(particles[si]), Nc))

        while True:
            candidate = particles[si] + perturbations[:, c]
            accepted = True

            for k in range(len(candidate)):
                if candidate[k] > maxv_arr[k]:
                    if bound_arr[k] == 0:
                        accepted = False
                        break
                    elif bound_arr[k] == 1:
                        candidate[k] = minv_arr[k] + candidate[k] - maxv_arr[k]
                    else:
                        raise NotImplementedError

                if candidate[k] < minv_arr[k]:
                    if bound_arr[k] == 0:
                        accepted = False
                        break
                    elif bound_arr[k] == 1:
                        candidate[k] = maxv_arr[k] + candidate[k] - minv_arr[k]
                    else:
                        raise NotImplementedError

            if accepted:
                break

            c += 1

            if c >= Nc - 1:
                c = 0
                perturbations = np.dot(kernel_lower, np.random.randn(len(particles[si]), Nc))

        iparams_arr[i] = candidate


@numba.njit
def _numba_perturb_particles_kernels(iparams_arr, particles, weights, kernels_lower, type_arr,
                                     bound_arr, maxv_arr, minv_arr):
    for i in range(len(iparams_arr)):
        # particle selector si
        r = np.random.rand(1)[0]

        for j in range(len(weights)):
            r -= weights[j]

            if r <= 0:
                si = j
                break

        # draw candidates until a candidate is within the valid range
        c = 0
        Nc = 25

        perturbations = np.dot(kernels_lower[si], np.random.randn(len(particles[si]), Nc))

        while True:
            candidate = particles[si] + perturbations[:, c]
            accepted = True

            for k in range(len(candidate)):
                if candidate[k] > maxv_arr[k]:
                    if bound_arr[k] == 0:
                        accepted = False

                        break
                    elif bound_arr[k] == 1:
                        while candidate[k] > maxv_arr[k]:
                            candidate[k] = minv_arr[k] + candidate[k] - maxv_arr[k]
                    else:
                        raise NotImplementedError

                if candidate[k] < minv_arr[k]:
                    if bound_arr[k] == 0:
                        accepted = False

                        break
                    elif bound_arr[k] == 1:
                        while candidate[k] < minv_arr[k]:
                            candidate[k] = maxv_arr[k] + candidate[k] - minv_arr[k]
                    else:
                        raise NotImplementedError

            if accepted:
                break

            c += 1

            if c >= Nc - 1:
                c = 0
                perturbations = np.dot(kernels_lower[si], np.random.randn(len(particles[si]), Nc))

        iparams_arr[i] = candidate


@numba.njit(parallel=True, nogil=True)
def _numba_calculate_weights_kernel(particles, particles_prev, weights, weights_prev, kernel):
    inv_kernel = np.linalg.pinv(kernel).astype(particles.dtype)

    for i in numba.prange(len(particles)):
        nw = 0

        for j in range(len(particles_prev)):
            v = _numba_calculate_weights_reduce(particles[i], particles_prev[j], inv_kernel)

            nw += weights_prev[j] * np.exp(-v)

        weights[i] = 1 / nw


@numba.njit(parallel=True, nogil=True)
def _numba_calculate_weights_kernels(particles, particles_prev, weights, weights_prev, kernels):
    inv_kernels = np.zeros_like(kernels).astype(particles.dtype)

    # compute kernel inverses
    for i in numba.prange(len(kernels)):
        inv_kernels[i] = np.linalg.pinv(kernels[i])

    for i in numba.prange(len(particles)):
        nw = 0

        for j in range(len(particles_prev)):
            v = _numba_calculate_weights_reduce(particles[i], particles_prev[j], inv_kernels[j])
            nw += weights_prev[j] * np.exp(-v)

        weights[i] = 1 / nw


@numba.njit
def _numba_calculate_weights_reduce(x1, x2, A):
    dx = (x1 - x2).astype(A.dtype)
    return np.dot(dx, np.dot(A, dx))


# @numba.njit(parallel=True)
def _numba_calculate_weights_priors(particles, weights, type_arr, dp1_arr, dp2_arr):
    for i in range(len(weights)):
        for j in range(len(type_arr)):
            if type_arr[j] <= 1:
                # fixed or uniform, no weighting
                pass
            elif type_arr[j] == 2:
                # gaussian distribution
                weights[i] *= np.exp(-0.5 * (particles[i, j] -
                                            dp1_arr[j])**2/dp2_arr[j]**2) / dp2_arr[j]
            elif type_arr[j] == 3:
                # gamma distribution
                weights[i] *= np.exp(-particles[i, j]/dp2_arr[j]) * particles[i, j] ** (
                    dp1_arr[j] - 1) / scipy.special.gamma(dp1_arr[j]) / dp2_arr[j]**dp1_arr[j]
            else:
                raise NotImplementedError
