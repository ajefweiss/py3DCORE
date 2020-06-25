# -*- coding: utf-8 -*-

"""model.py

Implements the base 3DCORE model classes.
"""

import logging
import numba.cuda as cuda
import numpy as np

from itertools import product
from numba.cuda.random import create_xoroshiro128p_states as cxoro128p
from scipy.optimize import least_squares

from py3dcore._extnumba import set_random_seed
from py3dcore.quaternions import generate_quaternions


class Base3DCOREModel(object):
    """Base 3DCORE model class.
    """
    parameters = None
    dtype = None

    launch = None
    iparams_arr = sparams_arr = None

    qs_sx = qs_xs = None

    rng_states = None

    g = f = h = p = None

    runs = 0
    use_gpu = False

    # internal
    _tva = _tvb = None
    _b = None

    def __init__(self, launch, functions, parameters, sparams_count, runs, use_gpu=False, **kwargs):
        """Initialize Base3DCOREModel.

        Initial parameters must be generated seperately (see generate_iparams).

        Parameters
        ----------
        launch : datetime.datetime
            Initial datetime.
        functions : dict
            Propagation, magnetic and transformation functions as a dictionary.
        parameters : py3dcore.model.Base3DCOREParameters
            Model parameters.
        sparams_count: int
            Number of state parameters.
        runs : int
            Number of parallel model runs.
        use_gpu : bool, optional
            GPU flag, by default False.

        Other Parameters
        ----------------
        cuda_device: int
            CUDA device, by default 0.

        Raises
        ------
        ValueError
            If the number of model runs is not divisible by 256 and gpu flag is set.
        """
        logger = logging.getLogger(__name__)

        self.launch = launch.timestamp()

        self.g = functions["g"]
        self.f = functions["f"]
        self.h = functions["h"]
        self.p = functions["p"]

        if self.use_gpu and runs > 512 and runs % 32 != 0:
            logger.exception("for gpu mode the number of runs must be a multiple of 32")
            raise ValueError("for gpu mode the number of runs must be a multiple of 32")

        self.parameters = parameters
        self.dtype = parameters.dtype
        self.runs = runs

        if use_gpu and cuda.is_available():
            cuda.select_device(kwargs.get("cuda_device", 0))
            self.use_gpu = True
        else:
            self.use_gpu = False

        if self.use_gpu:
            self.iparams_arr = cuda.device_array((self.runs, len(self.parameters)),
                                                 dtype=self.dtype)
            self.sparams_arr = cuda.device_array((self.runs, sparams_count),
                                                 dtype=self.dtype)

            self.qs_sx = cuda.device_array((self.runs, 4), dtype=self.dtype)
            self.qs_xs = cuda.device_array((self.runs, 4), dtype=self.dtype)

            self._tva = cuda.device_array((self.runs, 3), dtype=self.dtype)
            self._tvb = cuda.device_array((self.runs, 3), dtype=self.dtype)
        else:
            self.iparams_arr = np.empty(
                (self.runs, len(self.parameters)), dtype=self.dtype)
            self.sparams_arr = np.empty(
                (self.runs, sparams_count), dtype=self.dtype)

            self.qs_sx = np.empty((self.runs, 4), dtype=self.dtype)
            self.qs_xs = np.empty((self.runs, 4), dtype=self.dtype)

            self._tva = np.empty((self.runs, 3), dtype=self.dtype)
            self._tvb = np.empty((self.runs, 3), dtype=self.dtype)

    def generate_iparams(self, seed):
        """Generate initial parameters.

        Parameters
        ----------
        seed : int
            Random number seed.
        """
        if self.use_gpu:
            self.rng_states = cxoro128p(self.runs, seed=seed)
        else:
            set_random_seed(seed)

        self.parameters.generate(
            self.iparams_arr, use_gpu=self.use_gpu, rng_states=self.rng_states)

        generate_quaternions(self.iparams_arr, self.qs_sx, self.qs_xs, use_gpu=self.use_gpu,
                             indices=self.parameters.qindices)

    def perturb_iparams(self, particles, weights, kernels):
        """Generate initial parameters by perturbing given particles.

        Parameters
        ----------
        particles : np.ndarray
            Particles.
        weights : np.ndarray
            Particle weights.
        kernels : np.ndarray
            Transition kernels.
        """
        self.parameters.perturb(self.iparams_arr, particles, weights, kernels, use_gpu=self.use_gpu,
                                rng_states=self.rng_states)

        generate_quaternions(self.iparams_arr, self.qs_sx, self.qs_xs, use_gpu=self.use_gpu,
                             indices=self.parameters.qindices)

    def update_iparams(self, iparams_arr, seed=None):
        """Set initial parameters to given array. Array size must match the length of "self.runs".

        Parameters
        ----------
        iparams : np.ndarray
            Initial parameters array.
        seed : int, optional
            Random seed, by default None.
        """
        if seed:
            if self.use_gpu:
                self.rng_states = cxoro128p(self.runs, seed=seed)
            else:
                set_random_seed(seed)

        self.iparams_arr = iparams_arr.astype(self.dtype)

        generate_quaternions(self.iparams_arr, self.qs_sx, self.qs_xs, use_gpu=self.use_gpu,
                             indices=self.parameters.qindices)

    def propagate(self, t):
        """Propagate all model runs to datetime t.

        Parameters
        ----------
        t : datetime.datetime
            Datetime.
        """
        dt = self.dtype(t.timestamp() - self.launch)

        self.p(dt, self.iparams_arr, self.sparams_arr, use_gpu=self.use_gpu)

    def sim_field(self, s, b):
        """Calculate magnetic field vectors at (s) coordinates.

        Parameters
        ----------
        s : np.ndarray
            Position vector array.
        b : Union[np.ndarray, numba.cuda.cudadrv.devicearray.DeviceNDArray]
            Magnetic field output array.
        """
        self.transform_sq(s, q=self._tva)
        self.h(self._tva, b, self.iparams_arr, self.sparams_arr, self.qs_xs,
               use_gpu=self.use_gpu, rng_states=self.rng_states)

    def sim_fields(self, t, pos, b=None):
        """Calculate magnetic field vectors at (s) coordinates and at times t (datetime timestamps).

        Parameters
        ----------
        t : list[float]
            Evaluation datetimes.
        pos : np.ndarray
            Position vector array at evaluation datetimes.
        b : Union[List[np.ndarray], List[numba.cuda.cudadrv.devicearray.DeviceNDArray]], optional
            Magnetic field temporary array, by default None.

        Returns
        -------
        Union[list[np.ndarray], list[numba.cuda.cudadrv.devicearray.DeviceNDArray]]
            List of magnetic field output arrays at evaluation datetimes.

        Raises
        ------
        ValueError
            If pos array has invalid dimension (must be 1 or 2).
        """
        logger = logging.getLogger(__name__)

        if np.array(pos).ndim == 1:
            pos = [np.array(pos, dtype=self.dtype) for _ in range(0, len(t))]
        elif np.array(pos).ndim == 2:
            pos = [np.array(p, dtype=self.dtype) for p in pos]
        else:
            logger.exception("position array has invalid dimension")
            raise ValueError("position array has invalid dimension")

        if self.use_gpu:
            pos = [cuda.to_device(p) for p in pos]

        if b is None:
            if self.use_gpu:
                if self._b is None or len(self._b[0]) != len(t) \
                        or self._b[0][0].shape != (self.runs, 3):
                    self._b = [cuda.device_array((self.runs, 3), dtype=self.dtype)
                               for _ in range(0, len(t))]
            else:
                if self._b is None or len(self._b) != len(t) or self._b[0].shape != (self.runs, 3):
                    self._b = [np.empty((self.runs, 3), dtype=self.dtype)
                               for _ in range(0, len(t))]

            b = self._b

        for i in range(0, len(t)):
            self.propagate(t[i])
            self.sim_field(pos[i], b[i])

        return b

    def transform_sq(self, s, q):
        """Transform (s) coordinates to (q) coordinates.

        Parameters
        ----------
        s : Union[np.ndarray, numba.cuda.cudadrv.devicearray.DeviceNDArray]
            The (s) coordinate array.
        q : Union[np.ndarray, numba.cuda.cudadrv.devicearray.DeviceNDArray]
            The (q) coordinate array.
        """
        self.f(s, q, self.iparams_arr, self.sparams_arr, self.qs_sx,
               use_gpu=self.use_gpu)

    def transform_qs(self, s, q):
        """Transform (q) coordinates to (s) coordinates.

        Parameters
        ----------
        q : Union[np.ndarray, numba.cuda.cudadrv.devicearray.DeviceNDArray]
            The (q) coordinate array.
        s : Union[np.ndarray, numba.cuda.cudadrv.devicearray.DeviceNDArray]
            The (s) coordinate array.
        """
        self.g(q, s, self.iparams_arr, self.sparams_arr, self.qs_xs,
               use_gpu=self.use_gpu)


class Toroidal3DCOREModel(Base3DCOREModel):
    def visualize_fieldline(self, q0, index=0, steps=1000, step_size=0.01):
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
        # only implemented in cpu mode
        if self.use_gpu:
            raise NotImplementedError("visualize_fieldline not supported in gpu mode")

        _tva = np.empty((3,), dtype=self.dtype)
        _tvb = np.empty((3,), dtype=self.dtype)

        self.g(q0, _tva, self.iparams_arr[index], self.sparams_arr[index], self.qs_xs[index],
               use_gpu=False)

        fl = [np.array(_tva, dtype=self.dtype)]

        def iterate(s):
            self.f(s, _tva, self.iparams_arr[index], self.sparams_arr[index], self.qs_sx[index],
                   use_gpu=False)
            self.h(_tva, _tvb, self.iparams_arr[index], self.sparams_arr[index], self.qs_xs[index],
                   use_gpu=False, bounded=False)
            return _tvb / np.linalg.norm(_tvb)

        while len(fl) < steps:
            # use implicit method and least squares for calculating the next step
            sol = getattr(least_squares(
                lambda x: x - fl[-1] - step_size *
                iterate((x.astype(self.dtype) + fl[-1]) / 2),
                fl[-1]), "x")

            fl.append(np.array(sol.astype(self.dtype)))

        fl = np.array(fl, dtype=self.dtype)

        return fl

    def visualize_wireframe(self, index=0, r=1.0):
        """Generate model wireframe.

        Parameters
        ----------
        index : int, optional
            Model run index, by default 0.
        r : float, optional
            Surface radius (r=1 equals the boundary of the flux rope), by default 1.0.

        Returns
        -------
        np.ndarray
            Wireframe array (to be used with plot_wireframe).
        """
        r = np.array([np.abs(r)], dtype=self.dtype)
        d = 10
        c = 360 // d + 1
        u = np.radians(np.r_[0:360. + d:d])
        v = np.radians(np.r_[0:360. + d:d])

        # combination of wireframe points in (q)
        arr = np.array(list(product(r, u, v)), dtype=self.dtype).reshape(c ** 2, 3)

        # only implemented in cpu mode
        if self.use_gpu:
            raise NotImplementedError("visualize_fieldline not supported in gpu mode")

        for i in range(0, len(arr)):
            self.g(arr[i], arr[i], self.iparams_arr[index], self.sparams_arr[index],
                   self.qs_xs[index], use_gpu=False)

        return arr.reshape((c, c, 3))
