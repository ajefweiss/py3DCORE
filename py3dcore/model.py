# -*- coding: utf-8 -*-

"""base.py

Implements the base 3DCORE model classes.
"""

import logging
import numba.cuda as cuda
import numpy as np

from itertools import product
from numba.cuda.random import create_xoroshiro128p_states as cxoro128p
from scipy.optimize import least_squares

from py3dcore._extnumba import set_random_seed
from py3dcore.rotqs import generate_quaternions


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
    use_cuda = False

    # internal
    _tva = _tvb = None
    _b, _sp = None, None

    def __init__(self, launch, functions, parameters, sparams_count, runs, **kwargs):
        """Initialize Base3DCOREModel.

        Initial parameters must be generated seperately (see generate_iparams).

        Parameters
        ----------
        launch : datetime.datetime
            Initial datetime.
        functions : dict
            Propagation, magnetic and transformation functions as dict.
        parameters : py3dcore.model.Base3DCOREParameters
            Model parameters instance.
        sparams_count: int
            Number of state parameters.
        runs : int
            Number of parallel model runs.

        Other Parameters
        ----------------
        cuda_device: int
            CUDA device, by default 0.
        use_cuda : bool, optional
            CUDA flag, by default False.

        Raises
        ------
        ValueError
            If the number of model runs is not divisible by 256 and CUDA flag is set.
            If cuda is selected but not available.
        """
        self.launch = launch.timestamp()

        self.g = functions["g"]
        self.f = functions["f"]
        self.h = functions["h"]
        self.p = functions["p"]

        self.parameters = parameters
        self.dtype = parameters.dtype
        self.runs = runs

        self.use_cuda = kwargs.get("use_cuda", False)

        if self.use_cuda and cuda.is_available():
            raise NotImplementedError("CUDA functionality is not available yet")
            # cuda.select_device(kwargs.get("cuda_device", 0))
            # self.use_cuda = True
        elif self.use_cuda:
            raise ValueError("cuda is not available")

        if self.use_cuda:
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
        if self.use_cuda:
            self.rng_states = cxoro128p(self.runs, seed=seed)
        else:
            set_random_seed(seed)

        self.parameters.generate(
            self.iparams_arr, use_cuda=self.use_cuda, rng_states=self.rng_states)

        generate_quaternions(self.iparams_arr, self.qs_sx, self.qs_xs, use_cuda=self.use_cuda,
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
        self.parameters.perturb(self.iparams_arr, particles, weights, kernels,
                                use_cuda=self.use_cuda, rng_states=self.rng_states)

        generate_quaternions(self.iparams_arr, self.qs_sx, self.qs_xs, use_cuda=self.use_cuda,
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
            if self.use_cuda:
                self.rng_states = cxoro128p(self.runs, seed=seed)
            else:
                set_random_seed(seed)

        self.iparams_arr = iparams_arr.astype(self.dtype)

        generate_quaternions(self.iparams_arr, self.qs_sx, self.qs_xs, use_cuda=self.use_cuda,
                             indices=self.parameters.qindices)

    def propagate(self, t):
        """Propagate all model runs to datetime t.

        Parameters
        ----------
        t : datetime.datetime
            Datetime.
        """
        dt = self.dtype(t.timestamp() - self.launch)

        self.p(dt, self.iparams_arr, self.sparams_arr, use_cuda=self.use_cuda)

    def get_field(self, s, b):
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
               use_cuda=self.use_cuda, rng_states=self.rng_states)

    def get_sparams(self, sparams, sparams_out):
        """Get selected sparams values.

        Parameters
        ----------
        sparams : np.ndarray
            Selected sparams indices.
        sparams_out : np.ndarray
            Output array.
        """
        if self.use_cuda:
            raise NotImplementedError
        else:
            for i in range(len(sparams)):
                sparam = sparams[i]
                sparams_out[i] = self.sparams_arr[i, sparam]

    def sim_fields(self, *args, **kwargs):
        """Legacy dummy for simulate
        """
        if "b" in kwargs:
            kwargs["b_out"] = kwargs.pop("b")

        return self.simulate(*args, **kwargs)

    def simulate(self, t, pos, sparams=None, b_out=None, sparams_out=None):
        """Calculate magnetic field vectors at (s) coordinates and at times t (datetime timestamps).
        Additionally returns any selected sparams.

        Parameters
        ----------
        t : list[float]
            Evaluation datetimes.
        pos : np.ndarray
            Position vector array at evaluation datetimes.
        sparams : list[int]
            List of state parameters to return.
        b_out : Union[List[np.ndarray],
                List[numba.cuda.cudadrv.devicearray.DeviceNDArray]], optional
            Magnetic field temporary array, by default None.
        sparams_out : Union[List[np.ndarray],
                      List[numba.cuda.cudadrv.devicearray.DeviceNDArray]], optional
            State parameters temporary array, by default None.

        Returns
        -------
        Union[list[np.ndarray], list[numba.cuda.cudadrv.devicearray.DeviceNDArray]],
        Union[list[np.ndarray], list[numba.cuda.cudadrv.devicearray.DeviceNDArray]]
            List of magnetic field output and state parameters at evaluation datetimes.

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

        if self.use_cuda:
            pos = [cuda.to_device(p) for p in pos]

        if b_out is None:
            if self.use_cuda:
                if self._b is None or len(self._b[0]) != len(t) \
                        or self._b[0][0].shape != (self.runs, 3):
                    self._b = [cuda.device_array((self.runs, 3), dtype=self.dtype)
                               for _ in range(0, len(t))]
            else:
                if self._b is None or len(self._b) != len(t) or self._b[0].shape != (self.runs, 3):
                    self._b = [np.empty((self.runs, 3), dtype=self.dtype)
                               for _ in range(0, len(t))]

            b_out = self._b

        if sparams and len(sparams) > 0 and sparams_out is None:
            if self.use_cuda:
                if self._sp is None or len(self._sp[0]) != len(t) \
                        or self._sp[0][0].shape != (self.runs, len(sparams)):
                    self._sp = [cuda.device_array((self.runs, len(sparams)), dtype=self.dtype)
                                for _ in range(0, len(t))]
            else:
                if self._sp is None or len(self._sp) != len(t) or \
                        self._sp[0].shape != (self.runs, len(sparams)):
                    self._sp = [np.empty((self.runs, len(sparams)), dtype=self.dtype)
                                for _ in range(0, len(t))]

            sparams_out = self._sp

        for i in range(0, len(t)):
            self.propagate(t[i])
            self.get_field(pos[i], b_out[i])

            if sparams and len(sparams) > 0:
                self.get_sparams(sparams, sparams_out[i])

        if sparams and len(sparams) > 0:
            return b_out, sparams_out
        else:
            return b_out

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
               use_cuda=self.use_cuda)

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
               use_cuda=self.use_cuda)


class Toroidal3DCOREModel(Base3DCOREModel):
    def plot_fieldline(self, ax, fl, arrows=None, **kwargs):
        """Plot magnetic field line.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Matplotlib axes.
        fl : np.ndarray
            Field line in (s) coordinates.
        """
        if arrows:
            handles = []

            for index in list(range(len(fl)))[::arrows][:-1]:
                x, y, z = fl[index]
                xv, yv, zv = fl[index + arrows // 2] - fl[index]

                handles.append(ax.quiver(
                    x, y, z, xv, yv, zv, **kwargs
                ))

            return handles
        else:
            return ax.plot(*fl.T, **kwargs)

    def visualize_fieldline(self, q0, index=0, sign=1, steps=1000, step_size=0.005):
        """Integrates along the magnetic field lines starting at a point q0 in (q) coordinates and
        returns the field lines in (s) coordinates.

        Parameters
        ----------
        q0 : np.ndarray
            Starting point in (q) coordinates.
        index : int, optional
            Model run index, by default 0.
        sign : int, optional
            Integration direction, by default 1.
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
        if self.use_cuda:
            raise NotImplementedError("visualize_fieldline not supported in cuda mode")

        if self.iparams_arr[index, -1] > 0 or self.iparams_arr[index, -1] < 0:
            raise Warning("cannot generate field lines with non-zero noise level")

        _tva = np.empty((3,), dtype=self.dtype)
        _tvb = np.empty((3,), dtype=self.dtype)

        sign = sign / np.abs(sign)

        self.g(q0, _tva, self.iparams_arr[index], self.sparams_arr[index], self.qs_xs[index],
               use_cuda=False)

        fl = [np.array(_tva, dtype=self.dtype)]

        def iterate(s):
            self.f(s, _tva, self.iparams_arr[index], self.sparams_arr[index], self.qs_sx[index],
                   use_cuda=False)
            self.h(_tva, _tvb, self.iparams_arr[index], self.sparams_arr[index], self.qs_xs[index],
                   use_cuda=False, bounded=False)
            return sign * _tvb / np.linalg.norm(_tvb)

        while len(fl) < steps:
            # use implicit method and least squares for calculating the next step
            sol = getattr(least_squares(
                lambda x: x - fl[-1] - step_size *
                iterate((x.astype(self.dtype) + fl[-1]) / 2),
                fl[-1]), "x")

            fl.append(np.array(sol.astype(self.dtype)))

        fl = np.array(fl, dtype=self.dtype)

        return fl

    def visualize_fieldline_dpsi(self, q0, dpsi=np.pi, index=0, sign=1, step_size=0.005):
        """Integrates along the magnetic field lines starting at a point q0 in (q) coordinates and
        returns the field lines in (s) coordinates.

        Parameters
        ----------
        q0 : np.ndarray
            Starting point in (q) coordinates.
        dpsi: float
            Delta psi to integrate by, default np.pi
        index : int, optional
            Model run index, by default 0.
        sign : int, optional
            Integration direction, by default 1.
        step_size : float, optional
            Integration step size, by default 0.01.

        Returns
        -------
        np.ndarray
            Integrated magnetic field lines in (s) coordinates.
        """
        # only implemented in cpu mode
        if self.use_cuda:
            raise NotImplementedError("visualize_fieldline not supported in cuda mode")

        if self.iparams_arr[index, -1] > 0 or self.iparams_arr[index, -1] < 0:
            raise Warning("cannot generate field lines with non-zero noise level")

        _tva = np.empty((3,), dtype=self.dtype)
        _tvb = np.empty((3,), dtype=self.dtype)

        sign = sign / np.abs(sign)

        self.g(q0, _tva, self.iparams_arr[index], self.sparams_arr[index], self.qs_xs[index],
               use_cuda=False)

        fl = [np.array(_tva, dtype=self.dtype)]

        def iterate(s):
            self.f(s, _tva, self.iparams_arr[index], self.sparams_arr[index], self.qs_sx[index],
                   use_cuda=False)
            self.h(_tva, _tvb, self.iparams_arr[index], self.sparams_arr[index], self.qs_xs[index],
                   use_cuda=False, bounded=False)
            return sign * _tvb / np.linalg.norm(_tvb)

        psi_pos = q0[1]
        dpsi_count = 0

        while dpsi_count < dpsi:
            # use implicit method and least squares for calculating the next step
            sol = getattr(least_squares(
                lambda x: x - fl[-1] - step_size *
                iterate((x.astype(self.dtype) + fl[-1]) / 2),
                fl[-1]), "x")

            fl.append(np.array(sol.astype(self.dtype)))

            self.f(fl[-1], _tva, self.iparams_arr[index], self.sparams_arr[index],
                   self.qs_sx[index], use_cuda=False)

            dpsi_count += np.abs(psi_pos - _tva[1])
            psi_pos = _tva[1]

        fl = np.array(fl, dtype=self.dtype)

        return fl

    def visualize_wireframe(self, index=0, r=1.0, d=10):
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

        c = 360 // d + 1
        u = np.radians(np.r_[0:360. + d:d])
        v = np.radians(np.r_[0:360. + d:d])

        # combination of wireframe points in (q)
        arr = np.array(list(product(r, u, v)), dtype=self.dtype).reshape(c ** 2, 3)

        # only implemented in cpu mode
        if self.use_cuda:
            raise NotImplementedError("visualize_wireframe not supported in cuda mode")

        for i in range(0, len(arr)):
            self.g(arr[i], arr[i], self.iparams_arr[index], self.sparams_arr[index],
                   self.qs_xs[index], use_cuda=False)

        return arr.reshape((c, c, 3))

    def visualize_wireframe_dpsi(self, psi0, dpsi, index=0, r=1.0, d=10):
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

        deg_min = 180 * psi0 / np.pi
        deg_max = 180 * (psi0 + dpsi) / np.pi

        u = np.radians(np.r_[deg_min:deg_max + d:d])
        v = np.radians(np.r_[0:360. + d:d])

        c1 = len(u)
        c2 = len(v)

        # combination of wireframe points in (q)
        arr = np.array(list(product(r, u, v)), dtype=self.dtype).reshape(c1 * c2, 3)

        for i in range(0, len(arr)):
            self.g(arr[i], arr[i], self.iparams_arr[index], self.sparams_arr[index],
                   self.qs_xs[index], use_cuda=False)

        return arr.reshape((c1, c2, 3))
