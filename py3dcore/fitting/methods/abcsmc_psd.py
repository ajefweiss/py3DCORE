# -*- coding: utf-8 -*-

"""abcsmc.py

Implementations of a ABC-SMC algorithm for 3DCORE.
"""

import logging
import multiprocessing
import numba
import numpy as np
import os
import pickle
import py3dcore
import time

from py3dcore._extmath import cholesky
from py3dcore._extnumba import set_random_seed
from py3dcore.params import _numba_calculate_weights_reduce


class ABCSMC_PSD(object):
    t_data = []
    b_data = []
    o_data = []
    b_fft = []
    mask = []

    name = None

    def __init__(self):
        pass

    def add_observation(self, t_data, b_data, o_data, fft_data):
        """Add magnetic field observation

        Parameters
        ----------
        t_data : np.ndarray
            Time evaluation array.
        b_data : np.ndarray
            Magnetic field array.
        o_data : np.ndarray
            Observer position array.
        """
        self.t_data.extend(t_data)
        self.b_data.extend(b_data)
        self.o_data.extend(o_data)

        self.b_fft.append(fft_data)

        _mask = [1] * len(b_data)
        _mask[0] = 0
        _mask[-1] = 0

        self.mask.extend(_mask)

    def init(self, t_launch, model, **kwargs):
        """ABC SMC initialization.

        Parameters
        ----------
        t_launch : datetime.datetime
            Initial CME launch time.
        model : Base3DCOREModel
            3DCORE model class.

        Other Parameters
        ----------------
        seed : int
            Random seed, by default 42.
        set_params : dict
            Dictionary containing parameters to fix to given value.
        """
        logger = logging.getLogger(__name__)

        set_params = kwargs.get("set_params", None)

        self.t_launch = t_launch
        self.model = model
        self.parameters = py3dcore.params.Base3DCOREParameters(
            model.default_parameters())
        self.seed = kwargs.get("seed", 42)

        pdict = self.parameters.params_dict

        # fix parameters
        if set_params:
            for spkey, spval in set_params.items():
                for key in pdict:
                    if key == spkey:
                        logger.info("setting \"%s\"=%.3f", key, spval)
                        pdict[key]["distribution"] = "fixed"
                        pdict[key]["fixed_value"] = spval
                    elif spkey.isdigit() and pdict[key]["index"] == int(spkey):
                        logger.info("setting \"%s\"=%.3f", key, spval)
                        pdict[key]["distribution"] = "fixed"
                        pdict[key]["fixed_value"] = spval

        # disable gaussian noise
        pdict["noise"]["distribution"] = "fixed"
        pdict["noise"]["fixed_value"] = 0

        self.parameters._update_arr()

        # setup variables
        self.iter_i = 0

        self.acc_rej_hist = []
        self.eps_hist = []
        self.timer_hist = []

        self.particles = None
        self.weights = None
        self.kernels = None
        self.epses = None

        self.particles_prev = None
        self.weights_prev = None

    def load(self, path):
        logger = logging.getLogger(__name__)

        if os.path.isdir(path):
            files = os.listdir(path)

            if len(files) > 0:
                files.sort()
                path = os.path.join(path, files[-1])
            else:
                raise FileNotFoundError

            logger.info("loading particle file \"%s\"", path)
        elif os.path.exists(path):
            logger.info("loading particle file \"%s\"", path)
        else:
            raise FileNotFoundError

        with open(path, "rb") as fh:
            data = pickle.load(fh)

        self.t_launch = data["t_launch"]

        self.model = py3dcore.util.select_model(data["model"])

        self.parameters = data["parameters"]
        self.seed = data["seed"]

        self.iter_i = len(data["eps_hist"]) - 2

        self.acc_rej_hist = data["acc_rej_hist"]
        self.eps_hist = data["eps_hist"]
        self.timer_hist = data["timer_hist"]

        self.particles = data["particles"]
        self.weights = data["weights"]
        self.kernels = data["kernels"]
        self.epses = data["epses"]
        self.profiles = data["profiles"]

        self.particles_prev = None
        self.weights_prev = None

        self.t_data = data["t_data"]
        self.b_data = data["b_data"]
        self.o_data = data["o_data"]
        self.b_fft = data["b_fft"]
        self.mask = data["mask"]

        self.name = os.path.basename(path)

    def run(self, iter_end, particles, **kwargs):
        """Run ABC SMC algorithm.

        Parameters
        ----------
        iter_end : int
            Maximum number of iterations.

        particles : int
            Number of particles per iteration, by default 4096.

        Other Parameters
        ----------------
        eps_quantile: float
            Adaptive threshold stepping, by default 0.5.
        jobs : int
            Number of total jobs, by default 8.
        kernel_mode : str
            Transition kernel mode, by default "lcm".
        output : str
            Output folder.
        runs : int
            Number of model runs per worker, by default 2**16.
        sub_iter_max : int
            Maximum number of sub iterations, by default 50.
        workers : int
            Number of parallel workers, by default 8.
        toffset : float
            T marker offsets in hours, by default None.
        """
        eps_quantile = kwargs.get("eps_quantile", .5)
        jobs = kwargs.get("jobs", 8)
        kernel_mode = kwargs.get("kernel_mode", "lcm")
        output = kwargs.get("output", None)
        runs = kwargs.get("runs", 16)
        sub_iter_max = kwargs.get("sub_iter_max", 50)
        workers = kwargs.get("workers", 8)
        toff = kwargs.get("toffset", None)

        kill_flag = False

        pool = multiprocessing.Pool(processes=workers)

        logger = logging.getLogger(__name__)

        if len(self.eps_hist) == 0:
            eps_0 = rmse([np.zeros((1, 3))] * len(self.b_data), self.b_data)[0]
            self.eps_hist = [2 * eps_0, 2 * eps_0 * 0.98]

            logger.info("starting abc algorithm, eps_0 = %0.2fnT", self.eps_hist[-1])

        for iter_i in range(self.iter_i, iter_end):
            logger.info("starting iteration %i", iter_i)

            timer_iter = time.time()

            if iter_i > 0:
                # switch particles/particles_prev
                _tp = self.particles_prev
                self.particles_prev = self.particles

                _tw = self.weights_prev
                self.weights_prev = self.weights

                # decompose particle kernels
                if kernel_mode == "cm":
                    kernels_lower = cholesky(2 * self.kernels)
                elif kernel_mode == "lcm":
                    kernels_lower = cholesky(2 * self.kernels)
            else:
                kernels_lower = None

            sub_iter_i = 0
            boost = 0

            rseed = self.seed + 100000 * iter_i
            _results = pool.starmap(abcsmc_worker, [(iter_i, self.model, self.t_launch,
                                                     self.t_data, self.b_data, self.o_data,
                                                     self.b_fft,
                                                     self.mask, self.parameters, self.eps_hist[-1],
                                                     rseed + i, self.particles_prev, self.weights_prev,
                                                     kernels_lower, runs, boost, logger)
                                                    for i in range(jobs)])

            total_runs = jobs * int(2**runs)

             # perform additional runs if insufficient particles are collected
            while True:
                tlens = [len(jp[1]) for jp in _results]
                tlen = sum(tlens)

                self.particles = np.zeros((tlen, len(self.parameters)), dtype=np.float32)
                self.epses = np.zeros((tlen, ), dtype=np.float32)
                self.profiles = np.zeros((tlen, len(self.b_data), 3), dtype=np.float32)

                acc_rej = np.array([0, 0, 0])

                for i in range(0, len(_results)):
                    self.particles[sum(tlens[:i]):sum(tlens[:i + 1])] = _results[i][0]
                    self.epses[sum(tlens[:i]):sum(
                        tlens[:i + 1])] = _results[i][1]
                    self.profiles[sum(tlens[:i]):sum(tlens[:i + 1])] = _results[i][2]

                    acc_rej += _results[i][3]

                logger.info("step %i:%i with (%i/%i) particles", iter_i, sub_iter_i, tlen,
                            particles)

                if tlen > particles:
                    break

                # adaptive run boosting
                dr = 19 - runs - boost

                if dr > 0:
                    exp = particles / ((tlen + 1) * (sub_iter_i + 1))

                    if exp > 8 and dr > 3:
                        boost += 3
                    elif exp > 4 and dr > 2:
                        boost += 2
                    elif exp > 2:
                        boost += 1

                rseed = self.seed + 100000 * iter_i + 1000 * (sub_iter_i + 1)
                _results_ext = pool.starmap(abcsmc_worker, [(iter_i, self.model, self.t_launch,
                                                      self.t_data, self.b_data, self.o_data,
                                                      self.b_fft,
                                                      self.mask, self.parameters,
                                                      self.eps_hist[-1], rseed + i,
                                                      self.particles_prev, self.weights_prev, kernels_lower,
                                                      runs, boost, logger)
                                                     for i in range(jobs)])
                _results.extend(_results_ext)

                sub_iter_i += 1
                total_runs += jobs * int(2**(runs+boost))

                # kill conditions
                if sub_iter_i == 5 + boost:
                    if tlen * np.floor(sub_iter_max / 5) < particles:
                        logger.warning("expected to exceed maximum number of sub iterations (%i)",
                                       sub_iter_max)
                        logger.warning("aborting")
                        kill_flag = True
                        break

            if kill_flag:
                break

            logger.info("%.2f%% acc, %.2f%% hit, %.2f%% rej", 100 * acc_rej[0] / total_runs,
                        100 * acc_rej[1] / total_runs, 100 * acc_rej[2] / total_runs)

            if tlen > particles:
                self.particles = self.particles[:particles]
                self.epses = self.epses[:particles]

            if iter_i > 0:
                self.weights = np.ones((particles,), dtype=np.float32)
                self.parameters.weight(self.particles, self.particles_prev, self.weights,
                                       self.weights_prev, self.kernels)
                self.weights[np.where(self.weights == np.nan)] = 0
            else:
                self.weights = np.ones((particles,), dtype=np.float32) / particles

            # set new eps
            self.eps_hist.append(np.quantile(self.epses, eps_quantile))

            logger.info("setting new eps: %.3f => %.3f", self.eps_hist[-2], self.eps_hist[-1])

            # compute transition kernels
            if kernel_mode == "cm":
                self.kernels = np.cov(self.particles, rowvar=False, aweights=self.weights)

                # due to aweights sometimes very small numbers are generated
                self.kernels[np.where(self.kernels < 1e-14)] = 0
            elif kernel_mode == "lcm":
                kernels_cm = np.cov(self.particles, rowvar=False, aweights=self.weights)

                # due to aweights sometimes very small numbers are generated
                kernels_cm[np.where(kernels_cm < 1e-14)] = 0

                kernels_cm_inv = np.linalg.pinv(kernels_cm)

                logger.info("generating local kernels")
                self.kernels = np.array(pool.starmap(generate_kernels_lcm, [(i, self.particles, kernels_cm_inv) for i in range(particles)]))
            self.acc_rej_hist.append(acc_rej)

            self.timer_hist.append(time.time() - timer_iter)

            logger.info("step %i done, %i particles, %.2fM runs in %.2f seconds, (total: %s)",
                        iter_i, particles, total_runs / 1e6, time.time() - timer_iter,
                        time.strftime("%Hh %Mm %Ss", time.gmtime(np.sum(self.timer_hist))))

            self.iter_i = iter_i

            if output:
                self.save(output)

        pool.close()

    def save(self, path):
        logger = logging.getLogger(__name__)

        if not os.path.exists(path):
            os.makedirs(path)

        if os.path.isdir(path):
            path = os.path.join(path, "{0:02d}".format(self.iter_i))

        data = {
            "particles": self.particles,
            "weights": self.weights,
            "kernels": self.kernels,
            "epses": self.epses,
            "profiles": self.profiles,
            "acc_rej_hist": self.acc_rej_hist,
            "eps_hist": self.eps_hist,
            "timer_hist": self.timer_hist,
            "t_launch": self.t_launch,
            "model": self.model.__name__,
            "parameters": self.parameters,
            "seed": self.seed,
            "t_data": self.t_data,
            "b_data": self.b_data,
            "o_data": self.o_data,
            "b_fft": self.b_fft,
            "mask": self.mask
        }

        logger.info("saving file \"%s\"", path)

        with open(path, "wb") as fh:
            pickle.dump(data, fh)


def abcsmc_worker(iter_i, model, t_launch, t_data, b_data, o_data, b_fft, mask,
                  parameters, eps, seed, particles, weights, kernels_lower, runs, boost, logger):
    model_obj = model(t_launch, int(2**(runs + boost)),
                      parameters=parameters, use_gpu=False)
    #logger.info("starting worker")
    if iter_i == 0:
        model_obj.generate_iparams(seed=seed)
    else:
        set_random_seed(seed)
        model_obj.perturb_iparams(particles, weights, kernels_lower)

    profiles = np.array(model_obj.sim_fields(t_data, o_data))
    #logger.info("generated profiles")

    # generate PSD fluctuations for each component and observation
    obsc = len(b_fft)

    for i in range(3):
        dloffset = 0
        for j in range(obsc):
            datalen = len(b_fft[j])

            wni = np.fft.fft(np.random.normal(0, 1, size=(int(2**(runs + boost)), datalen)))
            noise = np.real(np.fft.ifft(wni * b_fft[j]) / np.sqrt(datalen)).T

            nullf = (profiles[1 + dloffset:dloffset + (datalen + 2) - 1, :, i] != 0)
            profiles[1 + dloffset:dloffset + (datalen + 2) - 1, :, i][nullf] += noise[nullf]

            dloffset += datalen + 2

    if obsc > 1:
        # compute max error for each observation
        errors = []

        for i in range(obsc):
            obslc = slice(i * (datalen + 2), (i + 1) * (datalen + 2))
            errors.append(rmse(profiles[obslc], b_data[obslc], mask=mask[obslc]))

        error = np.max(errors, axis=0)
    else:
        error = rmse(profiles, b_data, mask=mask)

    accept_mask = error < eps
    rej_mask = np.sum(error == np.inf)

    rej_error = np.sum((error != np.inf) & (error >= eps))
    acc_count = np.sum(accept_mask)

    return model_obj.iparams_arr[accept_mask], error[accept_mask], \
        np.swapaxes(profiles, 0, 1)[accept_mask], \
        np.array([acc_count, rej_error, rej_mask])


def generate_kernels_lcm(i, particles, kernel_cm_inv):
    distances = np.array([_numba_calculate_weights_reduce(particles[i], particles[j], kernel_cm_inv) for j in range(len(particles))])

    cutoff = np.median(distances)

    return np.cov(particles[np.where(distances < cutoff)], rowvar=False)


def rmse(values, reference, mask=None, use_gpu=False):
    """Compute RMSE between numerous generated 3DCORE profiles and a reference profile. If a
    mask is given, profiles that are masked if their values are not non-zero where the filter is
    set to non-zero.

    Parameters
    ----------
    values : Union[list[np.ndarray], list[numba.cuda.cudadrv.devicearray.DeviceNDArray]]
        List of magnetic field outputs.
    reference : Union[np.ndarray, numba.cuda.cudadrv.devicearray.DeviceNDArray]
        Reference magnetic field measurements.
    mask : np.ndarray, optional
        Mask array, by default None
    use_gpu : bool, optional
        GPU flag, by default False
    """
    if use_gpu:
        raise NotImplementedError
    else:
        rmse = np.zeros(len(values[0]))

        if mask is not None:
            for i in range(len(reference)):
                _error_rmse(values[i], reference[i], mask[i], rmse)

            rmse = np.sqrt(rmse / len(values))

            mask_arr = np.copy(rmse)

            for i in range(len(reference)):
                _error_mask(values[i], mask[i], mask_arr)

            return mask_arr
        else:
            for i in range(len(reference)):
                _error_rmse(values[i], reference[i], 1, rmse)

            rmse = np.sqrt(rmse / len(values))

            return rmse


@numba.njit
def _error_mask(values_t, mask, rmse):
    for i in numba.prange(len(values_t)):
        _v = np.sum(values_t[i]**2)
        if (_v > 0 and mask == 0) or (_v == 0 and mask != 0):
            rmse[i] = np.inf


@numba.njit
def _error_rmse(values_t, ref_t, mask, rmse):
    for i in numba.prange(len(values_t)):
        if mask == 1:
            rmse[i] += np.sum((values_t[i] - ref_t)**2)
