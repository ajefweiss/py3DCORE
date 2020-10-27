# -*- coding: utf-8 -*-

"""smc.py

Implementations of a SMC algorithm for 3DCORE.
"""

import logging
import multiprocessing
import numba
import numpy as np
import time

from py3dcore._extmath import cholesky
from py3dcore._extnumba import set_random_seed
from py3dcore.params import _numba_calculate_weights_reduce

from py3dcore.fitting import BaseFitter


class SMCFitter(BaseFitter):
    def init(self, *args, **kwargs):
        self.iter_i = 0

        self.timer_hist = []

        self.particles = None
        self.weights = None
        self.kernels = None
        self.epses = None

        self.particles_prev = None
        self.weights_prev = None

        super(SMCFitter, self).init(*args, **kwargs)

    def run(self, iter_max, n_particles=8192, **kwargs):
        """Run SMC algorithm.

        Parameters
        ----------
        iter_max : int
            Maximum number of iterations.

        n_particles : int
            Number of particles per iteration, by default 8192.

        Other Parameters
        ----------------
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
        """
        jobs = kwargs.get("jobs", 8)
        kernel_mode = kwargs.get("kernel_mode", "lcm")
        output = kwargs.get("output", None)
        runs = kwargs.get("runs", 16)
        sub_iter_max = kwargs.get("sub_iter_max", 50)
        workers = kwargs.get("workers", 8)

        kill_flag = False

        pool = multiprocessing.Pool(processes=workers)

        logger = logging.getLogger(__name__)

        logger.info("starting smc algorithm")

        for iter_i in range(self.iter_i, iter_max):
            logger.info("starting iteration %i", iter_i)

            timer_iter = time.time()

            if iter_i > 0:
                # switch particles/particles_prev
                self.particles_prev = self.particles
                self.weights_prev = self.weights

                # decompose particle kernels
                if kernel_mode == "cm":
                    kernels_lower = cholesky(2 * self.kernels)
                elif kernel_mode == "lcm":
                    kernels_lower = cholesky(2 * self.kernels)
                else:
                    raise NotImplementedError("kernel mode %s is not implemented", kernel_mode)
            else:
                kernels_lower = None

            sub_iter_i = 0
            boost = 0

            rseed = self.seed + 100000 * iter_i
            _results = pool.starmap(smc_worker, [(iter_i, self.model, self.t_launch,
                                                  self.t_data, self.b_data, self.o_data,
                                                  self.mask, self.parameters,
                                                  rseed + i, self.particles_prev, self.weights_prev,
                                                  kernels_lower, runs, boost, logger)
                                                 for i in range(jobs)])

            total_runs = jobs * int(2**runs)

            # perform additional runs if insufficient particles are collected
            while True:
                tlens = [len(jp[1]) for jp in _results]
                tlen = sum(tlens)

                logger.info("step %i:%i with (%i/%i) particles", iter_i, sub_iter_i, tlen,
                            n_particles)

                if tlen > n_particles:
                    break

                # adaptive boosting
                dr = 19 - runs - boost

                if dr > 0:
                    exp = n_particles / ((tlen + 1) * (sub_iter_i + 1))

                    if exp > 8 and dr > 3:
                        boost += 3
                    elif exp > 4 and dr > 2:
                        boost += 2
                    elif exp > 2:
                        boost += 1

                rseed = self.seed + 100000 * iter_i + 1000 * (sub_iter_i + 1)
                _results_ext = pool.starmap(smc_worker, [(iter_i, self.model, self.t_launch,
                                                          self.t_data, self.b_data, self.o_data,
                                                          self.mask, self.parameters,
                                                          rseed + i, self.particles_prev,
                                                          self.weights_prev,
                                                          kernels_lower, runs, boost, logger)
                                            for i in range(jobs)])
                _results.extend(_results_ext)

                sub_iter_i += 1
                total_runs += jobs * int(2**(runs+boost))

                # kill conditions
                if sub_iter_i == 5 + boost:
                    if tlen * np.floor(sub_iter_max / 5) < n_particles:
                        logger.warning("expected to exceed maximum number of sub iterations (%i)",
                                       sub_iter_max)
                        logger.warning("aborting")
                        kill_flag = True
                        break

            self.particles = np.zeros((tlen, 14), dtype=np.float32)
            self.likelihoods = np.zeros((tlen,), dtype=np.float32)
            self.profiles = np.zeros((tlen, len(self.b_data), 3), dtype=np.float32)

            acc_rej = np.array([0, 0]).astype(np.float32)

            for i in range(0, len(_results)):
                self.particles[sum(tlens[:i]):sum(tlens[:i + 1])] = _results[i][0]
                self.likelihoods[sum(tlens[:i]):sum(tlens[:i + 1])] = _results[i][1]
                self.profiles[sum(tlens[:i]):sum(tlens[:i + 1])] = _results[i][2]
                acc_rej += _results[i][3]

            if kill_flag:
                break

            logger.info("%.2f%% acc, %.2f%% rej", 100 * acc_rej[0] / total_runs,
                        100 * acc_rej[1] / total_runs)

            if tlen > n_particles:
                self.particles = self.particles[:n_particles]
                self.likelihoods = self.likelihoods[:n_particles]
                self.profiles = self.profiles[:n_particles]

            # anneal and re-convert log likelihoods
            self.likelihoods *= iter_i / 250
            self.likelihoods += -np.max(self.likelihoods)
            self.likelihoods = np.exp(self.likelihoods)

            if iter_i > 0:
                self.weights = np.ones((n_particles,), dtype=np.float32)
                self.parameters.weight(self.particles, self.particles_prev, self.weights,
                                       self.weights_prev, self.kernels, use_cuda=False)
                self.weights[np.where(self.weights == np.nan)] = 0
            else:
                self.weights = np.ones((n_particles,), dtype=np.float32) / n_particles

            self.weights *= self.likelihoods

            logger.info("ess: %i",
                        np.sum(self.weights)**2 / np.sum(self.weights**2))

            # compute transition kernels
            if kernel_mode == "cm":
                self.kernels = np.cov(self.particles, rowvar=False, aweights=self.weights)

                # due to aweights sometimes very small numbers are generated
                self.kernels[np.where(self.kernels < 1e-14)] = 0
            elif kernel_mode == "lcm":
                kernels_cm = np.cov(self.particles, rowvar=False, aweights=self.weights)

                # due to aweights sometimes very small numbers are generated
                kernels_cm[np.where(kernels_cm < 1e-14)] = 0

                k_cm_inv = np.linalg.pinv(kernels_cm)

                logger.info("generating local kernels")
                self.kernels = pool.starmap(generate_kernels_lcm, [(i, self.particles, k_cm_inv)
                                                                   for i in range(n_particles)])

                self.kernels = np.array(self.kernels)

            # logger.info("step %i done, %i particles, %.2fM runs in %.2f seconds, (total: %s)",
            #             iter_i, particles, total_runs / 1e6, time.time() - timer_iter,
            #             time.strftime("%Hh %Mm %Ss", time.gmtime(np.sum(self.timer_hist))))

            self.timer_hist.append(time.time() - timer_iter)

            self.iter_i = iter_i + 1

            if output:
                self.save(output)

        pool.close()


def generate_kernels_lcm(i, particles, kernel_cm_inv):
    distances = np.array([_numba_calculate_weights_reduce(particles[i], particles[j], kernel_cm_inv)
                          for j in range(len(particles))])

    cutoff = np.median(distances)

    return np.cov(particles[np.where(distances < cutoff)], rowvar=False)


def smc_worker(iter_i, model, t_launch, t_data, b_data, o_data, mask,  parameters,
               seed, particles, weights, kernels_lower, runs, boost, logger):
    model_obj = model(t_launch, int(2**(runs + boost)),
                      parameters=parameters, use_gpu=False)

    if iter_i == 0:
        model_obj.generate_iparams(seed=seed)
    else:
        set_random_seed(seed)
        model_obj.perturb_iparams(particles, weights, kernels_lower)

    profiles = np.array(model_obj.sim_fields(t_data, o_data))
    likelihoods = likelihood(profiles, b_data, model_obj.iparams_arr[:, -1], mask=mask)

    accept_mask = likelihoods > -np.inf
    rej_count = np.sum(likelihoods == -np.inf)
    acc_count = np.sum(accept_mask)

    return model_obj.iparams_arr[accept_mask], likelihoods[accept_mask], \
        np.swapaxes(profiles, 0, 1)[accept_mask], \
        np.array([acc_count, rej_count])


def likelihood(values, reference, noise, mask=None, use_gpu=False):
    """Compute likelihood between numerous generated 3DCORE profiles and a reference profile.

    Parameters
    ----------
    values : Union[list[np.ndarray], list[numba.cuda.cudadrv.devicearray.DeviceNDArray]]
        List of magnetic field outputs.
    reference : Union[np.ndarray, numba.cuda.cudadrv.devicearray.DeviceNDArray]
        Reference magnetic field measurements.
    noise : Union[np.ndarray, numba.cuda.cudadrv.devicearray.DeviceNDArray]
        Noise values.
    mask : np.ndarray, optional
        Mask array, by default None
    use_gpu : bool, optional
        GPU flag, by default False
    """
    if use_gpu:
        raise NotImplementedError
    else:
        L = np.zeros(len(values[0]))

        if mask is not None:
            for i in range(len(reference)):
                _likelihood_gaussian(values[i], reference[i], mask[i], L, noise)

            mask_arr = np.copy(L)

            for i in range(len(reference)):
                _error_mask_lval(values[i], mask[i], mask_arr)

            return mask_arr
        else:
            for i in range(len(reference)):
                _likelihood_gaussian(values[i], reference[i], 1, L, noise)

            return L


@numba.njit
def _error_mask_lval(values_t, mask, loglvals):
    for i in numba.prange(len(values_t)):
        _v = np.sum(values_t[i]**2)
        if (_v > 0 and mask == 0) or (_v == 0 and mask != 0):
            loglvals[i] = -np.inf


@numba.njit
def _likelihood_gaussian(values_t, ref_t, mask, loglvals, noise):
    for i in numba.prange(len(values_t)):
        if mask == 1:
            loglvals[i] += -0.5 * np.sum((values_t[i] - ref_t)**2) / noise[i]**2 \
                - np.sqrt(2 * np.pi) * np.log(noise[i])
