# -*- coding: utf-8 -*-

"""abcsmc.py

Implementations of a ABC-SMC algorithm for 3DCORE.
"""

import datetime
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
from py3dcore.fitting import BaseFitter
from py3dcore.fitting.methods.summary import sumstat
from py3dcore.params import _numba_calculate_weights_reduce
from scipy.signal import detrend, welch


class ABCSMC_v1(BaseFitter):
    """ABC-SMC 3DCORE fitting class.

    Includes PSD noise and adaptive time markers.
    """
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
        super(ABCSMC_v1, self).init(t_launch, model, **kwargs)

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
        eps_imul: float
            Initial eps multiplier.
        eps_quantile: float
            Adaptive threshold stepping, by default 0.5.
        frame : str
            Reference frame, by default "HCI".
        jobs : int
            Number of total jobs, by default 8.
        kernel_mode : str
            Transition kernel mode, by default "lcm".
        noise_mode : str
            Noise mode, by default "gaussian".
        output : str
            Output folder.
        runs : int
            Number of model runs per worker, by default 2**16.
        sampling_freq : int
            FFT sampling frequency in seconds, by default 300.
        sub_iter_max : int
            Maximum number of sub iterations, by default 50.
        summary : str
            Summary statistic, by default "rmse".
        workers : int
            Number of parallel workers, by default 8.
        """
        eps_imul = kwargs.get("eps_imul", 1)
        eps_quantile = kwargs.get("eps_quantile", .5)
        frame = kwargs.get("frame", "HCI")
        jobs = kwargs.get("jobs", 8)
        kernel_mode = kwargs.get("kernel_mode", "lcm")
        noise_mode = kwargs.get("noise_mode", "gaussian")
        output = kwargs.get("output", None)
        runs = kwargs.get("runs", 16)
        sampling_freq = kwargs.get("sampling_freq", 300)
        sub_iter_max = kwargs.get("sub_iter_max", 10)
        summary = kwargs.get("summary", "rmse")
        workers = kwargs.get("workers", 8)
        data_kwargs = kwargs.get("data_kwargs", {})

        kill_flag = False

        pool = multiprocessing.Pool(processes=workers)

        logger = logging.getLogger(__name__)

        if len(self.eps_hist) == 0:
            logger.info("starting abc algorithm")

        f_data_all = []

        # generate fft data
        if noise_mode == "psd":
            self.param_fix("noise", 0)

            logger.info("generating psd for noise")

            for i in range(len(self.observers)):
                observer, t, t_s, t_e, dt, dd, dti = self.observers[i]

                obs_inst = observer()

                _, _f_rawd = obs_inst.get_data([t[0], t[-1]], "mag", cache=True, frame=frame,
                                               sampling_freq=sampling_freq, **data_kwargs)

                # fix for NaN values
                nanc = np.sum(np.isnan(_f_rawd))

                if nanc > 0:
                    logger.warning("psd noise (obs %i): setting %i NaN values to zero", i, nanc)

                _f_rawd[np.isnan(_f_rawd)] = 0

                td = int(((t[-1] - t[0]).total_seconds() / 3600) - 1)

                logger.info("psd noise (obs %i): linear detrending using %i breakpoints", i, td)

                nperseg = np.min([len(_f_rawd), 256])

                dresX = detrend(_f_rawd[:, 0], type="linear", bp=td)
                dresY = detrend(_f_rawd[:, 1], type="linear", bp=td)
                dresZ = detrend(_f_rawd[:, 2], type="linear", bp=td)

                _,  wX = welch(dresX, fs=1 / sampling_freq, nperseg=nperseg)
                _,  wY = welch(dresY, fs=1 / sampling_freq, nperseg=nperseg)
                wF, wZ = welch(dresZ, fs=1 / sampling_freq, nperseg=nperseg)

                # average over all 3 components
                wS = (wX + wY + wZ) / 3

                fftF = np.fft.fftfreq(len(dresX), d=sampling_freq)

                # convert P(k) into more suitable form for fft
                fftS = np.zeros((len(fftF)))
                for i in range(len(fftF)):
                    k = np.abs(fftF[i])
                    arg = np.argmin(np.abs(k - wF))
                    fftS[i] = np.sqrt(wS[arg])

                # compute time indices
                kt = (len(fftS) - 1) / (t[-1].timestamp() - t[0].timestamp())
                tis = [int((_.timestamp() - t[0].timestamp()) * kt) for _ in t]

                w_data = (fftS, tis, sampling_freq)

                f_data_all.append(w_data)

        # main loop
        tlen = 0
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

            t_data_all = []
            b_data_all = []
            o_data_all = []
            m_data_all = []

            # generate t/b/o/m data
            dls = []
            for i in range(len(self.observers)):
                observer, t, t_s, t_e, dt, dd, dti = self.observers[i]

                obs_inst = observer()
                
                _, _b_data = obs_inst.get_data(t, "mag", cache=True, frame=frame, **data_kwargs)

                if iter_i > dti:
                    dtf = max([0, dt - dd * (iter_i - dti)])
                else:
                    dtf = dt

                logger.info("obs %i marker offset set to %0.1f hours", i, dtf)

                t_data = [t_s - datetime.timedelta(hours=dtf)]
                t_data.extend(t)
                t_data.append(t_e + datetime.timedelta(hours=dtf))

                o_data = obs_inst.trajectory(t_data, frame=frame)

                b_data = np.zeros((len(_b_data) + 2, 3))
                b_data[1:-1] = _b_data

                dls.append(len(_b_data))

                m_data = [1] * len(b_data)
                m_data[0] = 0
                m_data[-1] = 0

                t_data_all.extend(t_data)
                b_data_all.extend(b_data)
                o_data_all.extend(o_data)
                m_data_all.extend(m_data)

            if len(self.eps_hist) == 0:
                eps_0 = sumstat([np.zeros((1, 3))] * len(b_data_all), b_data_all, stype=summary, obsc=len(self.observers), dls=dls)[0]
                self.eps_hist = [eps_imul * eps_0, eps_imul * eps_0 * 0.98]

                if len(eps_0) > 1:
                    self.eps_dim = len(eps_0)
                else:
                    self.eps_dim = 1

                logger.info("initial eps_0 = %s", self.eps_hist[-1])
                logger.info("eps dim = %i", self.eps_dim)

            sub_iter_i = 0
            boost = 0

            rseed = self.seed + 100000 * iter_i

            abcsmc_args = [
                iter_i, self.model, self.t_launch,
                t_data_all, b_data_all, o_data_all, f_data_all, m_data_all,
                self.parameters, self.eps_hist[-1], self.particles_prev, self.weights_prev,
                kernels_lower, runs, boost, logger, noise_mode, summary
            ]

            _results = pool.starmap(abcsmc_worker, [(*abcsmc_args, rseed + i) for i in range(jobs)])

            total_runs = jobs * int(2**runs)

            # perform additional runs if insufficient particles are collected
            while True:
                tlens = [len(jp[1]) for jp in _results]
                _tlen = sum(tlens)
                dtlens = _tlen - tlen
                tlen = _tlen
                

                self.particles = np.zeros((tlen, len(self.parameters)), dtype=np.float32)
                self.epses = np.zeros((tlen, self.eps_dim), dtype=np.float32)
                self.profiles = np.zeros((tlen, len(b_data_all), 3), dtype=np.float32)

                acc_rej = np.array([0, 0, 0])

                for i in range(0, len(_results)):
                    self.particles[sum(tlens[:i]):sum(tlens[:i + 1])] = _results[i][0]
                    self.epses[sum(tlens[:i]):sum(tlens[:i + 1])] = _results[i][1]
                    self.profiles[sum(tlens[:i]):sum(tlens[:i + 1])] = _results[i][2]

                    acc_rej += _results[i][3]

                logger.info("step %i:%i with (%i/%i) particles", iter_i, sub_iter_i, tlen,
                            particles)

                if tlen > particles:
                    break

                # adaptive boosting
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

                abcsmc_args[14] = boost

                _results_ext = pool.starmap(abcsmc_worker, [(*abcsmc_args, rseed + i)
                                                            for i in range(jobs)])
                _results.extend(_results_ext)

                sub_iter_i += 1
                total_runs += jobs * int(2**(runs+boost))

                # kill conditions
                if sub_iter_i == 2 and iter_i > 0:
                    if dtlens * (sub_iter_max - 2) + tlen < particles:
                        print(dtlens, sub_iter_max - 2, tlen)
                        logger.warning("expected to exceed maximum number of sub iterations (%i)",
                                       sub_iter_max)
                        logger.warning("aborting")
                        kill_flag = True
                        break
                
                if tlen == 0:
                    logger.warning("no hits, aborting")
                    kill_flag = True
                    break

            if kill_flag:
                break

            logger.info("%.2f%% acc, %.2f%% hit, %.2f%% rej", 100 * acc_rej[0] / total_runs,
                        100 * acc_rej[1] / total_runs, 100 * acc_rej[2] / total_runs)

            if tlen > particles:
                self.particles = self.particles[:particles]
                self.epses = self.epses[:particles]

            # reshuffle
            if iter_i == 0:
                for k, v in self.parameters.params_dict.items():
                    if v.get("reshuffle", False):
                        logger.info("reshuffling %s parameters", k)

                        # generate iparams
                        model_obj = self.model(self.t_launch, particles, parameters=self.parameters, use_gpu=False)
                        model_obj.generate_iparams(seed=10000 * sub_iter_max  * sub_iter_i + self.seed)

                        self.particles[:, v["index"]] = model_obj.iparams_arr[:, v["index"]]



            if iter_i > 0:
                self.weights = np.ones((particles,), dtype=np.float32)
                self.parameters.weight(self.particles, self.particles_prev, self.weights,
                                       self.weights_prev, self.kernels)
                self.weights[np.where(self.weights == np.nan)] = 0
            else:
                self.weights = np.ones((particles,), dtype=np.float32) / particles

            # set new eps
            

            if isinstance(eps_quantile, float):
                self.eps_hist.append(np.quantile(self.epses, eps_quantile, axis=0))
            elif isinstance(eps_quantile, list) or isinstance(eps_quantile, np.ndarray):
                eps_quantile_eff = eps_quantile ** (1 / self.eps_dim)
                _k = len(eps_quantile_eff)

                self.eps_hist.append(
                    np.array([np.quantile(self.epses, eps_quantile_eff[i], axis=0)[i] for i in range(_k)])
                    )

            logger.info("setting new eps: %s => %s", self.eps_hist[-2], self.eps_hist[-1])

            # compute transition kernels
            if kernel_mode == "cm":
                self.kernels = np.cov(self.particles, rowvar=False, aweights=self.weights)

                # due to aweights sometimes very small numbers are generated
                self.kernels[np.where(self.kernels < 1e-14)] = 0
            elif kernel_mode == "lcm":
                kernels_cm = np.cov(self.particles, rowvar=False, aweights=self.weights)

                # due to aweights sometimes very small numbers are generated
                kernels_cm[np.where(np.abs(kernels_cm) < 1e-14)] = 0
                kernels_cm_inv = np.linalg.pinv(kernels_cm)

                logger.info("generating local kernels")
                self.kernels = np.array(pool.starmap(generate_kernels_lcm, [(i, self.particles, kernels_cm_inv) for i in range(particles)]))

            self.acc_rej_hist.append(acc_rej)

            self.timer_hist.append(time.time() - timer_iter)

            logger.info("step %i done, %i particles, %.2fM runs in %.2f seconds, (total: %s)",
                        iter_i, particles, total_runs / 1e6, time.time() - timer_iter,
                        time.strftime("%Hh %Mm %Ss", time.gmtime(np.sum(self.timer_hist))))

            if output:
                self.save(output)

            self.iter_i = iter_i + 1

        pool.close()


def abcsmc_worker(*args):
    (iter_i, model, t_launch, 
     t_data_all, b_data_all, o_data_all, f_data_all, m_data_all,
     parameters, eps_last, particles, weights,
     kernels_lower, runs, boost, logger, noise_mode, summary,
     seed) = args

    model_obj = model(t_launch, int(2**(runs + boost)), parameters=parameters, use_gpu=False)

    if iter_i == 0:
        model_obj.generate_iparams(seed=seed)
    else:
        set_random_seed(seed)
        model_obj.perturb_iparams(particles, weights, kernels_lower)

    profiles = np.array(model_obj.sim_fields(t_data_all, o_data_all))

    if noise_mode == "psd":
        # generate PSD fluctuations
        obsc = len(f_data_all)
        dls = []

        for c in range(3):
            _off = 0
            for o in range(obsc):
                (fftS, tis, sampling_freq) = f_data_all[o]

                dl = len(tis)

                Cfac =  np.sqrt(sampling_freq)

                wni = np.fft.fft(np.random.normal(0, 1, size=(int(2**(runs + boost)), len(fftS))))
                noise = np.real(np.fft.ifft(wni * fftS) / Cfac).T

                noise = noise[tis]

                nullf = (profiles[1 + _off:_off + (dl + 2) - 1, :, c] != 0)
                profiles[1 + _off:_off + (dl + 2) - 1, :, c][nullf] += noise[nullf]

                _off += dl + 2

                if c == 0:
                    dls.append(dl)

        error = sumstat(profiles, b_data_all, mask=m_data_all, stype=summary, obsc=obsc, dls=dls)
    else:
        error = sumstat(profiles, b_data_all, mask=m_data_all, stype=summary)

    if error.ndim == 1:
        accept_mask = error < eps_last

        rej_mask = np.sum(error == np.inf)
        rej_error = np.sum((error != np.inf) & (error >= eps_last))
        acc_count = np.sum(accept_mask)
    elif error.ndim == 2:
        accept_mask = np.all(error < eps_last, axis=1)
        rej_mask = 0#np.sum(error == np.inf)
        rej_error = 0#np.sum((error != np.inf) & (error >= eps_last))
        acc_count = np.sum(accept_mask)
    else:
        raise IndexError

    return model_obj.iparams_arr[accept_mask], error[accept_mask], \
        np.swapaxes(profiles, 0, 1)[accept_mask], \
        np.array([acc_count, rej_error, rej_mask])


def generate_kernels_lcm(i, particles, kernel_cm_inv):
    distances = np.array([_numba_calculate_weights_reduce(particles[i], particles[j], kernel_cm_inv) for j in range(len(particles))])

    cutoff = np.median(distances)

    return np.cov(particles[np.where(distances < cutoff)], rowvar=False)
