# -*- coding: utf-8 -*-

import datetime
import logging
import heliosat
import multiprocessing
import numpy as np
import os
import pickle
import time

from .base import BaseFitter, FittingData
from ..model import SimulationBlackBox
from ..util import set_random_seed
from heliosat.util import sanitize_dt
from typing import Any, Optional, Sequence, Tuple, Union

import faulthandler

faulthandler.enable()


def starmap(func, args):
    return [func(*_) for _ in args]


class ABC_SMC(BaseFitter):
    iter_i: int

    hist_eps: list
    hist_eps_dim: int
    hist_time: list

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(ABC_SMC, self).__init__(*args, **kwargs)

    def initialize(self, *args: Any, **kwargs: Any) -> None:
        super(ABC_SMC, self).initialize(*args, **kwargs)

        self.iter_i = 0
        self.hist_eps = []
        self.hist_time = []
    
    def run(self, iter_max: int, ensemble_size: int, reference_frame: str, **kwargs: Any) -> None:
        logger = logging.getLogger(__name__)
 
        # read kwargs
        balanced_iterations = kwargs.pop("balanced_iterations", 3)
        data_kwargs = kwargs.pop("data_kwargs", {})
        eps_quantile = kwargs.pop("eps_quantile", 0.25)
        kernel_mode = kwargs.pop("kernel_mode", "cm")
        output = kwargs.get("output", None)
        random_seed = kwargs.pop("random_seed", 42)
        summary_type = kwargs.pop("summary_statistic", "norm_rmse")
        time_offsets = kwargs.pop("time_offsets", [0])

        jobs = kwargs.pop("jobs", 8)
        workers = kwargs.pop("workers", multiprocessing.cpu_count())

        mpool = multiprocessing.Pool(processes=workers)

        data_obj = FittingData(self.observers, reference_frame)
        data_obj.generate_noise(kwargs.get("noise_model", "psd"), kwargs.get("sampling_freq", 300), **data_kwargs)

        kill_flag = False
        pcount = 0
        timer_iter = None

        try:
            for iter_i in range(self.iter_i, iter_max):
                logger.info("running iteration %i", iter_i)

                timer_iter = time.time()

                if iter_i >= len(time_offsets):
                    _time_offset = time_offsets[-1]
                else:
                    _time_offset = time_offsets[iter_i]

                data_obj.generate_data(_time_offset, **data_kwargs)

                if len(self.hist_eps) == 0:
                    eps_init = data_obj.sumstat([np.zeros((1, 3))] * len(data_obj.data_b), use_mask=False)[0]
                    self.hist_eps = [eps_init, eps_init * 0.98]
                    self.hist_eps_dim = len(eps_init)

                    logger.info("initial eps_init = %s", self.hist_eps[-1])
                    
                    model_obj_kwargs = dict(self.model_kwargs)  # type: ignore
                    model_obj_kwargs["ensemble_size"] = ensemble_size
                    model_obj = self.model(self.dt_0, **model_obj_kwargs)

                sub_iter_i = 0

                _random_seed = random_seed + 100000 * iter_i

                worker_args = (iter_i, self.dt_0, self.model, self.model_kwargs, model_obj.iparams_arr, model_obj.iparams_weight, model_obj.iparams_kernel_decomp,
                            data_obj, summary_type, self.hist_eps[-1], kernel_mode)

                logger.info("starting simulations")
                _results = mpool.starmap(abc_smc_worker, [(*worker_args, _random_seed + i) for i in range(jobs)])
                total_runs = jobs * int(self.model_kwargs["ensemble_size"])  # type: ignore

                # repeat until enough samples are collected
                while True:
                    pcounts = [len(r[1]) for r in _results]
                    _pcount = sum(pcounts)
                    dt_pcount = _pcount - pcount
                    pcount = _pcount

                    particles_temp = np.zeros((pcount, model_obj.iparams_arr.shape[1]), model_obj.dtype)
                    epses_temp = np.zeros((pcount, self.hist_eps_dim), model_obj.dtype)

                    for i in range(0, len(_results)):
                        particles_temp[sum(pcounts[:i]):sum(pcounts[:i + 1])] = _results[i][0]
                        epses_temp[sum(pcounts[:i]):sum(pcounts[:i + 1])] = _results[i][1]

                    logger.info("step %i:%i with (%i/%i) particles", iter_i, sub_iter_i, pcount, ensemble_size)

                    if pcount > ensemble_size:
                        break

                    _random_seed = random_seed + 100000 * iter_i + 1000 * (sub_iter_i + 1)

                    _results_ext = mpool.starmap(abc_smc_worker, [(*worker_args, _random_seed + i) for i in range(jobs)])
                    _results.extend(_results_ext)

                    sub_iter_i += 1
                    total_runs += jobs * int(self.model_kwargs["ensemble_size"])  # type: ignore

                    if pcount == 0:
                        logger.warning("no hits, aborting")
                        kill_flag = True
                        break

                if kill_flag:
                    break

                if pcount > ensemble_size:
                    particles_temp = particles_temp[:ensemble_size]

                if iter_i == 0:
                    model_obj.update_iparams(particles_temp, update_weights_kernels=False, kernel_mode=kernel_mode)
                    model_obj.iparams_weight = np.ones((ensemble_size,), dtype=model_obj.dtype) / ensemble_size
                    model_obj.update_kernels(kernel_mode=kernel_mode)
                else:
                    model_obj.update_iparams(particles_temp, update_weights_kernels=True, kernel_mode=kernel_mode)

                if isinstance(eps_quantile, float):
                    new_eps = np.quantile(epses_temp, eps_quantile, axis=0)

                    if balanced_iterations > iter_i:
                        new_eps[:] = np.max(new_eps)
                    
                    self.hist_eps.append(new_eps)
                elif isinstance(eps_quantile, list) or isinstance(eps_quantile, np.ndarray):
                    eps_quantile_eff = eps_quantile ** (1 / self.hist_eps_dim)  # type: ignore
                    _k = len(eps_quantile_eff)  # type: ignore

                    new_eps = np.array([np.quantile(epses_temp, eps_quantile_eff[i], axis=0)[i] for i in range(_k)])

                    self.hist_eps.append(new_eps)  # type: ignore

                logger.info("setting new eps: %s => %s", self.hist_eps[-2], self.hist_eps[-1])

                self.hist_time.append(time.time() - timer_iter)

                logger.info("step %i done, %i particles, %.2fM runs in %.2f seconds, (total: %s)",
                            iter_i, ensemble_size, total_runs / 1e6, time.time() - timer_iter,
                            time.strftime("%Hh %Mm %Ss", time.gmtime(np.sum(self.hist_time))))

                self.iter_i = iter_i + 1

                if output:
                    output_file = os.path.join(output, "{0:02d}.pickle".format(self.iter_i - 1))

                    extra_args = {
                        "model_obj": model_obj,
                        "data_obj": data_obj,
                        "epses": epses_temp,
                    }

                    self.save(output_file, **extra_args)
        finally:
            pass
            #mpool.close()
            #mpool.join()


def abc_smc_worker(*args: Any) -> Tuple[np.ndarray, np.ndarray]:
    iter_i, dt_0, model_class, model_kwargs, old_iparams, old_weights, old_kernels, data_obj, summary_type, eps_value, kernel_mode, random_seed = args

    if iter_i == 0:
        model_obj = model_class(dt_0, **model_kwargs)
        model_obj.generator(random_seed=random_seed)
    else:
        set_random_seed(random_seed)
        model_obj = model_class(dt_0, **model_kwargs)
        model_obj.perturb_iparams(old_iparams, old_weights, old_kernels, kernel_mode=kernel_mode)

    # TODO: sort data_dt by time

    # sort
    sort_index = np.argsort([_.timestamp() for _ in data_obj.data_dt])

    # generate synthetic profiles
    profiles = np.array(model_obj.simulator(np.array(data_obj.data_dt)[sort_index], np.array(data_obj.data_o)[sort_index])[0], dtype=model_obj.dtype)

    # resort profiles
    sort_index_rev = np.argsort(sort_index)
    profiles = profiles[sort_index_rev]

    # TODO: revert profiles to proper order after data_dt resorting

    # generate synthetic noise
    profiles = data_obj.add_noise(profiles)

    error = data_obj.sumstat(profiles, stype=summary_type, use_mask=True)

    accept_mask = np.all(error < eps_value, axis=1)

    result = model_obj.iparams_arr[accept_mask]
    #print("WORKER DONE", result.shape, error[accept_mask].shape)
    return result, error[accept_mask]
