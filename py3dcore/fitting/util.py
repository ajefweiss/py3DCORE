# -*- coding: utf-8 -*-

"""util.py

Utility function for fitting results.
"""

import heliosat
import numpy as np
import py3dcore


def generate_ensemble(path, ts, satellite, frame, return_spread=True, remove_noise=False):
    obj = py3dcore.fitting.BaseFitter()
    obj.load(path)

    model_obj = obj.model(obj.t_launch, runs=len(obj.particles), use_gpu=False)

    if remove_noise:
        obj.particles[:, -1] = 0

    model_obj.update_iparams(obj.particles, seed=142)

    sat = heliosat.satellites.select_satellite(satellite)()

    ensemble = np.squeeze(np.array(model_obj.sim_fields(ts, sat.trajectory(ts, frame="HCI"))))

    if frame != "HCI":
        ensemble = heliosat.coordinates.transform_pos(ts, ensemble, "HCI", frame)

    ensemble[np.where(ensemble == 0)] = np.nan

    # generate quantiles
    b_m = np.nanmean(ensemble, axis=1)

    b_s2p = np.nanquantile(ensemble, 0.5 + 0.95 / 2, axis=1)
    b_s2n = np.nanquantile(ensemble, 0.5 - 0.95 / 2, axis=1)

    b_t = np.sqrt(np.sum(ensemble**2, axis=2))
    b_tm = np.nanmean(b_t, axis=1)

    b_ts2p = np.nanquantile(b_t, 0.5 + 0.95 / 2, axis=1)
    b_ts2n = np.nanquantile(b_t, 0.5 - 0.95 / 2, axis=1)

    if return_spread:
        return b_m, b_tm, (b_s2p, b_s2n), (b_ts2p, b_ts2n)
    else:
        return b_m, b_tm
