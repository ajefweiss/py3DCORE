# -*- coding: utf-8 -*-

"""mode.py

Estimate mode from sample distribution
"""

import numpy as np


def estimate_mode(iparams_arr, weights, bin_density=250):
    samples, params = iparams_arr.shape

    mode = np.empty((params,))

    for i in range(params):
        if np.std(iparams_arr[:, i]) < 1e-8:
            mode[i] = iparams_arr[0, i]
            continue

        hist_counts, hist_edges = np.histogram(iparams_arr[:, i], weights=weights, bins=samples // bin_density)

        argmax = np.argmax(hist_counts)

        if argmax == 0:
            mode[i] = hist_edges[0] // 2
        elif argmax == len(hist_counts) - 1:
            mode[i] = hist_edges[-1]
        else:
            mode[i] = (hist_edges[argmax] + hist_edges[argmax - 1]) / 2
    

    return mode
