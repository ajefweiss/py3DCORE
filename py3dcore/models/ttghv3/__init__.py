# -*- coding: utf-8 -*-

import json
import numpy as np
import os
import py3dcore

from py3dcore.model import Toroidal3DCOREModel
from py3dcore.params import Base3DCOREParameters
from py3dcore.models.ttghv3.coordinates import g, f
from py3dcore.models.ttghv3.magfield import h
from py3dcore.models.ttghv3.propagation import p


class TTGHv3(Toroidal3DCOREModel):
    """Implements the thin torus gold-hoyle (v3) 3DCORE model.

    Extended Summary
    ================
        For this specific model there are a total of 14 initial parameters which are as follows:
        0: t_i          time offset
        1: lon          longitude
        2: lat          latitude
        3: inc          inclination

        4: pa           pitch angle
        5: w            cme width ratio
        6: f            flattening factor

        7: r0           initial cme radius
        8: v0           initial cme velocity
        9: tau          magnetic field turns over entire flux rope
        10: b            magnetic field strength at center at 1AU
        11: bg_d        solar wind background drag coefficient
        12: bg_v        solar wind background speed
        13: noise       instrument noise

        There are 6 state parameters which are as follows:
        0: t_t          current time
        1: v_t          current velocity
        2: rho_0        torus major radius
        3: rho_1        torus minor radius
        4: delta        aspect-ratio
        5: b            magnetic field strength at center
    """

    def __init__(self, launch, runs, **kwargs):
        """Initialize the thin torus gold-hoyle (v3) 3DCORE model.

        Parameters
        ----------
        launch: datetime.datetime
            Initial datetime.
        runs : int
            Number of parallel model runs.

        Other Parameters
        ----------------
        cuda_device: int
            CUDA device, by default 0.
        dtype: type
            Data type, by default np.float32.
        use_cuda : bool
            CUDA flag, by default False.
        """
        funcs = {
            "g": g,
            "f": f,
            "h": h,
            "p": p
        }

        dtype = kwargs.pop("dtype", np.float32)

        parameters = kwargs.pop("parameters", self.default_parameters())

        if isinstance(parameters, dict):
            parameters = Base3DCOREParameters(parameters, dtype=dtype)

        super(TTGHv3, self).__init__(
            launch, funcs, parameters, sparams_count=6, runs=runs, **kwargs
        )

    @classmethod
    def default_parameters(cls):
        path = os.path.join(os.path.dirname(py3dcore.__file__), "models/ttghv3/parameters.json")

        with open(path) as fh:
            data = json.load(fh)

        return data
