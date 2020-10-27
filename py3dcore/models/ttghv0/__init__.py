# -*- coding: utf-8 -*-

import json
import numpy as np
import os
import py3dcore

from py3dcore.model import Toroidal3DCOREModel
from py3dcore.params import Base3DCOREParameters
from py3dcore.models.ttghv0.coordinates import g, f
from py3dcore.models.ttghv0.magfield import h
from py3dcore.models.ttghv0.propagation import p


class TTGHv0(Toroidal3DCOREModel):
    """Implements the torus gold-hoyle (v0) 3DCORE model.

    Extended Summary
    ================
        For this specific model there are a total of 14 initial parameters which are as follows:
        0: t_i          time offset
        1: lon          longitude
        2: lat          latitude
        3: inc          inclination
        4: diameter     cross section diameter at 1 AU
        5: delta        cross section aspect ratio
        6: radius       initial cme radius
        7: velocity     initial cme velocity
        8: turns        magnetic field turns over entire flux rope
        9: m_coeff      magnetic field coefficient needed for some magnetic models
        10: b           magnetic field strength at center at 1AU
        11: bg_d        solar wind background drag coefficient
        12: bg_v        solar wind background speed
        13: noise       instrument noise

        There are 5 state parameters which are as follows:
        0: t_t          current time
        1: v_t          current velocity
        2: rho_0        torus major radius
        3: rho_1        torus minor radius
        4: b            magnetic field strength at center
    """

    def __init__(self, launch, runs, **kwargs):
        """Initialize ThinTorusGH3DCOREModel model.

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

        super(TTGHv0, self).__init__(
            launch, funcs, parameters, sparams_count=5, runs=runs, **kwargs
        )

    @classmethod
    def default_parameters(cls):
        path = os.path.join(os.path.dirname(py3dcore.__file__), "models/ttghv0/parameters.json")

        with open(path) as fh:
            data = json.load(fh)

        return data
