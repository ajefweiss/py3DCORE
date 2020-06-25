# -*- coding: utf-8 -*-

import numpy as np

from py3dcore.model import Toroidal3DCOREModel
from py3dcore.params import Base3DCOREParameters
from py3dcore.models.torus_gh.coordinates import g, f
from py3dcore.models.torus_gh.magfield import h
from py3dcore.models.torus_gh.propagation import p


class TorusGH3DCOREModel(Toroidal3DCOREModel):
    """Implements the TorusGH3DCOREModel model.

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

    def __init__(self, launch, runs, use_gpu=False, **kwargs):
        """Initialize TorusGH3DCOREModel model.

        Parameters
        ----------
        launch: datetime.datetime
            Initial datetime.
        runs : int
            Number of parallel model runs.
        use_gpu : bool, optional
            GPU flag, by default False.

        Other Parameters
        ----------------
        cuda_device: int
            CUDA device, by default 0.
        dtype: type
            Data type, by default np.float32.
        """
        funcs = {
            "g": g,
            "f": f,
            "h": h,
            "p": p
        }

        dtype = kwargs.pop("dtype", np.float32)

        parameters = kwargs.pop("parameters", torus_gh_parameters)

        if isinstance(parameters, dict):
            parameters = Base3DCOREParameters(parameters, dtype=dtype)

        super(TorusGH3DCOREModel, self).__init__(
            launch, funcs, parameters, sparams_count=5, runs=runs, use_gpu=use_gpu, **kwargs
        )

    @classmethod
    def default_parameters(cls):
        return dict(torus_gh_parameters)


torus_gh_parameters = {
    "cme_launch_offset": {
        "index": 0,
        "name": "launch time",
        "distribution": "fixed",
        "fixed_value": 0,
        "maximum": 3600,
        "minimum": -3600,
        "boundary": "continuous",
        "label": "T_0"
    },
    "cme_longitude": {
        "index": 1,
        "name": "longitude",
        "distribution": "uniform",
        "maximum": 180,
        "minimum": -180,
        "boundary": "periodic",
        "label": "L_1"
    },
    "cme_latitude": {
        "index": 2,
        "name": "latitude",
        "distribution": "uniform",
        "maximum": 90,
        "minimum": -90,
        "boundary": "periodic",
        "label": "L_2"
    },
    "cme_inclination": {
        "index": 3,
        "name": "inclination",
        "distribution": "uniform",
        "maximum": 360,
        "minimum": 0,
        "boundary": "periodic",
        "label": "I"
    },
    "cme_diameter_1au": {
        "index": 4,
        "name": "diameter",
        "distribution": "uniform",
        "maximum": .35,
        "minimum": .05,
        "boundary": "continuous",
        "label": "D_{1AU}"
    },
    "cme_aspect_ratio": {
        "index": 5,
        "name": "aspect ratio",
        "distribution": "uniform",
        "maximum": 8.5,
        "minimum": .5,
        "boundary": "continuous",
        "label": "\delta"
    },
    "cme_launch_radius": {
        "index": 6,
        "name": "initial radius",
        "distribution": "fixed",
        "fixed_value": 20,
        "maximum": 25,
        "minimum": 15,
        "boundary": "continuous",
        "label": "R_0"
    },
    "cme_launch_velocity": {
        "index": 7,
        "name": "velocity",
        "distribution": "uniform",
        "maximum": 1000,
        "minimum": 400,
        "boundary": "continuous",
        "label": "V_0"
    },
    "magnetic_field_twist": {
        "index": 8,
        "name": "magnetic twist",
        "distribution": "uniform",
        "maximum": 25,
        "minimum": -25,
        "boundary": "continuous",
        "label": "\\tau"
    },
    "magnetic_field_radius": {
        "index": 9,
        "name": "magnetic coeff",
        "distribution": "uniform",
        "maximum": 2,
        "minimum": 0.1,
        "boundary": "continuous",
        "label": "B_S"
    },
    "magnetic_field_strength_1au": {
        "index": 10,
        "name": "axial strength",
        "distribution": "uniform",
        "maximum": 50,
        "minimum": 5,
        "boundary": "continuous",
        "label": "B_{1AU}"
    },
    "background_drag": {
        "index": 11,
        "name": "background drag",
        "distribution": "uniform",
        "maximum": 2,
        "minimum": 0.2,
        "boundary": "continuous",
        "label": "\gamma"
    },
    "background_velocity": {
        "index": 12,
        "name": "background velocity",
        "distribution": "uniform",
        "maximum": 450,
        "minimum": 250,
        "boundary": "continuous",
        "label": "V_{SW}"
    },
    "noise": {
        "index": 13,
        "name": "gaussian noise",
        "distribution": "uniform",
        "maximum": 5,
        "minimum": 0,
        "boundary": "continuous",
        "label": "\sigma"
    }
}
