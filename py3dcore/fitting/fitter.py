# -*- coding: utf-8 -*-

import logging
import numpy as np
import os
import pickle
import py3dcore

from py3dcore.util import select_model


class BaseFitter(object):
    """Base 3DCORE fitting class.
    """
    def __init__(self):
        self.name = None
        self.observers = []

    def add_observer(self, observer, t, t_s, t_e, dt=0, dd=1, dti=0):
        """Add magnetic field observation

        Parameters
        ----------
        observer : heliosat.Spacecraft
            Observer object.
        t : np.ndarray
            Observer datetimes.
        t_s : np.ndarray
            Start marker.
        t_e : np.ndarray
            End marker.
        dt : float
            Market offset (in hours), by default 0,
        dd : int
            Marker offset reduction (in hours) per iteration, by default 1.
        ddi : int
            Marker tightening offset, by default 0.
        """
        self.observers.append([observer, t, t_s, t_e, dt, dd, dti])

    def init(self, t_launch, model, **kwargs):
        """Fitter initialization.

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
            model.default_parameters()
        )
        self.seed = kwargs.get("seed", 42)

        # fix parameters
        if set_params:
            pdict = self.parameters.params_dict

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

            self.parameters._update_arr()

    def load(self, path):
        logger = logging.getLogger(__name__)

        if os.path.isdir(path):
            files = os.listdir(path)

            if len(files) > 0:
                files.sort()
                path = os.path.join(path, files[-1])
            else:
                raise FileNotFoundError("could not find %s", path)

            logger.info("loading fitting file \"%s\"", path)
        elif os.path.exists(path):
            logger.info("loading fitting file \"%s\"", path)
        else:
            raise FileNotFoundError("could not find %s", path)

        with open(path, "rb") as fh:
            data = pickle.load(fh)

        for attr in data:
            setattr(self, attr, data[attr])

        self.model = select_model(self.model)
        self.name = os.path.basename(path)

    def run(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, path):
        logger = logging.getLogger(__name__)

        if not os.path.exists(path):
            os.makedirs(path)

        if os.path.isdir(path):
            path = os.path.join(path, "{0:02d}".format(self.iter_i))

        data = {attr: getattr(self, attr) for attr in self.__dict__ if attr[0] != "_"}

        data["model"] = data["model"].__name__

        logger.info("saving fitting file \"%s\"", path)

        with open(path, "wb") as fh:
            pickle.dump(data, fh)

    def param_fix(self, param, value):
        """Set param to fixed value.

        Parameters
        ----------
        param : str
            Parameter name.
        value : float
            Parameter value.
        """
        self.parameters.params_dict[param]["distribution"] = "fixed"
        self.parameters.params_dict[param]["fixed_value"] = value

        self.parameters._update_arr()

    def param_minmax(self, param, minv, maxv, **kwargs):
        """Set param min/max values. If None, the value stays unchanged.

        Parameters
        ----------
        param : str
            Parameter name.
        value : float
            Parameter value.
        """
        if minv is not None:
            self.parameters.params_dict[param]["minimum"] = minv

        if maxv is not None:
            self.parameters.params_dict[param]["maximum"] = maxv

        for k, v in kwargs.items():
            self.parameters.params_dict[param][k] = v

        self.parameters._update_arr()

