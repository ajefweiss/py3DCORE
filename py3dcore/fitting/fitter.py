# -*- coding: utf-8 -*-

import logging
import os
import pickle
import py3dcore

from py3dcore.util import select_model


class BaseFitter(object):
    """Base 3DCORE fitting class.
    """
    t_data = []
    b_data = []
    o_data = []
    mask = []

    name = None

    def __init__(self):
        pass

    def add_observation(self, t_data, b_data, o_data):
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

        _mask = [1] * len(b_data)
        _mask[0] = 0
        _mask[-1] = 0

        self.mask.extend(_mask)

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
            model.default_parameters())
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
