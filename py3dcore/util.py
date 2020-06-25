# -*- coding: utf-8 -*-

"""util.py

3DCORE utility functions.
"""

import logging
import py3dcore
import sys


def configure_logging(debug=False, logfile=None, verbose=False):
    """Configures built in python logger.

    Parameters
    ----------
    debug : bool, optional
        Enable debug logging.
    logfile : None, optional
        Logfile path.
    verbose : bool, optional
        Enable verbose logging.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    stream = logging.StreamHandler(sys.stdout)
    if debug and verbose:
        stream.setLevel(logging.DEBUG)
    elif verbose:
        stream.setLevel(logging.INFO)
    else:
        stream.setLevel(logging.WARNING)

    stream.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(message)s"))
    root.addHandler(stream)

    if logfile:
        file = logging.FileHandler(logfile, "a")
        if debug:
            file.setLevel(logging.DEBUG)
        else:
            file.setLevel(logging.INFO)

        file.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root.addHandler(file)

    # disable annoying loggers
    logging.getLogger("numba.byteflow").setLevel("WARNING")
    logging.getLogger("numba.cuda.cudadrv.driver").setLevel("WARNING")


def select_model(model):
    if model.upper() == "THIN_TORUS_GH":
        return py3dcore.models.thin_torus_gh.ThinTorusGH3DCOREModel
    elif model.upper() == "THIN_TORUS_NC":
        return py3dcore.models.thin_torus_nc.ThinTorusNC3DCOREModel
    elif model.upper() == "TORUS_GH":
        return py3dcore.models.torus_gh.TorusGH3DCOREModel
    else:
        raise NotImplementedError("unkown model \"%s\"", model.upper())
