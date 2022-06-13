# -*- coding: utf-8 -*-

import datetime
import logging
import heliosat
import numpy as np
import os
import pickle

from ..model import SimulationBlackBox
from .sumstat import sumstat
from ..util import mag_fft
from heliosat.util import sanitize_dt
from heliosat.transform import transform_reference_frame
from typing import Any, List, Optional, Sequence, Type, Union


def generate_ensemble(path: str, dt: Sequence[datetime.datetime], reference_frame: str = "HCI", reference_frame_to: str = "HCI", perc: float = 0.95, max_index=None) -> np.ndarray:
    observers = BaseFitter(path).observers
    ensemble_data = []
    

    for (observer, _, _, _, _) in observers:
        ftobj = BaseFitter(path)
        observer_obj = getattr(heliosat, observer)()
        ensemble = np.squeeze(np.array(ftobj.model_obj.simulator(dt, observer_obj.trajectory(dt, reference_frame=reference_frame))[0]))

        if max_index is None:
            max_index =  ensemble.shape[1]

        ensemble = ensemble[:, :max_index, :]

        # transform frame
        if reference_frame != reference_frame_to:
            for k in range(0, ensemble.shape[1]):
                ensemble[:, k, :] = transform_reference_frame(dt, ensemble[:, k, :], reference_frame, reference_frame_to)

        ensemble[np.where(ensemble == 0)] = np.nan

        # generate quantiles
        b_m = np.nanmean(ensemble, axis=1)

        b_s2p = np.nanquantile(ensemble, 0.5 + perc / 2, axis=1)
        b_s2n = np.nanquantile(ensemble, 0.5 - perc / 2, axis=1)

        b_t = np.sqrt(np.sum(ensemble**2, axis=2))
        b_tm = np.nanmean(b_t, axis=1)

        b_ts2p = np.nanquantile(b_t, 0.5 + perc / 2, axis=1)
        b_ts2n = np.nanquantile(b_t, 0.5 - perc / 2, axis=1)

        ensemble_data.append([None, None, (b_s2p, b_s2n), (b_ts2p, b_ts2n)])

    return ensemble_data


class BaseFitter(object):
    dt_0: datetime.datetime
    locked: bool
    model: Type[SimulationBlackBox]
    model_obj: SimulationBlackBox
    model_kwargs: Optional[dict]
    observers: list

    def __init__(self, path: Optional[str] = None) -> None:
        if path:
            self.locked = True
            self._load(path)
        else:
            self.locked = False

    def add_observer(self, observer: str, dt: Union[str, datetime.datetime, Sequence[str], Sequence[datetime.datetime]], dt_s: Union[str, datetime.datetime], dt_e: Union[str, datetime.datetime], dt_shift: datetime.timedelta = None) -> None:        
        self.observers.append([observer, sanitize_dt(dt), sanitize_dt(dt_s), sanitize_dt(dt_e), dt_shift])

    def initialize(self, dt_0: Union[str, datetime.datetime], model: Type[SimulationBlackBox], model_kwargs: dict = {}) -> None:
        if self.locked:
            raise RuntimeError("is locked")
        
        self.dt_0 = sanitize_dt(dt_0)
        self.model_kwargs = model_kwargs
        self.observers = []
        self.model =  model

    def save(self, path: str, **kwargs: Any) -> None:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        data = {attr: getattr(self, attr) for attr in self.__dict__ if not callable(attr) and not attr.startswith("_")}

        for k, v in kwargs.items():
            data[k] = v

        with open(path, "wb") as fh:
            pickle.dump(data, fh)

    def _load(self, path: str) -> None:
        with open(path, "rb") as fh:
            data = pickle.load(fh)

        for k, v in data.items():
            setattr(self, k, v)

    def _run(self, *args: Any, **kwargs: Any) -> None:
        pass


class FittingData(object):
    data_dt: List[np.ndarray]
    data_b: List[np.ndarray]
    data_o: List[np.ndarray]
    data_m: List[np.ndarray]
    data_l: List[int]

    psd_dt: List[np.ndarray]
    psd_fft: List[np.ndarray]    

    length: int
    noise_model: str
    observers: list
    reference_frame: str
    sampling_freq: int

    def __init__(self, observers: list, reference_frame: str) -> None:
        self.observers = observers
        self.reference_frame = reference_frame
        self.length = len(self.observers)

    def add_noise(self, profiles: np.ndarray) -> np.ndarray:
        if self.noise_model == "psd":
            _offset = 0
            for o in range(self.length):
                dt = self.psd_dt[o]
                fft = self.psd_fft[o]
                dtl = self.data_l[o]

                sampling_fac = np.sqrt(self.sampling_freq)

                ensemble_size = len(profiles[0])

                null_flt = (profiles[1 + _offset:_offset + (dtl + 2) - 1, :, 0] != 0)

                # generate noise for each component
                for c in range(3):
                    noise = np.real(np.fft.ifft(np.fft.fft(np.random.normal(0, 1, size=(ensemble_size, len(fft))).astype(np.float32)) * fft) / sampling_fac).T
                    profiles[1 + _offset:_offset + (dtl + 2) - 1, :, c][null_flt] += noise[dt][null_flt]

                _offset += dtl + 2
        else:
            raise NotImplementedError

        return profiles

    def generate_noise(self, noise_model: str = "psd", sampling_freq: int = 300, **kwargs: Any) -> None:
        self.psd_dt = []
        self.psd_fft = []
        self.sampling_freq = sampling_freq

        self.noise_model = noise_model

        if noise_model == "psd":
            for o in range(self.length):
                observer, dt, dt_s, dt_e, _ = self.observers[o]

                observer_obj = getattr(heliosat, observer)()

                _, data = observer_obj.get([dt_s, dt_e], "mag", reference_frame=self.reference_frame, sampling_freq=sampling_freq, use_cache=True, as_endpoints=True)

                data[np.isnan(data)] = 0

                fF, fS = mag_fft(dt, data, sampling_freq=sampling_freq)

                kdt = (len(fS) - 1) / (dt[-1].timestamp() - dt[0].timestamp())
                fT = np.array([int((_.timestamp() - dt[0].timestamp()) * kdt) for _ in dt])

                self.psd_dt.append(fT)
                self.psd_fft.append(fS)
        else:
            raise NotImplementedError

    def generate_data(self, time_offset: Union[int, Sequence], **kwargs: Any) -> None:
        self.data_dt = []
        self.data_b = []
        self.data_o = []
        self.data_m = []
        self.data_l = []

        for o in range(self.length):
            observer, dt, dt_s, dt_e, dt_shift = self.observers[o]

            if hasattr(time_offset, "__len__"):
                dt_s -= datetime.timedelta(hours=time_offset[o])  # type: ignore
                dt_e += datetime.timedelta(hours=time_offset[o])  # type: ignore
            else:
                dt_s -= datetime.timedelta(hours=time_offset)  # type: ignore
                dt_e += datetime.timedelta(hours=time_offset)  # type: ignore

            observer_obj = getattr(heliosat, observer)()

            _, data = observer_obj.get(dt, "mag", reference_frame=self.reference_frame, use_cache=True, **kwargs)

            dt_all = [dt_s] + dt + [dt_e]
            trajectory = observer_obj.trajectory(dt_all, reference_frame=self.reference_frame)
            b_all = np.zeros((len(data) + 2, 3))
            b_all[1:-1] = data
            mask = [1] * len(b_all)
            mask[0] = 0
            mask[-1] = 0

            if dt_shift:
                self.data_dt.extend([_ + dt_shift for _ in dt_all])
            else:
                self.data_dt.extend(dt_all)
            self.data_b.extend(b_all)
            self.data_o.extend(trajectory)
            self.data_m.extend(mask)
            self.data_l.append(len(data))

    def sumstat(self, values: np.ndarray, stype: str = "norm_rmse", use_mask: bool = True) -> np.ndarray:
        if use_mask:
            return sumstat(values, self.data_b, stype, mask=self.data_m, data_l=self.data_l, length=self.length)
        else:
            return sumstat(values, self.data_b, stype, data_l=self.data_l, length=self.length)
