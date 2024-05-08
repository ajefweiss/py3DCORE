# -*- coding: utf-8 -*-

import datetime
from typing import Any, List, Optional, Sequence, Tuple, Union

import heliosat
import numba
import numpy as np
from scipy.signal import detrend, welch


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

                null_flt = profiles[1 + _offset : _offset + (dtl + 2) - 1, :, 0] != 0

                # generate noise for each component
                for c in range(3):
                    noise = np.real(
                        np.fft.ifft(
                            np.fft.fft(
                                np.random.normal(
                                    0, 1, size=(ensemble_size, len(fft))
                                ).astype(np.float32)
                            )
                            * fft
                        )
                        / sampling_fac
                    ).T

                    profiles[1 + _offset : _offset + (dtl + 2) - 1, :, c][
                        null_flt
                    ] += noise[dt][null_flt]

                _offset += dtl + 2
        elif self.noise_model == "gaussian":
            _offset = 0
            for o in range(self.length):
                dtl = self.data_l[o]
                sampling_fac = np.sqrt(self.sampling_freq)

                ensemble_size = len(profiles[0])

                null_flt = profiles[1 + _offset : _offset + (dtl + 2) - 1, :, 0] != 0

                # generate noise for each component
                for c in range(3):
                    noise = (
                        np.random.normal(
                            0, self.gauss_noise_level[o], size=(ensemble_size, dtl)
                        )
                        .astype(np.float32)
                        .T
                    )

                    profiles[1 + _offset : _offset + (dtl + 2) - 1, :, c][
                        null_flt
                    ] += noise[null_flt]

                _offset += dtl + 2
        else:
            raise NotImplementedError

        return profiles

    def generate_noise(
        self, noise_model: str = "psd", sampling_freq: int = 300, **kwargs: Any
    ) -> None:
        self.psd_dt = []
        self.psd_fft = []
        self.sampling_freq = sampling_freq

        self.gauss_noise_level = []

        self.noise_model = noise_model

        print("using noise model", noise_model)

        if noise_model == "psd":
            for o in range(self.length):
                observer, dt, dt_s, dt_e, _ = self.observers[o]

                observer_obj = getattr(heliosat, observer)()

                _, data = observer_obj.get(
                    [dt_s, dt_e],
                    "mag",
                    reference_frame=self.reference_frame,
                    sampling_freq=sampling_freq,
                    cached=True,
                    as_endpoints=True,
                )

                data[np.isnan(data)] = 0

                # fF, fS = power_spectral_density(dt, data, format_for_fft=True)
                fF, fS = mag_fft(dt, data, sampling_freq=sampling_freq)

                kdt = (len(fS) - 1) / (dt[-1].timestamp() - dt[0].timestamp())
                fT = np.array(
                    [int((_.timestamp() - dt[0].timestamp()) * kdt) for _ in dt]
                )

                self.psd_dt.append(fT)
                self.psd_fft.append(fS)
        elif noise_model == "gaussian":
            for o in range(self.length):
                observer, dt, dt_s, dt_e, _ = self.observers[o]

                observer_obj = getattr(heliosat, observer)()

                _, data = observer_obj.get(
                    [dt_s, dt_e],
                    "mag",
                    reference_frame=self.reference_frame,
                    sampling_freq=sampling_freq,
                    cached=True,
                    as_endpoints=True,
                )

                data[np.isnan(data)] = 0

                # TODO: remove hard coding
                self.gauss_noise_level.append(kwargs.get("noise_level"))
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

            instrument = kwargs.get("instrument", "mag")

            if hasattr(time_offset, "__len__"):
                dt_s -= datetime.timedelta(hours=time_offset[o])
                dt_e += datetime.timedelta(hours=time_offset[o])
            else:
                dt_s -= datetime.timedelta(hours=time_offset)
                dt_e += datetime.timedelta(hours=time_offset)

            observer_obj = getattr(heliosat, observer)()

            _, data = observer_obj.get(
                dt,
                instrument,
                reference_frame=self.reference_frame,
                cached=True,
                **kwargs
            )

            dt_all = [dt_s] + dt + [dt_e]
            trajectory = observer_obj.trajectory(
                dt_all, reference_frame=self.reference_frame
            )
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

    def sumstat(
        self, values: np.ndarray, stype: str = "norm_rmse", use_mask: bool = True
    ) -> np.ndarray:
        if use_mask:
            return sumstat(
                values,
                self.data_b,
                stype,
                mask=self.data_m,
                data_l=self.data_l,
                length=self.length,
            )
        else:
            return sumstat(
                values, self.data_b, stype, data_l=self.data_l, length=self.length
            )


def sumstat(
    values: np.ndarray, reference: np.ndarray, stype: str = "norm_rmse", **kwargs: Any
) -> np.ndarray:
    if stype == "norm_rmse":
        data_l = np.array(kwargs.pop("data_l"))
        length = kwargs.pop("length")
        mask = kwargs.pop("mask", None)

        varr = np.array(values)

        rmse_all = np.zeros((length, varr.shape[1]))

        _dl = 0

        for i in range(length):
            slc = slice(_dl, _dl + data_l[i] + 2)
            values_i = varr[slc]
            reference_i = np.array(reference)[slc]

            normfac = np.mean(np.sqrt(np.sum(reference_i**2, axis=1)))

            if mask is not None:
                mask_i = np.array(mask)[slc]
            else:
                mask_i = None

            rmse_all[i] = rmse(values_i, reference_i, mask=mask_i) / normfac

            _dl += data_l[i] + 2

        return rmse_all.T
    else:
        raise NotImplementedError


@numba.njit
def rmse(
    values: np.ndarray, reference: np.ndarray, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    rmse = np.zeros(len(values[0]))

    if mask is not None:
        for i in range(len(reference)):
            _error_rmse(values[i], reference[i], mask[i], rmse)

        rmse = np.sqrt(rmse / len(values))

        mask_arr = np.copy(rmse)

        for i in range(len(reference)):
            _error_mask(values[i], mask[i], mask_arr)

        return mask_arr
    else:
        for i in range(len(reference)):
            _error_rmse(values[i], reference[i], 1, rmse)

        rmse = np.sqrt(rmse / len(values))

        return rmse


@numba.njit
def _error_mask(values_t: np.ndarray, mask: np.ndarray, rmse: np.ndarray) -> None:
    for i in numba.prange(len(values_t)):
        _v = np.sum(values_t[i] ** 2)
        if (_v > 0 and mask == 0) or (_v == 0 and mask != 0):
            rmse[i] = np.inf


@numba.njit
def _error_rmse(
    values_t: np.ndarray, ref_t: np.ndarray, mask: np.ndarray, rmse: np.ndarray
) -> None:
    for i in numba.prange(len(values_t)):
        if mask == 1:
            rmse[i] += np.sum((values_t[i] - ref_t) ** 2)


def mag_fft(
    dt: Sequence[datetime.datetime], bdt: np.ndarray, sampling_freq: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the mean power spectrum distribution from a magnetic field measurements over all three vector components.
    Note: Assumes that P(k) is the same for all three vector components.
    """
    n_s = int(((dt[-1] - dt[0]).total_seconds() / 3600) - 1)
    n_perseg = np.min([len(bdt), 256])

    p_bX = detrend(bdt[:, 0], type="linear", bp=n_s)
    p_bY = detrend(bdt[:, 1], type="linear", bp=n_s)
    p_bZ = detrend(bdt[:, 2], type="linear", bp=n_s)

    _, wX = welch(p_bX, fs=1 / sampling_freq, nperseg=n_perseg)
    _, wY = welch(p_bY, fs=1 / sampling_freq, nperseg=n_perseg)
    wF, wZ = welch(p_bZ, fs=1 / sampling_freq, nperseg=n_perseg)

    wS = (wX + wY + wZ) / 3

    # convert P(k) into suitable form for fft
    fF = np.fft.fftfreq(len(p_bX), d=sampling_freq)
    fS = np.zeros((len(fF)))

    for i in range(len(fF)):
        k = np.abs(fF[i])
        fS[i] = np.sqrt(wS[np.argmin(np.abs(k - wF))])

    return fF, fS
