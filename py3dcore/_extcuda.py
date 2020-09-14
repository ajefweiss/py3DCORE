# -*- coding: utf-8 -*-

"""_extcuda.py

Utility functions for Numba CUDA.
"""

import logging
import numba.cuda as cuda
import numpy as np

from functtools import lru_cache


@lru_cache
def get_threads_blocks(length, max_thread_size=9):
    """Calculate thread/block sizes for launching CUDA kernels.

    Parameters
    ----------
    length : int
        Number of threads.
    max_thread_size : int
        Maximum thread size in 2^N.

    Returns
    -------
    (int, int)
        Number of threads and thread blocks.
    """
    logger = logging.getLogger(__name__)

    counter = 0

    _length = length

    while True:
        if _length % 2 == 1:
            logger.warning("get_threads_blocks could not fully factorize the number of threads")
            break

        _length //= 2
        counter += 1

        if counter >= max_thread_size:
            break

    return 2**counter, _length


def initialize_array(shape, value, dtype=np.float32):
    """Create and initialize cuda device array.

    Parameters
    ----------
    shape : tuple
        Array shape.
    value: np.float32
        Value to fill array with.
    dtype: type, optional
        Array data type, by default np.float32.

    Returns
    -------
    Initialized cuda device array.
        numba.cuda.cudadrv.devicearray.DeviceNDArray
    """
    arr = cuda.device_array(shape, dtype=dtype).ravel()

    threads, blocks = get_threads_blocks(len(arr))

    _cuda_initialize_array[blocks, threads](arr, value)

    return arr.reshape(shape)


@cuda.jit
def _cuda_initialize_array(array, value):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    array[i] = value


def extract_array_column(array, column):
    """Exctract column from array .

    Parameters
    ----------
    array : numba.cuda.cudadrv.devicearray.DeviceNDArray
        Cuda Array
    column: int, optional
        Column to be extracted

    Returns
    -------
    Extracted column as cuda device array.
        numba.cuda.cudadrv.devicearray.DeviceNDArray
    """
    _array = cuda.device_array((len(array),), dtype=array.dtype)

    threads, blocks = get_threads_blocks(len(_array))

    _cuda_extract_column[blocks, threads](array, column, _array)

    return _array


@cuda.jit
def _cuda_extract_column(array, column, out):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    out[i] = array[i, column]
