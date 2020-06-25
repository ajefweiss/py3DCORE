# -*- coding: utf-8 -*-

"""_extcuda.py

Utility functions for Numba CUDA.
"""

import math
import numba
import numba.cuda as cuda
import numpy as np


def get_threads_blocks(length):
    """Calculate thread/block sizes for launching CUDA kernels.

    Parameters
    ----------
    length : int
        Total number of threads.

    Returns
    -------
    (int, int)
        Number of threads and thread blocks.
    """
    if length < 512:
        return length, 1
    elif length % 512 == 0:
        return 512, length // 512
    elif length % 256 == 0:
        return 256, length // 256
    elif length % 128 == 0:
        return 128, length // 128
    elif length % 64 == 0:
        return 64, length // 64
    elif length % 32 == 0:
        return 32, length // 32
    else:
        raise ValueError("number of threads is invalid")


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
