import logging
import os
from typing import List, Union
import numpy as np
from polars import (DataFrame, read_parquet)
import polars as pl
from pandas import Series
from scipy.signal import resample
from scipy.ndimage import median_filter
import io

SAMPLING_RATE = int(os.environ.get('SAMPLING_RATE', 100))

def serialize(data):
    if isinstance(data, DataFrame):
        buffer = io.BytesIO()
        data.write_parquet(buffer)
        buffer.seek(0)
        return buffer.getvalue()
    else:
        data.tobytes()

def deserialize(parquet_bytes):
    return read_parquet(io.BytesIO(parquet_bytes))

def deserialize_np(numpy_bytes):
    return np.frombuffer(numpy_bytes)

def remove_outliers(data, boundary=1.5):
    quartile1 = data.quantile(0.01)
    quartile3 = data.quantile(0.99)
    iqr = quartile3 - quartile1
    lower_bound = quartile1 - boundary * iqr
    upper_bound = quartile3 + boundary * iqr
    data = data.with_columns(
        [pl.col(col).clip(lower_bound[col], upper_bound[col]) for col in data.columns]
    )
    return data

# min-max-scaling to [-1, 1]
def minMax(data):
    min_ = np.nanmin(data, axis=(-1, -2), keepdims=True)
    max_ = np.nanmax(data, axis=(-1, -2), keepdims=True)
    max_min_diff = (max_ - min_)
    max_min_diff = np.where(max_min_diff == 0, 1, max_min_diff)
    data = 2*((data - min_) / max_min_diff) -1
    return data, min_, max_


# mean-std-standardization
def meanStd(data):
    mean_ = np.nanmean(data, axis=(-1, -2), keepdims=True)
    std_ = np.std(data, axis=(-1, -2), keepdims=True)
    #var_ = np.var(data, axis=(-1, -2), keepdims=True)
    data = ((data - mean_) / std_)
    return data, mean_, std_

def movingMean(data, window_size=201):
        norm_data = np.zeros_like(data)
        convolve = np.ones(window_size)
        data_convs = []
        for channel in range(data.shape[0]):
            data_conv = np.convolve(data[channel], convolve, 'same') / window_size
            data_convs.append(data_conv)
            norm_data[channel] = (data[channel] - data_conv)

        return norm_data, data_convs


def movingMedian(data, window_size=201, mode='reflect'):
    norm_data = np.zeros_like(data)

    # Iterate over each channel independently
    for channel in range(data.shape[0]):
        # Compute the moving median
        median = median_filter(data[channel], size=window_size, mode=mode, cval=0, axes=(-1))
        norm_data[channel] = (data[channel] - median)
        
    return norm_data

def ecg_normalize(
    data: Union[np.ndarray, List[np.ndarray], DataFrame, List[DataFrame]],
    **kwargs
):
    if isinstance(data, list):
        datalist = []
        for single_data in data:
            datalist.append(ecg_normalize(single_data, **kwargs))
        return np.asarray(datalist)
    
    sampling_rate: int = kwargs.get('sampling_rate', 100)
    trend_removal: Union[None, str] = kwargs.get('trend_removal', "movingMean")
    standardizer: Union[None, str] = kwargs.get('standardizer', None)
    scale: Union[None, str] = kwargs.get('scale', "milliVolt")
    
    if isinstance(data, DataFrame):
        data = data.to_numpy()

    if sampling_rate != SAMPLING_RATE:
        data = resample_multichannel(data, SAMPLING_RATE, sampling_rate)
    
    if data.shape[0] > data.shape[1]:
        data = data.transpose(1,0)

    match trend_removal:
        case "movingMean":
            data, _ = movingMean(data, window_size=(sampling_rate*2+1))
        case "movingMedian":
            data = movingMedian(data, window_size=(sampling_rate*2+1), mode='reflect')
        case _:
            pass
    
    curr_scale = checkScale(data)
    data = normalize_scale(data, curr_scale, scale)

    match standardizer:
        case "minMax":
            data, _, _ = minMax(data)
        case "meanStd":
            data, _, _ = meanStd(data)
        case _:
            pass

    if data.shape[1] > data.shape[0]:
        data = data.transpose(1,0)


    return data.transpose(-1,-2).astype('float32')
    
def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1)

def checkScale(x):
    amplitude = np.max(np.abs(x))
    
    if amplitude > 100:
        return "mikroVolt"
    else:
        return "milliVolt"

    
def normalize_scale(x, curr_scale, to_scale):
    scale_factors = {
        ("milliVolt", "mikroVolt"): 1000,
        ("mikroVolt", "milliVolt"): 1/1000,
    }
    if curr_scale == to_scale:
        return x
    else:
        return x * scale_factors[(curr_scale, to_scale)]
#
#
#
# The following have been modified from wfdb`s resample_multichan and resample_sig
# Reduces dependencies
#
def resample_multichannel(xs, fs, fs_target):
    """
    Resample multiple channels with their annotations.

    Parameters
    ----------
    xs: ndarray
        The signal array.
    fs : int, float
        The original frequency.
    fs_target : int, float
        The target frequency.

    Returns
    -------
    ndarray
        Array of the resampled signal values.

    """
    lx = []
    for chan in range(xs.shape[1]):
        resampled_x = resample_signal(xs[:, chan], fs, fs_target)
        lx.append(resampled_x)

    return np.column_stack(lx)


def resample_signal(x, fs, fs_target):
    """
    Resample a signal to a different frequency.

    Parameters
    ----------
    x : ndarray
        Array containing the signal.
    fs : int, float
        The original sampling frequency.
    fs_target : int, float
        The target frequency.

    Returns
    -------
    resampled_x : ndarray
        Array of the resampled signal values.

    """
    if fs == fs_target:
        return x

    new_length = int(x.shape[0] * fs_target / fs)
    # Resample the array if NaN values are present
    if np.isnan(x).any():
        x = Series(x.reshape((-1,))).interpolate().values
    resampled_x = resample(x, num=new_length)
   
    return resampled_x
