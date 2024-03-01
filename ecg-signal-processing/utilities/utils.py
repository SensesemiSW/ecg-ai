import numpy as np
import os

###########################################################################

def remove_nans(signal):
    if not is_numpy_array(signal):
        signal = np.array(signal)
    nan_mask = np.isnan(signal)
    processedSignals = signal[~nan_mask]
    return processedSignals

def is_numpy_array(data):
    return isinstance(data, np.ndarray)

def normalize_signal(signal):
    if not is_numpy_array(signal):
        signal = np.array(signal)
    signal = signal - np.mean(signal)
    return signal

def convert_to_voltage(signal):
    if not is_numpy_array(signal):
        signal = np.array(signal)
    signal = signal * (5/16777216)
    return signal

def invert_signal(signal):
    if not is_numpy_array(signal):
        signal = np.array(signal)
    # signal = np.max(signal)-signal
    signal = signal - np.mean(signal)
    return signal

def change_signal_sign(signal):
    if not is_numpy_array(signal):
        signal = np.array(signal)
    signal = -signal
    return signal
