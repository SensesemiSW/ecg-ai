import os
import numpy as np
from scipy.signal import detrend
from heart_rate import get_heart_rate, T
from utilities.filters import filter_50hz_noise, filter_hpass, filter_moving_avg_binomial, filter_savitzky_golay
from utilities.plotter import plot_time_signals_array_multiplot, plot_amplitude_freq_spectrum, plot_1, plot_12, save_as_png, plot_6, close_plots
from utilities.utils import convert_to_voltage, normalize_signal, remove_nans

###########################################################################

def preprocess_ecg_signal(sig):

    plot_signals = []
    time_vector = np.arange(0, len(sig)) * T

    # Regular Preprocessing
    sig = remove_nans(sig)
    # sig = np.nan_to_num(sig)  # Replace NaN with 0
    sig = convert_to_voltage(sig)
    sig = detrend(sig)
    sig = normalize_signal(sig)

    plot_signal = {
        'title': 'Before 50Hz Noise Removal',
        'xlabel': 'time(s)',
        'ylabel': 'Voltage (V)',
        'data': sig
    }
    plot_signals.append(plot_signal)

    freq_spectrum = {
        'title': 'Before 50Hz Noise Removal',
        'xlabel': 'Frequency (Hz)',
        'ylabel': 'Amplitude',
        'sampling_freq': 1000,
        'data': sig
    }
    plot_amplitude_freq_spectrum(freq_spectrum)

    # # 50Hz noise removal

    # sig = filter_50hz_noise(sig)

    # plot_signal = {
    #     'title': 'After 50Hz Noise Removal',
    #     'xlabel': 'time(s)',
    #     'ylabel': 'Voltage (V)',
    #     'data': sig
    # }
    # plot_signals.append(plot_signal)

    # freq_spectrum = {
    #     'title': 'After 50Hz Noise Removal',
    #     'xlabel': 'Frequency (Hz)',
    #     'ylabel': 'Amplitude',
    #     'sampling_freq': 1000,
    #     'data': sig
    # }
    # plot_amplitude_freq_spectrum(freq_spectrum)

    # Moving average binomial filter

    sig = filter_moving_avg_binomial(sig)

    plot_signal = {
        'title': 'After moving average binomial filter',
        'xlabel': 'time(s)',
        'ylabel': 'Voltage (V)',
        'data': sig
    }
    plot_signals.append(plot_signal)

    # Savitzky-Golay filter

    sig = filter_savitzky_golay(sig)

    plot_signal = {
        'title': 'After Savitzky-Golay filter',
        'xlabel': 'time(s)',
        'ylabel': 'Voltage (V)',
        'data': sig
    }
    plot_signals.append(plot_signal)

    # High pass filter
    sig = filter_hpass(sig)

    actual_signal = {
        'title': 'After high pass filter',
        'xlabel': 'time(s)',
        'ylabel': 'Voltage (V)',
        'data': sig
    }
    plot_signals.append(actual_signal)

    # plot_time_signals_array_multiplot(time_vector, plot_signals)

    return sig

def normalize_ecg(ecg_signal):
    min_val = np.min(ecg_signal)
    max_val = np.max(ecg_signal)

    normalized_ecg = (ecg_signal - min_val) / (max_val - min_val)

    return normalized_ecg

###########################################################################

# ECG lead aVR, aVF and aVL (Goldberger’s leads)

# These leads were originally constructed by Goldberger.
# In these leads the exploring electrode is compared with a reference which
# is based on an average of the other two limb electrodes.
# The letter a stands for augmented, V for voltage and R is right arm, L is left arm and F is foot.

# There are three advantages of inverting aVR into –aVR:
# 1) -aVR fills the gap between lead I and lead II in the coordinate system.
# 2) -aVR facilitates the calculation of the heart’s electrical axis.
# 3) -aVR improves diagnosis of acute ischemia/infarction (inferior and lateral ischemia/infarction).

def goldberger(lead1, lead2, lead3):
    aVR = (lead1 + lead2) / 2
    aVR = -aVR
    aVF = (lead2 + lead3) / 2
    aVL = (lead1 - lead3) / 2
    return aVR, aVF, aVL

###########################################################################

def get_ecg_lead12(lead_one, lead_two, v1, v2, v3, v4, v5, v6, record_id):

    lead_two_temp = np.copy(lead_two)
    heart_rate, intervals, pqrst_locs = get_heart_rate(lead_two_temp, True, False)

    lead_one = preprocess_ecg_signal(lead_one)
    lead_one_plot = normalize_ecg(lead_one)
    lead_two = preprocess_ecg_signal(lead_two)
    lead_two_plot = normalize_ecg(lead_two)
    lead_three = np.subtract(lead_two, lead_one)
    lead_three_plot = normalize_ecg(lead_three)

    v1 = preprocess_ecg_signal(v1)
    v1_plot = normalize_ecg(v1)
    v2 = preprocess_ecg_signal(v2)
    v2_plot = normalize_ecg(v2)
    v3 = preprocess_ecg_signal(v3)
    v3_plot = normalize_ecg(v3)
    v4 = preprocess_ecg_signal(v4)
    v4_plot = normalize_ecg(v4)
    v5 = preprocess_ecg_signal(v5)
    v5_plot = normalize_ecg(v5)
    v6 = preprocess_ecg_signal(v6)
    v6_plot = normalize_ecg(v6)

    aVR, aVF, aVL = goldberger(lead_one, lead_two, lead_three)
    aVR_plot = normalize_ecg(aVR)
    aVF_plot = normalize_ecg(aVF)
    aVL_plot = normalize_ecg(aVL)

    ecg_data = np.array([lead_one_plot.tolist(), lead_two_plot.tolist(), lead_three_plot.tolist(), aVR_plot.tolist(), aVL_plot.tolist(), aVF_plot.tolist(), v1_plot.tolist(), v2_plot.tolist(), v3_plot.tolist(), v4_plot.tolist(), v5_plot.tolist(), v6_plot.tolist()])

    plot_12(ecg_data)
    default_path = os.path.join(os.getcwd(), "images")
    if not os.path.exists(default_path):
        os.mkdir(default_path)
    save_as_png(f'{record_id}')
    close_plots()
    return lead_one, lead_two, lead_three, v1, v2, v3, v4, v5, v6, aVR, aVF, aVL, heart_rate, intervals, pqrst_locs, default_path

###########################################################################

def get_ecg_lead6(lead_one, lead_two, record_id):

    lead_two_temp = np.copy(lead_two)
    heart_rate, intervals, pqrst_locs = get_heart_rate(lead_two_temp, True, False)

    lead_one = preprocess_ecg_signal(lead_one)
    lead_one_plot = normalize_ecg(lead_one)
    lead_two = preprocess_ecg_signal(lead_two)
    lead_two_plot = normalize_ecg(lead_two)
    lead_three = np.subtract(lead_two, lead_one)
    lead_three_plot = normalize_ecg(lead_three)

    aVR, aVF, aVL = goldberger(lead_one, lead_two, lead_three)
    aVR_plot = normalize_ecg(aVR)
    aVF_plot = normalize_ecg(aVF)
    aVL_plot = normalize_ecg(aVL)
    ecg_data = np.array([lead_one_plot.tolist(), lead_two_plot.tolist(), lead_three_plot.tolist(), aVR_plot.tolist(), aVL_plot.tolist(), aVF_plot.tolist()])
    plot_6(ecg_data)
    default_path = os.path.join(os.getcwd(), "images")
    if not os.path.exists(default_path):
        os.mkdir(default_path)
    save_as_png(f'{record_id}')
    close_plots()
    return lead_one, lead_two, lead_three, aVR, aVF, aVL, heart_rate, intervals, pqrst_locs, default_path

###########################################################################

def get_ecg_lead1(lead_one, record_id):
    lead_one_temp = np.copy(lead_one)
    heart_rate, intervals, pqrst_locs = get_heart_rate(lead_one_temp, True, False)
    lead_one = preprocess_ecg_signal(lead_one)
    ecg_data = np.array([lead_one.tolist()])
    plot_1(ecg_data)
    default_path = os.path.join(os.getcwd(), "images")
    if not os.path.exists(default_path):
        os.mkdir(default_path)
    save_as_png(f'{record_id}')
    close_plots()
    file_path = os.path.join(default_path, f'{record_id}.png')
    return lead_one, heart_rate, intervals, pqrst_locs, default_path
