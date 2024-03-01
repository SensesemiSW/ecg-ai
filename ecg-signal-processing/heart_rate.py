import numpy as np
from scipy.signal import detrend
from utilities.plotter import plot_time_signals, plot_time_signals_array_multiplot, show_plots
from utilities.filters import filter_firls, filter_hpass, filter_lpass_cheby1, filter_moving_avg_binomial
from utilities.rr_interval_finder import find_ecg_intervals, find_rr_intervals_heart_rate
from utilities.utils import change_signal_sign, remove_nans, convert_to_voltage, normalize_signal
from pywt import waverec, wavedec
# from app.common.utils import is_debug_environment
# from app.algos.spo2 import ppg_wavelet_transform

###########################################################################

# Filter design parameters

Fs = 104    # Sampling frequency # ??? Or should we use 112
T = 1 / Fs  # Sampling time
N = 7       # Low Pass Filter order - numtaps (Must be odd)
Fp = 35     # Passband Frequency
Ap = 1      # Passband Ripple (dB)

# Antialiasing Filter
antialiasingFilterorder = 10 # Antialiasing filter order
Fpass = 38      # Passband Frequency
Fstop = 45      # Stopband Frequency
Wpass = 1       # Passband Weight
Wstop = 1       # Stopband Weight

nyquist = 0.5 * Fs # Nyquist frequency
bands = [0, Fpass / nyquist, Fstop / nyquist, 1.0]
desired = [1, 1, 0, 0]
weights = [Wpass, Wstop]

###########################################################################

def preprocess_hr(sig):

    time_axis = np.arange(len(sig)) * T

    sig = remove_nans(sig)

    show_plots()

    plot_signals = []
    plot_signal = {
        'title': 'Actual Data',
        'xlabel': 'time(s)',
        'ylabel': 'Raw Data',
        'data': sig
    }
    plot_signals.append(plot_signal)

    # Convert to voltage
    sig = convert_to_voltage(sig)

    # Plot signal
    # plot_signal = {
    #     'title': 'Before detrending',
    #     'xlabel': 'time(s)',
    #     'ylabel': 'Voltage (V)',
    #     'data': sig
    # }
    # plot_signals.append(plot_signal)

    # Detrend signal (Linear detrending)
    sig = detrend(sig)

    plot_signal = {
        'title': 'After detrending',
        'xlabel': 'time(s)',
        'ylabel': 'Voltage (mV)',
        'data': sig
    }
    plot_signals.append(plot_signal)

    sig = ppg_wavelet_transform(sig)

    plot_signal = {
        'title': 'After wavelet transform',
        'xlabel': 'time(s)',
        'ylabel': 'Voltage (mV)',
        'data': sig
    }
    plot_signals.append(plot_signal)

    # Normalize signal
    sig = normalize_signal(sig)

    # plot_signal = {
    #     'title': 'After normalization',
    #     'xlabel': 'time(s)',
    #     'ylabel': 'Voltage (mV)',
    #     'data': sig
    # }
    # plot_signals.append(plot_signal)

    # Anti-aliasing filter
    sig = filter_firls(sig, antialiasingFilterorder, bands, desired, weights)

    plot_signal = {
        'title': 'After anti-aliasing filter',
        'xlabel': 'time(s)',
        'ylabel': 'Voltage (mV)',
        'data': sig
    }
    plot_signals.append(plot_signal)

    # Low pass filter
    sig = filter_lpass_cheby1(sig, N, Ap, Fp, Fs)

    plot_signal = {
        'title': 'After low pass filter',
        'xlabel': 'time(s)',
        'ylabel': 'Voltage (mV)',
        'data': sig
    }
    plot_signals.append(plot_signal)

    # High pass filter
    sig = filter_hpass(sig)

    plot_signal = {
        'title': 'After high pass filter',
        'xlabel': 'time(s)',
        'ylabel': 'Voltage (mV)',
        'data': sig
    }
    plot_signals.append(plot_signal)

    # Plot frequency spectrum
    # freq_spectrum_data = {
    #     'title': 'Frequency Amplitude Spectrum',
    #     'xlabel': 'frequency (Hz)',
    #     'ylabel': '|mV(f)|',
    #     'data': sig,
    #     'sampling_freq': 103
    # }
    # plot_amplitude_freq_spectrum(freq_spectrum_data)

    plot_time_signals_array_multiplot(time_axis, plot_signals)

    return sig

###########################################################################

def find_form_factor(ecg, q_locs):
    if is_debug_environment():
        # Calculate form factor
        ECG_CYCLE_SPAN = 80 # 80 represents span of 400 ms (one ECG cycle)
        ff = []
        for i in range(len(q_locs) - 1):
            sig = ecg[q_locs[i]:q_locs[i] + ECG_CYCLE_SPAN]
            sig_d = np.diff(sig)  # first derivative
            var1 = np.var(sig)  # signal variance
            var2 = np.var(sig_d)  # variance of first derivative
            var2d = np.var(np.diff(sig_d))  # variance of second derivative
            Mx = np.sqrt(var2 / var1)
            Mxd = np.sqrt(var2d / var2)
            ff.append(Mxd / Mx)
        # print("ff = ", ff)

# find_form_factor(ecg, q_locs)

###########################################################################

def ppg_wavelet_transform(ppg_signal):
    # Define the wavelet filters
    # low, high = Wavelet('db8').filter_bank # ??? Not using low and high
    wt = wavedec(ppg_signal, 'db8', mode='symmetric', level=10)

    # Zeroing coefficients that hold very low-frequency components (d7, d8, d9, d10, and a10)
    wt_rec = wt.copy()
    wt_rec[0][7:] = 0
    wt_rec[1][7:] = 0
    wt_rec[2][7:] = 0
    # wt_rec = [0] * len(wt)
    # for i in range(6):
    #     wt_rec[i] = wt[i]

    # Reconstruct without very high-frequency and very low-frequency coefficients
    aux = waverec(wt_rec, 'db8')

    offset = np.mean(ppg_signal)
    ppg_signal = aux - np.mean(aux) + offset
    transformed_sig = ppg_signal

    return transformed_sig
###########################################################################

def get_heart_rate(sig, mirror_signal=False, average_heart_rate=False):

    # Find R-peaks
    f_sample = Fs
    processed = preprocess_hr(sig)
    r_locs = find_rr_intervals_heart_rate(processed, f_sample)
    if len(r_locs) < 2 :
        if mirror_signal: # Try again with mirrored signal
            updated_signal = change_signal_sign(sig)
            processed = preprocess_hr(updated_signal)
            r_locs = find_rr_intervals_heart_rate(processed, f_sample)
    if len(r_locs) < 2 :
        print("Not enough R peaks found")
        return None, None, None

    # Calculate heart rate
    rr = np.diff(r_locs)
    heart_rate = Fs / (r_locs[1] - r_locs[0]) * 60
    heart_rate = round(heart_rate)
    if average_heart_rate:
        heart_rate = np.mean(Fs / rr) * 60

    processed = filter_moving_avg_binomial(processed)
    intervals, pqrst_locs = find_ecg_intervals(processed, f_sample)

    return heart_rate, intervals, pqrst_locs

###########################################################################

def sanitize_hr_durations(heart_rate, durations):
    durationQR  = None
    durationRS  = None
    durationQRS = None
    if durations is not None and type(durations) is dict:
        if 'qr' in durations and durations['qr'] is not None:
            durationQR = round(durations['qr'] * 1000)
        if 'rs' in durations and durations['rs'] is not None:
            durationRS = round(durations['rs'] * 1000)
        if 'qrs' in durations and durations['qrs'] is not None:
            durationQRS = round(durations['qrs'] * 1000)
    heart_rate = round(heart_rate) if heart_rate is not None else None
    return heart_rate, durationQR, durationRS, durationQRS

###########################################################################
