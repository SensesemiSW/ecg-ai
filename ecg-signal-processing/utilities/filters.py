from scipy.signal import firls, iirdesign, butter, savgol_filter, iirnotch
from scipy.signal import firwin, lfilter, cheby1, dlti, kaiserord
import numpy as np

##########################################################################

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

#########################################################################

# Common filter constructor function
# Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html

# Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firls.html
def filter_firls(input_signal, antialiasingFilterorder, bands, desired, weights):
    # Low pass filter design
    # h = firls(N, bands, desired=desired, weight=weights) #OR
    ntaps = antialiasingFilterorder - 1  # ntaps should be odd, here antialiasingFilterorder = 10 is even, so ntaps = 9
    h =firls(ntaps, bands, desired=desired, weight=weights)
    return lfilter(h, 1, input_signal)

# Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html
def filter_firwin(input_signal, N, Fpass, Fstop, Fs):
    taps_firwin = firwin(N, cutoff=[Fpass, Fstop], width=None, window='hamming', pass_zero=False, scale=True, fs=Fs)
    hd = lfilter(taps_firwin, 1.0, input_signal)
    return lfilter(hd, 1, input_signal)

def filter_hpass(input_signal):
    b = [-1 / 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 / 32]
    a = [1, -1]
    return lfilter(b, a, input_signal)

# Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cheby1.html
def filter_lpass_cheby1(input_signal, N, Ap, Fp, Fs):
    b, a = cheby1(N, rp=Ap, Wn=Fp, btype='low', analog=False, output='ba', fs=Fs)
    return lfilter(b, a, input_signal)

##########################################################################

# Bandpass filter

Fstop1 = 0.1    # First Stopband Frequency
Fpass1 = 0.4    # First Passband Frequency
Fpass2 = 10     # Second Passband Frequency
Fstop2 = 12     # Second Stopband Frequency
Astop1 = 60     # First Stopband Attenuation (dB)
Apass = 1       # Passband Ripple (dB)
Astop2 = 80     # Second Stopband Attenuation (dB)
match = 'stopband'  # Band to match exactly

# Convert dB values to linear scale
Astop1 = 10 ** (-Astop1 / 20)
Apass = 10 ** (Apass / 20)
Astop2 = 10 ** (-Astop2 / 20)

def filter_bandpass(input_signal, Fs):
    # b, a = iirdesign(wp=[Fpass1, Fpass2], ws=[Fstop1, Fstop2], gpass=Apass, gstop=Astop2, analog=False, ftype='butter', output='ba')
    b, a = butter(4, [Fstop1, Fpass1, Fpass2, Fstop2], fs=Fs, btype='band', analog=False, output='ba')
    return lfilter(b, a, input_signal)

##########################################################################

def filter_moving_avg_binomial(input_signal):

    # Define the filter coefficients
    h = np.array([1/10, 1/10])
    # h = np.array([1/4, 1/4, 1/4])

    # Initialize the binomial coefficients with a single convolution of h
    binomial_coefficients = np.convolve(h, h, mode='full')

    # Repeat the convolution process 4 times to compute the desired coefficients
    for n in range(4):
        binomial_coefficients = np.convolve(binomial_coefficients, h, mode='full')
    # print(binomial_coefficients)

    return lfilter(binomial_coefficients, 1, input_signal)

##########################################################################

def filter_50hz_noise(input_signal):

    # Design a notch filter to remove the 50Hz noise
    freq_tobe_removed = 50.0  # Frequency to be removed in Hz
    Fsampling = 1000  # Sampling frequency in Hz
    Q = 1.5  # Quality factor
    b, a = iirnotch(freq_tobe_removed, Q, fs=Fsampling)
    return lfilter(b, a, input_signal)

##########################################################################

def filter_savitzky_golay(input_signal):
    window_length = 9
    poly_order = 5
    return savgol_filter(input_signal, window_length=window_length, polyorder=poly_order)

##########################################################################

