# ECG Signal Processing - Features

### Time Domain Features:

1.	Variance: Measures the spread of signal values around the mean. Suitable for ECG signals as it provides information about signal variability, which can be indicative of heart rate variability (HRV) and arrhythmias.

2.	Zero Crossing: These features can help identify the frequency of signal crossings through zero, which can provide insights into waveform characteristics. In ECG, it reflects the heart rate variability.

3.	Root Mean Square (RMS): Describes the average power of the signal. In ECG analysis, RMS can indicate signal amplitude and overall energy, aiding in the detection of abnormalities or noise.

4.	Mean Absolute Value (MAV): Provides the average absolute magnitude of the signal. Useful for assessing signal amplitude and detecting changes in waveform shape. This represents the average of the absolute values of the signal. In ECG, it reflects the overall electrical activity of the heart.

5.	Average Amplitude Change: Indicates the average rate of change in signal amplitude. Helpful for detecting dynamic changes in ECG waveform morphology.
 
### Frequency Domain Features:

1.	Mean Power: Represents the average power distribution across different frequency bands. Valuable for ECG signals to analyze spectral content and identify frequency components associated with cardiac activity.

2.	Mean Frequency: Indicates the average frequency of the signal spectrum. Relevant for ECG analysis to assess dominant frequency components and detect abnormalities in rhythm.

3.	Total Power: Represents the overall power content of the signal spectrum. Important for ECG analysis to assess signal strength and detect anomalies.

### Wavelet Domain Features:

Mean, Kurtosis, Standard Deviation, Energy, and Entropy of the Coefficients: These features provide insights into signal characteristics across multiple scales in both time and frequency domains. Suitable for ECG signals to capture complex waveform variations and detect abnormalities with enhanced sensitivity.
 
### Other Features:

1. Kurtosis: Measures the "tailedness" of the signal distribution. Relevant for ECG signals to assess waveform shape and detect deviations from normal distributions.
  
3. H2-H1: Represents the difference between two halves of the signal. Helpful for ECG signals to assess waveform symmetry and detect abnormalities in waveform morphology.
