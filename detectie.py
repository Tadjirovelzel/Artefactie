'''
Deelt het databestand op in stukken en kijkt door middel van een fast fourier transform
of een opvallende verandering plaats vindt in het signaal. Dat kan duiden op een artefact.
'''

import numpy as np
from scipy.signal import find_peaks


def compute_fft(signal, fs, frange=(0.5, 10)):
    """
    Compute FFT on signal and return dominant frequency within a physiological range.

    Inputs:
        signal: 1D numpy array
        fs: sampling rate (Hz)
        frange: (min, max) frequency range to consider (Hz), default 0.5-10 Hz for BP

    Outputs:
        freqs: frequency array clipped to frange (Hz)
        magnitudes: corresponding FFT magnitudes
        dominant_freq: frequency with highest magnitude within frange (Hz)
    """
    N = len(signal)
    freqs = np.fft.rfftfreq(N, d=1 / fs)
    magnitudes = np.abs(np.fft.rfft(signal)) / N

    mask = (freqs >= frange[0]) & (freqs <= frange[1])
    freqs = freqs[mask]
    magnitudes = magnitudes[mask]

    dominant_freq = freqs[np.argmax(magnitudes)]

    return freqs, magnitudes, dominant_freq


def compute_rms(signal):
    """
    Compute Root Mean Square (RMS) of signal.

    Inputs:
        signal: 1D numpy array

    Outputs:
        rms: scalar RMS value
    """
    return np.sqrt(np.mean(np.array(signal, dtype=float) ** 2))


def detect_peaks_abp(signal, fs, rms_factor=0.5, hr_range=(0.5, 3.0)):
    """
    Detect systolic peaks in ABP signal.

    Uses the dominant FFT frequency (heart rate) to set the minimum distance
    between peaks, and the RMS amplitude to set the prominence threshold.

    Inputs:
        signal: 1D numpy array of ABP values
        fs: sampling rate (Hz)
        rms_factor: prominence threshold = rms_factor * RMS  (default 0.5)
        hr_range: (min, max) heart rate frequency in Hz, default (0.5, 3.0) = 30-180 bpm

    Outputs:
        peaks: indices of detected peaks
        dominant_freq: dominant heart rate frequency (Hz)
        prominence_threshold: prominence value used for detection
    """
    _, _, dominant_freq = compute_fft(signal, fs, frange=hr_range)
    rms = compute_rms(signal)

    # Minimum distance between peaks: 60% of the expected cardiac cycle length
    min_distance = int(fs / dominant_freq * 0.6)
    prominence_threshold = rms * rms_factor

    peaks, _ = find_peaks(signal, distance=min_distance, prominence=prominence_threshold)

    return peaks, dominant_freq, prominence_threshold
