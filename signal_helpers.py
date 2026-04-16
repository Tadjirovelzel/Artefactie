"""
signal_helpers.py
Helper functions for filtering and spectral analysis of pressure signals.
"""

import numpy as np
from scipy.signal import butter, filtfilt, spectrogram as _spectrogram


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def lowpass(signal, fs, cutoff_hz=2.0, order=4):
    """
    Zero-phase Butterworth low-pass filter.

    Parameters
    ----------
    signal     : 1D array
    fs         : sample rate (Hz)
    cutoff_hz  : -3 dB frequency (Hz)
    order      : filter order

    Returns
    -------
    1D array of the same length as *signal*
    """
    nyq = 0.5 * fs
    b, a = butter(order, cutoff_hz / nyq, btype="low")
    return filtfilt(b, a, signal)


def highpass(signal, fs, cutoff_hz=2.0, order=4):
    """
    Zero-phase Butterworth high-pass filter.

    Parameters
    ----------
    signal     : 1D array
    fs         : sample rate (Hz)
    cutoff_hz  : -3 dB frequency (Hz)
    order      : filter order

    Returns
    -------
    1D array of the same length as *signal*
    """
    nyq = 0.5 * fs
    b, a = butter(order, cutoff_hz / nyq, btype="high")
    return filtfilt(b, a, signal)


# ---------------------------------------------------------------------------
# Rolling standard deviation
# ---------------------------------------------------------------------------

def rolling_std(signal, fs, window_s=1.0):
    """
    Compute a causal rolling standard deviation using a rectangular window.

    Parameters
    ----------
    signal   : 1D array
    fs       : sample rate (Hz)
    window_s : window length in seconds

    Returns
    -------
    1D array of the same length as *signal* (first window_s seconds are NaN)
    """
    n   = max(int(round(window_s * fs)), 2)
    out = np.full(len(signal), np.nan)
    for i in range(n - 1, len(signal)):
        out[i] = np.std(signal[i - n + 1 : i + 1], ddof=1)
    return out


# ---------------------------------------------------------------------------
# Spectrogram
# ---------------------------------------------------------------------------

def compute_spectrogram(signal, fs, window_s=2.0, overlap_frac=0.75,
                        fmin=0.0, fmax=15.0):
    """
    Compute a power spectrogram (dB) clipped to a frequency band.

    Parameters
    ----------
    signal       : 1D array
    fs           : sample rate (Hz)
    window_s     : FFT window length (s)
    overlap_frac : fractional overlap between successive windows
    fmin / fmax  : frequency limits for the returned slice (Hz)

    Returns
    -------
    freqs  : 1D array  — frequency axis (Hz)
    times  : 1D array  — time axis (s)
    Sxx_db : 2D array  — power in dB, shape (len(freqs), len(times))
    """
    nperseg = max(int(window_s * fs), 32)
    noverlap = int(nperseg * overlap_frac)
    freqs, times, Sxx = _spectrogram(signal, fs=fs,
                                     nperseg=nperseg, noverlap=noverlap,
                                     nfft=nperseg, scaling="density")
    # Clip to requested band
    mask  = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[mask]
    Sxx   = Sxx[mask, :]
    # Convert to dB (floor at -60 dB relative to max)
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    Sxx_db = np.clip(Sxx_db, Sxx_db.max() - 60, None)
    return freqs, times, Sxx_db

