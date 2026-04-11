"""
signal_helpers.py
Helper functions for loading, filtering, and visualising pressure signals.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, spectrogram as _spectrogram
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_signal(path):
    """
    Load a pressure recording from an Excel file.

    Expected layout
    ---------------
    Row 0 : column headers  (Time, ABP, CVP)
    Row 1 : units           (s, mmHg, mmHg)
    Row 2+: numeric data

    Returns
    -------
    t   : 1D float array  — time relative to the first sample (s)
    abp : 1D float array  — arterial blood pressure (mmHg)
    cvp : 1D float array  — central venous pressure (mmHg)
    fs  : float           — sample rate (Hz), inferred from time column
    """
    df = pd.read_excel(path, header=0, skiprows=[1])
    t_abs = pd.to_numeric(df["Time"], errors="coerce").values
    abp   = pd.to_numeric(df["ABP"],  errors="coerce").values
    cvp   = pd.to_numeric(df["CVP"],  errors="coerce").values

    # Drop any NaN rows
    valid = ~(np.isnan(t_abs) | np.isnan(abp) | np.isnan(cvp))
    t_abs, abp, cvp = t_abs[valid], abp[valid], cvp[valid]

    t  = t_abs - t_abs[0]
    fs = 1.0 / np.median(np.diff(t_abs))
    return t, abp, cvp, fs


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


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_signal_overview(t, raw, lp, hp, std,
                         label, lp_cutoff, hp_cutoff, window_s,
                         axes):
    """
    Fill one row of axes with raw/LP/HP signals and rolling std.

    Parameters
    ----------
    t, raw, lp, hp, std : 1D arrays (same length)
    label               : channel name string, e.g. "ABP"
    lp_cutoff           : low-pass cutoff (Hz) — for legend
    hp_cutoff           : high-pass cutoff (Hz) — for legend
    window_s            : rolling-std window (s) — for legend
    axes                : sequence of 2 Axes — [signal_ax, std_ax]
    """
    ax_sig, ax_std = axes

    ax_sig.plot(t, raw, color="0.7", lw=0.8, label="Raw")
    ax_sig.plot(t, lp,  color="C0", lw=1.2,
                label=f"Low-pass  ≤{lp_cutoff} Hz")
    ax_sig.plot(t, hp,  color="C1", lw=1.0,
                label=f"High-pass ≥{hp_cutoff} Hz")
    ax_sig.set_ylabel(f"{label} (mmHg)")
    ax_sig.legend(loc="upper right", fontsize=7, ncol=3)
    ax_sig.set_xlim(t[0], t[-1])
    ax_sig.tick_params(labelbottom=False)

    ax_std.plot(t, std, color="C2", lw=1.0,
                label=f"Rolling σ ({window_s} s window)")
    ax_std.set_ylabel("σ (mmHg)")
    ax_std.legend(loc="upper right", fontsize=7)
    ax_std.set_xlim(t[0], t[-1])


def plot_spectrogram(freqs, times, Sxx_db, label, ax, fs):
    """
    Draw a filled spectrogram on *ax*.

    Parameters
    ----------
    freqs, times, Sxx_db : output of compute_spectrogram
    label                : channel name string
    ax                   : Axes to draw on
    fs                   : sample rate — used only to annotate Nyquist
    """
    pcm = ax.pcolormesh(times, freqs, Sxx_db,
                        shading="gouraud", cmap="inferno")
    plt.colorbar(pcm, ax=ax, label="Power (dB)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"{label} spectrogram")
    ax.set_ylim(freqs[0], freqs[-1])
    ax.set_xlim(times[0], times[-1])
