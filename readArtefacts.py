'''
Model with function to load artefact data from Excel .xls and .xlsx files and splits 
data in separate vectors to process. Written for KT3401 - Assignment Artefact Detection

'''
# #%% Clear system
# from IPython import get_ipython
# # Clear all variables (IPython/Jupyter)
# get_ipython().magic('reset -sf')
# import matplotlib.pyplot as plt
# # Close all figures
# plt.close('all')
# import os
# # Clear the console
# os.system('cls' if os.name == 'nt' else 'clear')e

#%% Import modules
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from detectie import compute_fft, detect_peaks_abp, detect_artifacts

def get_project_root() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd()

project_root = get_project_root()
data_path = project_root / "data" / "KT3401_AFdata_2025"
fs = 100
resolution = 1    # seconds per spectrogram window
frange = [0, 10]  # frequency range of interest (Hz)


def read_Artefacts(filepath, fs):
    """
    Inputs:
    filepath: full path to the Excel file
    fs: sampling rate (in Hz)

    Outputs:
    t: time vector based on length of signal
    ABP, CVP: arterial and venous pressure vectors
    """
    try:
        raw = pd.read_excel(filepath, sheet_name=0, header=None)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None, None

    raw = raw.iloc[2:, :]
    data = raw.to_numpy()
    ABP = pd.to_numeric(data[:, 1], errors="coerce")
    CVP = pd.to_numeric(data[:, 2], errors="coerce")
    t = np.arange(1 / fs, len(ABP) / fs + 1 / fs, 1 / fs)
    return t, ABP, CVP


def run_pipeline(filepath):
    filepath = Path(filepath)
    folder = filepath.parent.name
    filename = filepath.name
    print(f"\nLoading: {folder}/{filename}")

    t, ABP, CVP = read_Artefacts(filepath, fs)
    if t is None:
        return

    # FFT
    abp_fft_freqs, abp_fft_mags, abp_dominant_freq = compute_fft(ABP, fs, frange=(0.5, 10))
    cvp_fft_freqs, cvp_fft_mags, cvp_dominant_freq = compute_fft(CVP, fs, frange=(0.5, 10))

    # Peak detection
    abp_peaks, abp_dominant_freq, abp_prominence = detect_peaks_abp(ABP, fs)
    cvp_peaks, cvp_dominant_freq, cvp_prominence = detect_peaks_abp(CVP, fs)
    print(f"ABP — Dominant: {abp_dominant_freq:.2f} Hz ({abp_dominant_freq * 60:.0f} bpm), Peaks: {len(abp_peaks)}, Prominence threshold: {abp_prominence:.2f}")
    print(f"CVP — Dominant: {cvp_dominant_freq:.2f} Hz ({cvp_dominant_freq * 60:.0f} bpm), Peaks: {len(cvp_peaks)}, Prominence threshold: {cvp_prominence:.2f}")

    # Artifact detection
    abp_flush, abp_cal, abp_flush_thr, abp_cal_thr = detect_artifacts(ABP, fs, abp_dominant_freq, abp_peaks)
    cvp_flush, cvp_cal, cvp_flush_thr, cvp_cal_thr = detect_artifacts(CVP, fs, cvp_dominant_freq, cvp_peaks)
    print(f"ABP — Flush artifacts: {len(abp_flush)}, Calibration artifacts: {len(abp_cal)}")
    print(f"CVP — Flush artifacts: {len(cvp_flush)}, Calibration artifacts: {len(cvp_cal)}")

    # Spectrograms
    abp_freqs, abp_times, abp_Sxx = spectrogram(ABP, fs, nperseg=int(resolution * fs), noverlap=0, nfft=int(fs * resolution))
    cvp_freqs, cvp_times, cvp_Sxx = spectrogram(CVP, fs, nperseg=int(resolution * fs), noverlap=0, nfft=int(fs * resolution))
    abp_Sxx_db = 10 * np.log10(abp_Sxx)
    cvp_Sxx_db = 10 * np.log10(cvp_Sxx)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f"{folder} — {filename}")

    def shade_artifacts(ax, t, flush_periods, cal_periods):
        for s, e in flush_periods:
            ax.axvspan(t[s], t[min(e, len(t) - 1)], color="red", alpha=0.2,
                       label="Flush type artifact")
        for s, e in cal_periods:
            ax.axvspan(t[s], t[min(e, len(t) - 1)], color="blue", alpha=0.2,
                       label="Calibration artifact")
        # Deduplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, labels):
            seen.setdefault(l, h)
        ax.legend(seen.values(), seen.keys())

    # Row 1: Time series
    axes[0, 0].plot(t, ABP)
    axes[0, 0].plot(t[abp_peaks], ABP[abp_peaks], "x", color="red", label=f"Peaks (n={len(abp_peaks)})")
    axes[0, 0].axhline(abp_flush_thr, color="red", linestyle=":", linewidth=0.8, label=f"Flush threshold ({abp_flush_thr:.1f} mmHg)")
    axes[0, 0].axhline(abp_cal_thr, color="blue", linestyle=":", linewidth=0.8, label=f"Cal. threshold 0.7× avg peak ({abp_cal_thr:.1f} mmHg)")
    shade_artifacts(axes[0, 0], t, abp_flush, abp_cal)
    axes[0, 0].set_title("Arterial Blood Pressure (ABP)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("ABP (mmHg)")

    axes[0, 1].plot(t, CVP, color="tab:orange")
    axes[0, 1].plot(t[cvp_peaks], CVP[cvp_peaks], "x", color="red", label=f"Peaks (n={len(cvp_peaks)})")
    axes[0, 1].axhline(cvp_flush_thr, color="red", linestyle=":", linewidth=0.8, label=f"Flush threshold ({cvp_flush_thr:.1f} mmHg)")
    axes[0, 1].axhline(cvp_cal_thr, color="blue", linestyle=":", linewidth=0.8, label=f"Cal. threshold 0.7× avg peak ({cvp_cal_thr:.1f} mmHg)")
    shade_artifacts(axes[0, 1], t, cvp_flush, cvp_cal)
    axes[0, 1].set_title("Central Venous Pressure (CVP)")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("CVP (mmHg)")

    # Row 2: FFT
    axes[1, 0].plot(abp_fft_freqs, abp_fft_mags)
    axes[1, 0].axvline(abp_dominant_freq, color="red", linestyle="--", label=f"Dominant: {abp_dominant_freq:.2f} Hz ({abp_dominant_freq * 60:.0f} bpm)")
    axes[1, 0].set_title("ABP Frequency Spectrum (FFT)")
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("Magnitude (mmHg)")
    axes[1, 0].legend()

    axes[1, 1].plot(cvp_fft_freqs, cvp_fft_mags, color="tab:orange")
    axes[1, 1].axvline(cvp_dominant_freq, color="red", linestyle="--", label=f"Dominant: {cvp_dominant_freq:.2f} Hz ({cvp_dominant_freq * 60:.0f} bpm)")
    axes[1, 1].set_title("CVP Frequency Spectrum (FFT)")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Magnitude (mmHg)")
    axes[1, 1].legend()

    # Row 3: Spectrograms
    im_abp = axes[2, 0].pcolormesh(abp_times, abp_freqs, abp_Sxx_db, shading="auto", cmap="viridis")
    axes[2, 0].set_ylim(frange)
    axes[2, 0].set_title("ABP Spectrogram")
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 0].set_ylabel("Frequency (Hz)")
    fig.colorbar(im_abp, ax=axes[2, 0], label="Power (dB)")

    im_cvp = axes[2, 1].pcolormesh(cvp_times, cvp_freqs, cvp_Sxx_db, shading="auto", cmap="viridis")
    axes[2, 1].set_ylim(frange)
    axes[2, 1].set_title("CVP Spectrogram")
    axes[2, 1].set_xlabel("Time (s)")
    axes[2, 1].set_ylabel("Frequency (Hz)")
    fig.colorbar(im_cvp, ax=axes[2, 1], label="Power (dB)")

    plt.tight_layout()
    plt.show()


#%% Main loop — file selector
root = tk.Tk()
root.withdraw()  # hide the empty tkinter window

while True:
    filepath = filedialog.askopenfilename(
        title="Select a dataset",
        initialdir=data_path,
        filetypes=[("Excel files", "*.xlsx *.xls")]
    )
    if not filepath:
        print("No file selected — exiting.")
        break
    run_pipeline(filepath)
