"""
readArtefacts.py — Data loading, artefact detection pipeline, and visualisation.

Loads ABP and CVP pressure signals from Excel files, runs the full artefact
detection pipeline (flush, calibration, gas bubble), and displays a 3×2
figure with time series, FFT spectra, and spectrograms.

A file-selector dialog opens on startup; closing a figure reopens the selector.
Written for KT3401 - Assignment Artefact Detection.
"""

from pathlib import Path
import tkinter as tk
from tkinter import filedialog

from readartefacts import read_artefacts
from visualisation import plot_results
from decisiontree import decision_tree
from detectie import (
    compute_fft,
    detect_peaks_abp,
    detect_artifacts,
    detect_infuus_cvp,
    redetect_peaks_clean,
    detect_gasbubble,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _get_project_root() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd()


DATA_PATH  = _get_project_root() / "data" / "KT3401_AFdata_2025"
FS         = 100    # sampling rate (Hz)
RESOLUTION = 1      # spectrogram window length (s)
FRANGE     = [0, 10]  # spectrogram frequency range of interest (Hz)


# ---------------------------------------------------------------------------
# Detection pipeline
# ---------------------------------------------------------------------------

def run_pipeline(filepath):
    """
    Run the full artefact detection pipeline on one file and display results.
    """
    # ------------------------------------------------
    # Load ABP and CVP signals
    # ------------------------------------------------
    filepath = Path(filepath)
    folder   = filepath.parent.name
    filename = filepath.name
    print(f"\nLoading: {folder}/{filename}")

    t, ABP, CVP = read_artefacts(filepath, FS)
    if t is None:
        raise ValueError("Time is None, indicating file loading failed. Check the file path and format.")

    # ------------------------------------------------
    # Initial analysis: FFT and peak detection to establish artefact thresholds
    # ------------------------------------------------

    # Compute FFT to find dominant cardiac frequency for each signal
    abp_fft_freqs, abp_fft_mags, abp_dominant_freq = compute_fft(ABP, FS, frange=(0.5, 10))
    cvp_fft_freqs, cvp_fft_mags, cvp_dominant_freq = compute_fft(CVP, FS, frange=(0.5, 10))

    # Initial systolic peak detection (used to establish artefact thresholds)
    abp_peaks, abp_dominant_freq, abp_prominence = detect_peaks_abp(ABP, FS)
    cvp_peaks, cvp_dominant_freq, cvp_prominence = detect_peaks_abp(CVP, FS, physiological_max=25)
    print(f"ABP — {abp_dominant_freq:.2f} Hz ({abp_dominant_freq * 60:.0f} bpm), "
          f"{len(abp_peaks)} peaks")
    print(f"CVP — {cvp_dominant_freq:.2f} Hz ({cvp_dominant_freq * 60:.0f} bpm), "
          f"{len(cvp_peaks)} peaks")
    
    # ------------------------------------------------
    # Go threw decision tree
    # ------------------------------------------------
    decision_tree(t, ABP, CVP)

    # Step 7 — Plot all results
    results = dict(
        abp_peaks=abp_peaks,         cvp_peaks=cvp_peaks,
        abp_dominant_freq=abp_dominant_freq, cvp_dominant_freq=cvp_dominant_freq,
        abp_fft_freqs=abp_fft_freqs, abp_fft_mags=abp_fft_mags,
        cvp_fft_freqs=cvp_fft_freqs, cvp_fft_mags=cvp_fft_mags,
    )
    plot_results(t, ABP, CVP, results, folder, filename)


# ---------------------------------------------------------------------------
# Entry point — interactive file selector
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    while True:
        filepath = filedialog.askopenfilename(
            title="Select a dataset",
            initialdir=DATA_PATH,
            filetypes=[("Excel files", "*.xlsx *.xls")],
        )
        if not filepath:
            print("No file selected — exiting.")
            break
        run_pipeline(filepath)
