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
    # Step 1 — Load ABP and CVP signals
    filepath = Path(filepath)
    folder   = filepath.parent.name
    filename = filepath.name
    print(f"\nLoading: {folder}/{filename}")

    t, ABP, CVP = read_artefacts(filepath, FS)
    if t is None:
        raise ValueError("Time is None, indicating file loading failed. Check the file path and format.")

    # Step 2 — Compute FFT to find dominant cardiac frequency for each signal
    abp_fft_freqs, abp_fft_mags, abp_dominant_freq = compute_fft(ABP, FS, frange=(0.5, 10))
    cvp_fft_freqs, cvp_fft_mags, cvp_dominant_freq = compute_fft(CVP, FS, frange=(0.5, 10))

    # Step 3 — Initial systolic peak detection (used to establish artefact thresholds)
    abp_peaks, abp_dominant_freq, abp_prominence = detect_peaks_abp(ABP, FS)
    cvp_peaks, cvp_dominant_freq, cvp_prominence = detect_peaks_abp(CVP, FS, physiological_max=25)
    print(f"ABP — {abp_dominant_freq:.2f} Hz ({abp_dominant_freq * 60:.0f} bpm), "
          f"{len(abp_peaks)} peaks")
    print(f"CVP — {cvp_dominant_freq:.2f} Hz ({cvp_dominant_freq * 60:.0f} bpm), "
          f"{len(cvp_peaks)} peaks")
    
    
    ############################################
    # vanaf hier zou de keuzeboom moeten komen
    ############################################

    # Step 4 — Detect flush and calibration artefacts (ABP only for calibration)
    abp_flush, abp_cal, abp_flush_thr, abp_cal_thr = detect_artifacts(
        ABP, FS, abp_dominant_freq, abp_peaks)
    _, _, cvp_flush_thr, _ = detect_artifacts(
        CVP, FS, cvp_dominant_freq, cvp_peaks, physiological_max=25)

    # CVP elevated periods are split into infuus vs flush by oscillation check
    cvp_infuus, cvp_flush, cvp_hp = detect_infuus_cvp(
        CVP, FS, cvp_dominant_freq, cvp_flush_thr)

    print(f"ABP — {len(abp_flush)} flush, {len(abp_cal)} calibration artefacts")
    print(f"CVP — {len(cvp_flush)} flush, {len(cvp_infuus)} infuus artefacts")

    # Step 5 — Re-detect peaks on clean signal sections only
    abp_peaks = redetect_peaks_clean(ABP, FS, abp_dominant_freq, abp_flush + abp_cal)
    cvp_peaks = redetect_peaks_clean(CVP, FS, cvp_dominant_freq, cvp_flush + cvp_infuus,
                                     physiological_max=25)
    print(f"ABP — {len(abp_peaks)} peaks after re-detection")
    print(f"CVP — {len(cvp_peaks)} peaks after re-detection")

    # Step 6 — Detect gas-bubble artefacts using clean peaks
    abp_gasbubble, abp_avg_sys, abp_avg_dia = detect_gasbubble(
        ABP, FS, abp_dominant_freq, abp_peaks, artifact_periods=abp_flush + abp_cal)
    print(f"ABP — {len(abp_gasbubble)} gasbubble artefact(s)")

    # Step 7 — Plot all results
    results = dict(
        abp_peaks=abp_peaks,         cvp_peaks=cvp_peaks,
        abp_dominant_freq=abp_dominant_freq, cvp_dominant_freq=cvp_dominant_freq,
        abp_fft_freqs=abp_fft_freqs, abp_fft_mags=abp_fft_mags,
        cvp_fft_freqs=cvp_fft_freqs, cvp_fft_mags=cvp_fft_mags,
        abp_flush=abp_flush,         abp_cal=abp_cal,
        abp_flush_thr=abp_flush_thr, abp_cal_thr=abp_cal_thr,
        cvp_flush=cvp_flush,         cvp_flush_thr=cvp_flush_thr,
        cvp_infuus=cvp_infuus,       cvp_hp=cvp_hp,
        abp_gasbubble=abp_gasbubble,
        abp_avg_sys=abp_avg_sys,     abp_avg_dia=abp_avg_dia,
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
