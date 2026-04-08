"""
Loads ABP and CVP pressure signals from Excel files
Written for KT3401 - Assignment Artefact Detection.
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

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
# Data loading
# ---------------------------------------------------------------------------

def read_artefacts(filepath, fs):
    """
    Load ABP and CVP signals from an Excel file.

    The file is expected to have ABP in column 2 and CVP in column 3
    (0-indexed), with the first two rows containing metadata.
    """
    try:
        raw = pd.read_excel(filepath, sheet_name=0, header=None)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None, None

    data = raw.iloc[2:, :].to_numpy()
    ABP  = pd.to_numeric(data[:, 1], errors="coerce")
    CVP  = pd.to_numeric(data[:, 2], errors="coerce")
    t    = np.arange(1 / fs, len(ABP) / fs + 1 / fs, 1 / fs)
    return t, ABP, CVP


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def shade_artifacts(ax, t, flush_periods, cal_periods, gasbubble_periods,
                    infuus_periods=None):
    """
    Shade artefact periods on a time-series axis.

    Flush artefacts are shaded red, calibration artefacts blue, gas-bubble
    artefacts green, and infuus artefacts purple. Duplicate legend entries
    are automatically removed.

    Parameters
    ----------
    ax                : matplotlib Axes
    t                 : 1D array  time vector (s)
    flush_periods     : list of (start_idx, end_idx) tuples
    cal_periods       : list of (start_idx, end_idx) tuples
    gasbubble_periods : list of (start_idx, end_idx) tuples
    infuus_periods    : list of (start_idx, end_idx) tuples  (optional)
    """
    infuus_periods = infuus_periods or []

    for s, e in flush_periods:
        ax.axvspan(t[s], t[min(e, len(t) - 1)], color="red",    alpha=0.2, label="Flush type artifact")
    for s, e in cal_periods:
        ax.axvspan(t[s], t[min(e, len(t) - 1)], color="blue",   alpha=0.2, label="Calibration artifact")
    for s, e in gasbubble_periods:
        ax.axvspan(t[s], t[min(e, len(t) - 1)], color="green",  alpha=0.2, label="Gasbubble artifact")
    for s, e in infuus_periods:
        ax.axvspan(t[s], t[min(e, len(t) - 1)], color="purple", alpha=0.2, label="Infuus artifact")

    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for h, l in zip(handles, labels):
        unique.setdefault(l, h)
    ax.legend(unique.values(), unique.keys())


def plot_results(t, ABP, CVP, results, folder, filename):
    """
    Render the 3×2 analysis figure.

    Layout:
      Row 1 — ABP time series with detected peaks and artefact shading
              CVP time series with detected peaks and artefact shading
      Row 2 — ABP FFT magnitude spectrum
              CVP FFT magnitude spectrum
      Row 3 — ABP spectrogram 
              CVP spectrogram

    Parameters
    ----------
    t        : 1D array  time vector (s)
    ABP, CVP : 1D arrays pressure signals (mmHg)
    results  : dict      output of run_pipeline (keys documented there)
    folder   : str       parent folder name (used in figure title)
    filename : str       file name (used in figure title)
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f"{folder} — {filename}")

    # --- Row 1: Time series ---
    ax = axes[0, 0]
    ax.plot(t, ABP)
    ax.plot(t[results["abp_peaks"]], ABP[results["abp_peaks"]], "x", color="red",
            label=f"Peaks (n={len(results['abp_peaks'])})")
    ax.axhline(results["abp_flush_thr"], color="red",       linestyle=":", linewidth=0.8,
               label=f"Flush threshold ({results['abp_flush_thr']:.1f} mmHg)")
    ax.axhline(results["abp_cal_thr"],   color="blue",      linestyle=":", linewidth=0.8,
               label=f"Cal. threshold ({results['abp_cal_thr']:.1f} mmHg)")
    ax.axhline(results["abp_avg_sys"],   color="green",     linestyle=":", linewidth=0.8,
               label=f"Avg systolic ({results['abp_avg_sys']:.1f} mmHg)")
    ax.axhline(results["abp_avg_dia"],   color="darkgreen", linestyle=":", linewidth=0.8,
               label=f"Avg diastolic ({results['abp_avg_dia']:.1f} mmHg)")
    shade_artifacts(ax, t, results["abp_flush"], results["abp_cal"], results["abp_gasbubble"])
    ax.set_title("Arterial Blood Pressure (ABP)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ABP (mmHg)")

    ax = axes[0, 1]
    ax.plot(t, CVP, color="tab:orange", label="CVP")
    ax.plot(t, results["cvp_hp"], color="tab:gray", linewidth=0.8,
            alpha=0.7, label="CVP (high-pass filtered)")
    ax.plot(t[results["cvp_peaks"]], CVP[results["cvp_peaks"]], "x", color="red",
            label=f"Peaks (n={len(results['cvp_peaks'])})")
    ax.axhline(results["cvp_flush_thr"], color="red", linestyle=":", linewidth=0.8,
               label=f"Flush threshold ({results['cvp_flush_thr']:.1f} mmHg)")
    shade_artifacts(ax, t, results["cvp_flush"], [], [],
                    infuus_periods=results["cvp_infuus"])
    ax.set_title("Central Venous Pressure (CVP)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("CVP (mmHg)")

    # --- Row 2: FFT spectra ---
    ax = axes[1, 0]
    ax.plot(results["abp_fft_freqs"], results["abp_fft_mags"])
    ax.axvline(results["abp_dominant_freq"], color="red", linestyle="--",
               label=f"Dominant: {results['abp_dominant_freq']:.2f} Hz "
                     f"({results['abp_dominant_freq'] * 60:.0f} bpm)")
    ax.set_title("ABP Frequency Spectrum (FFT)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (mmHg)")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(results["cvp_fft_freqs"], results["cvp_fft_mags"], color="tab:orange")
    ax.axvline(results["cvp_dominant_freq"], color="red", linestyle="--",
               label=f"Dominant: {results['cvp_dominant_freq']:.2f} Hz "
                     f"({results['cvp_dominant_freq'] * 60:.0f} bpm)")
    ax.set_title("CVP Frequency Spectrum (FFT)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (mmHg)")
    ax.legend()

    # --- Row 3: Spectrograms ---
    abp_freqs, abp_times, abp_Sxx = spectrogram(
        ABP, FS, nperseg=int(RESOLUTION * FS), noverlap=0, nfft=int(FS * RESOLUTION))
    cvp_freqs, cvp_times, cvp_Sxx = spectrogram(
        CVP, FS, nperseg=int(RESOLUTION * FS), noverlap=0, nfft=int(FS * RESOLUTION))

    im_abp = axes[2, 0].pcolormesh(abp_times, abp_freqs, 10 * np.log10(abp_Sxx),
                                    shading="auto", cmap="viridis")
    axes[2, 0].set_ylim(FRANGE)
    axes[2, 0].set_title("ABP Spectrogram")
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 0].set_ylabel("Frequency (Hz)")
    fig.colorbar(im_abp, ax=axes[2, 0], label="Power (dB)")

    im_cvp = axes[2, 1].pcolormesh(cvp_times, cvp_freqs, 10 * np.log10(cvp_Sxx),
                                    shading="auto", cmap="viridis")
    axes[2, 1].set_ylim(FRANGE)
    axes[2, 1].set_title("CVP Spectrogram")
    axes[2, 1].set_xlabel("Time (s)")
    axes[2, 1].set_ylabel("Frequency (Hz)")
    fig.colorbar(im_cvp, ax=axes[2, 1], label="Power (dB)")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Detection pipeline
# ---------------------------------------------------------------------------

def run_pipeline(filepath):
    """
    Run the full artefact detection pipeline on one file and display results.

    Steps
    -----
    1. Load ABP and CVP signals.
    2. Compute FFT to find dominant cardiac frequency for each signal.
    3. Initial systolic peak detection (used to establish artefact thresholds).
    4. Detect flush and calibration artefacts (ABP only for calibration).
    5. Re-detect peaks on clean signal sections only.
    6. Detect gas-bubble artefacts using clean peaks.
    7. Plot all results.

    Parameters
    ----------
    filepath : str or Path  full path to the data file
    """
    filepath = Path(filepath)
    folder   = filepath.parent.name
    filename = filepath.name
    print(f"\nLoading: {folder}/{filename}")

    t, ABP, CVP = read_artefacts(filepath, FS)
    if t is None:
        raise ValueError("Time is None, indicating file loading failed. Check the file path and format.")

    # Step 2 — FFT
    abp_fft_freqs, abp_fft_mags, abp_dominant_freq = compute_fft(ABP, FS, frange=(0.5, 10))
    cvp_fft_freqs, cvp_fft_mags, cvp_dominant_freq = compute_fft(CVP, FS, frange=(0.5, 10))

    # Step 3 — Initial peak detection
    abp_peaks, abp_dominant_freq, abp_prominence = detect_peaks_abp(ABP, FS)
    cvp_peaks, cvp_dominant_freq, cvp_prominence = detect_peaks_abp(CVP, FS, physiological_max=25)
    print(f"ABP — {abp_dominant_freq:.2f} Hz ({abp_dominant_freq * 60:.0f} bpm), "
          f"{len(abp_peaks)} peaks")
    print(f"CVP — {cvp_dominant_freq:.2f} Hz ({cvp_dominant_freq * 60:.0f} bpm), "
          f"{len(cvp_peaks)} peaks")

    # Step 4 — Artefact detection (calibration for ABP only)
    abp_flush, abp_cal, abp_flush_thr, abp_cal_thr = detect_artifacts(
        ABP, FS, abp_dominant_freq, abp_peaks)
    _, _, cvp_flush_thr, _ = detect_artifacts(
        CVP, FS, cvp_dominant_freq, cvp_peaks, physiological_max=25)

    # CVP elevated periods are split into infuus vs flush by oscillation check
    cvp_infuus, cvp_flush, cvp_hp = detect_infuus_cvp(
        CVP, FS, cvp_dominant_freq, cvp_flush_thr)

    print(f"ABP — {len(abp_flush)} flush, {len(abp_cal)} calibration artefacts")
    print(f"CVP — {len(cvp_flush)} flush, {len(cvp_infuus)} infuus artefacts")

    # Step 5 — Re-detect peaks on clean sections
    abp_peaks = redetect_peaks_clean(ABP, FS, abp_dominant_freq, abp_flush + abp_cal)
    cvp_peaks = redetect_peaks_clean(CVP, FS, cvp_dominant_freq, cvp_flush + cvp_infuus,
                                     physiological_max=25)
    print(f"ABP — {len(abp_peaks)} peaks after re-detection")
    print(f"CVP — {len(cvp_peaks)} peaks after re-detection")

    # Step 6 — Gas-bubble detection (ABP only)
    abp_gasbubble, abp_avg_sys, abp_avg_dia = detect_gasbubble(
        ABP, FS, abp_dominant_freq, abp_peaks, artifact_periods=abp_flush + abp_cal)
    print(f"ABP — {len(abp_gasbubble)} gasbubble artefact(s)")

    # Step 7 — Plot
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
