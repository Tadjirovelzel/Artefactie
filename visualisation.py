"""
displays a 3×2 figure with time series, FFT spectra, and spectrograms.
Written for KT3401 - Assignment Artefact Detection.
"""
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FS         = 100    # sampling rate (Hz)
RESOLUTION = 1      # spectrogram window length (s)
FRANGE     = [0, 10]  # spectrogram frequency range of interest (Hz)


# ---------------------------------------------------------------------------
# Visualisation functions
# ---------------------------------------------------------------------------
def shade_artifacts(ax, t, flush_periods, cal_periods, gasbubble_periods,
                    infuus_periods=None):
    """ Shades artefact periods on a time-series axis."""
    
    infuus_periods = infuus_periods or [] # why is this necessary

    for start, end in flush_periods:
        ax.axvspan(t[start], t[min(end, len(t) - 1)], color="red",    alpha=0.2, label="Flush type artifact")
    for start, end in cal_periods:
        ax.axvspan(t[start], t[min(end, len(t) - 1)], color="blue",   alpha=0.2, label="Calibration artifact")
    for start, end in gasbubble_periods:
        ax.axvspan(t[start], t[min(end, len(t) - 1)], color="green",  alpha=0.2, label="Gasbubble artifact")
    for start, end in infuus_periods:
        ax.axvspan(t[start], t[min(end, len(t) - 1)], color="purple", alpha=0.2, label="Infuus artifact")

    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    ax.legend(unique.values(), unique.keys())


def plot_results(t, ABP, CVP, results, folder, filename):
    """
    Render the 3×2 analysis figure.
      Row 1 — time series with detected peaks and artefact shading
      Row 2 — FFT magnitude spectrum
      Row 3 — spectrogram
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
