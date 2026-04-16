"""
visualise.py  —  plotting helpers for the artefact detection pipeline
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from signal_helpers import lowpass, highpass, compute_spectrogram


ARTEFACT_COLOURS = {
    "cal_abp"    : "#1f77b4",   # blue
    "flush_abp"  : "#ff7f0e",   # orange
    "infuus_abp" : "#8c564b",   # brown
    "cal_cvp"    : "#17becf",   # teal
    "flush_cvp"  : "#ffbb78",   # light orange
    "infuus_cvp" : "#c49c94",   # light brown
    "slinger_abp": "#d62728",   # red
    "slinger_cvp": "#e57373",   # light red
    "gasbubble"  : "#2ca02c",   # green
    "transducer" : "#9467bd",   # purple
}

# Which subplot column(s) each artefact key is drawn on
ARTEFACT_AXES = {
    "cal_abp"    : ("abp",),
    "flush_abp"  : ("abp",),
    "infuus_abp" : ("abp",),
    "cal_cvp"    : ("cvp",),
    "flush_cvp"  : ("cvp",),
    "infuus_cvp" : ("cvp",),
    "slinger_abp": ("abp",),
    "slinger_cvp": ("cvp",),
    "gasbubble"  : ("abp", "cvp"),
    "transducer" : ("abp", "cvp"),
}


def plot_results(t, abp, cvp, fs, artefacts,
                 slinger_diag_abp, slinger_diag_cvp,
                 path, lp_cutoff, hp_cutoff, spec_fmax):
    """
    Render the four-row diagnostic figure:

      Row 0 — filtered signals (raw / LP / HP)
      Row 1 — spectrograms
      Row 2 — detected artefacts overlaid on the raw signal
      Row 3 — slinger HF energy diagnostic (log energy, smoothed, threshold)

    Parameters
    ----------
    t, abp, cvp      : signal arrays
    fs               : sample rate (Hz)
    artefacts        : dict of {key: [(start, end), ...]}
    slinger_diag_abp / slinger_diag_cvp : diagnostic dicts from detect_slinger,
                       or None if slinger was never run
    path             : file path (used for figure title)
    lp_cutoff        : low-pass cutoff (Hz) — for axis labels
    hp_cutoff        : high-pass cutoff (Hz) — for axis labels
    spec_fmax        : upper frequency limit for spectrograms (Hz)
    """
    n = len(abp)

    # ── pre-compute filtered signals and spectrograms ─────────────────────────
    lp_abp = lowpass(abp,  fs, cutoff_hz=lp_cutoff)
    hp_abp = highpass(abp, fs, cutoff_hz=hp_cutoff)
    lp_cvp = lowpass(cvp,  fs, cutoff_hz=lp_cutoff)
    hp_cvp = highpass(cvp, fs, cutoff_hz=hp_cutoff)

    f_abp, ts_abp, S_abp = compute_spectrogram(abp, fs, fmax=spec_fmax)
    f_cvp, ts_cvp, S_cvp = compute_spectrogram(cvp, fs, fmax=spec_fmax)

    # ── figure layout ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 2, figsize=(14, 14), sharex="row")
    fig.suptitle(os.path.basename(path), fontsize=11)

    row_titles = ["Filtered signal", "Spectrogram", "Detected artefacts",
                  "Slinger HF energy"]
    for row, title in enumerate(row_titles):
        axes[row, 0].set_title(f"ABP — {title}", fontsize=10)
        axes[row, 1].set_title(f"CVP — {title}", fontsize=10)

    # ── row 0 : filtered signals ──────────────────────────────────────────────
    for col, (raw, lp, hp, label) in enumerate([
        (abp, lp_abp, hp_abp, "ABP"),
        (cvp, lp_cvp, hp_cvp, "CVP"),
    ]):
        ax = axes[0, col]
        ax.plot(t, raw, color="0.75", lw=0.7, label="Raw")
        ax.plot(t, lp,  color="C0",   lw=1.2, label=f"LP ≤{lp_cutoff} Hz")
        ax.plot(t, hp,  color="C1",   lw=1.0, label=f"HP ≥{hp_cutoff} Hz")
        ax.set_ylabel("mmHg")
        ax.set_xlim(t[0], t[-1])
        ax.legend(loc="upper right", fontsize=7, ncol=3)

    # ── row 1 : spectrograms ──────────────────────────────────────────────────
    for col, (f_s, ts, S) in enumerate([
        (f_abp, ts_abp, S_abp),
        (f_cvp, ts_cvp, S_cvp),
    ]):
        ax = axes[1, col]
        pcm = ax.pcolormesh(ts, f_s, S, shading="gouraud", cmap="inferno")
        plt.colorbar(pcm, ax=ax, label="Power (dB)", pad=0.02)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(f_s[0], f_s[-1])
        ax.set_xlim(ts[0], ts[-1])

    # ── row 2 : detected artefacts ────────────────────────────────────────────
    ax_abp2 = axes[2, 0]
    ax_cvp2 = axes[2, 1]

    ax_abp2.plot(t, abp, color="0.3", lw=0.8)
    ax_abp2.set_ylabel("mmHg")
    ax_abp2.set_xlabel("Time (s)")
    ax_abp2.set_xlim(t[0], t[-1])

    ax_cvp2.plot(t, cvp, color="0.3", lw=0.8)
    ax_cvp2.set_ylabel("mmHg")
    ax_cvp2.set_xlabel("Time (s)")
    ax_cvp2.set_xlim(t[0], t[-1])

    artefact_axes_map = {"abp": ax_abp2, "cvp": ax_cvp2}
    legend_patches = []
    for key, periods in artefacts.items():
        if not periods:
            continue
        colour = ARTEFACT_COLOURS[key]
        for ax_name in ARTEFACT_AXES[key]:
            ax = artefact_axes_map[ax_name]
            for s, e in periods:
                ax.axvspan(t[s], t[min(e, n - 1)], color=colour, alpha=0.35, lw=0)
        legend_patches.append(mpatches.Patch(color=colour, alpha=0.6, label=key))

    if legend_patches:
        ax_abp2.legend(handles=legend_patches, loc="upper right", fontsize=7)

    # ── row 3 : slinger HF diagnostic ────────────────────────────────────────
    for col, (diag, slinger_key) in enumerate([
        (slinger_diag_abp, "slinger_abp"),
        (slinger_diag_cvp, "slinger_cvp"),
    ]):
        ax = axes[3, col]
        colour = ARTEFACT_COLOURS[slinger_key]
        if diag is not None:
            ax.plot(diag["ts"], diag["hf_log"],    color="0.75", lw=0.8,
                    label="HF energy (log)")
            ax.plot(diag["ts"], diag["hf_smooth"], color="C0",   lw=1.2,
                    label="Smoothed")
            ax.axhline(diag["threshold"], color="C3", lw=1.0, ls="--",
                       label="Threshold")
            for s, e in artefacts[slinger_key]:
                ax.axvspan(t[s], t[min(e, n - 1)], color=colour, alpha=0.35, lw=0)
            ax.set_ylabel("log(1 + HF power)")
            ax.set_xlim(diag["ts"][0], diag["ts"][-1])
            ax.legend(loc="upper right", fontsize=7, ncol=3)
        else:
            ax.text(0.5, 0.5, "no slinger run", transform=ax.transAxes,
                    ha="center", va="center", color="0.5")

    plt.tight_layout()
    plt.show()
