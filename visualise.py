"""
visualise.py  —  plotting helpers for the artefact detection pipeline
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from signal_helpers import lowpass, highpass, compute_spectrogram


# Dutch display name and signal per artefact key
ARTEFACT_INFO = {
    "cal_abp"      : ("Calibratie",      "ABP"),
    "flush_abp"    : ("Flush",           "ABP"),
    "infuus_abp"   : ("Infuus",          "ABP"),
    "cal_cvp"      : ("Calibratie",      "CVP"),
    "flush_cvp"    : ("Flush",           "CVP"),
    "infuus_cvp"   : ("Infuus",          "CVP"),
    "slinger_abp"  : ("Slinger",         "ABP"),
    "slinger_cvp"  : ("Slinger",         "CVP"),
    "gasbubble_abp": ("Gasbel",          "ABP"),
    "gasbubble_cvp": ("Gasbel",          "CVP"),
    "transducer"   : ("Transducer hoog", "ABP"),
}

# Colour per artefact key (used by plot_results)
ARTEFACT_COLOURS = {
    "cal_abp"      : "#1f77b4",   # blue
    "flush_abp"    : "#ff7f0e",   # orange
    "infuus_abp"   : "#8c564b",   # brown
    "cal_cvp"      : "#17becf",   # teal
    "flush_cvp"    : "#ffbb78",   # light orange
    "infuus_cvp"   : "#c49c94",   # light brown
    "slinger_abp"  : "#d62728",   # red
    "slinger_cvp"  : "#e57373",   # light red
    "gasbubble_abp": "#2ca02c",   # green
    "gasbubble_cvp": "#98df8a",   # light green
    "transducer"   : "#9467bd",   # purple
}

# Which subplot column(s) each artefact key is drawn on
ARTEFACT_AXES = {
    "cal_abp"      : ("abp",),
    "flush_abp"    : ("abp",),
    "infuus_abp"   : ("abp",),
    "cal_cvp"      : ("cvp",),
    "flush_cvp"    : ("cvp",),
    "infuus_cvp"   : ("cvp",),
    "slinger_abp"  : ("abp",),
    "slinger_cvp"  : ("cvp",),
    "gasbubble_abp": ("abp",),
    "gasbubble_cvp": ("cvp",),
    "transducer"   : ("abp", "cvp"),
}



def plot_results(t, ABP, CVP, fs, artefacts,
                 slinger_diag_abp, slinger_diag_cvp,
                 path, lp_cutoff, hp_cutoff, spec_fmax):
    """
    Four-row diagnostic figure:

      Row 0 — filtered signals (raw / LP / HP)
      Row 1 — spectrograms
      Row 2 — detected artefacts overlaid on the raw signal
      Row 3 — slinger HF energy diagnostic (log energy, smoothed, threshold)
    """
    n = len(ABP)

    # ── pre-compute filtered signals and spectrograms ─────────────────────────
    lp_ABP = lowpass(ABP,  fs, cutoff_hz=lp_cutoff)
    hp_ABP = highpass(ABP, fs, cutoff_hz=hp_cutoff)
    lp_CVP = lowpass(CVP,  fs, cutoff_hz=lp_cutoff)
    hp_CVP = highpass(CVP, fs, cutoff_hz=hp_cutoff)

    f_ABP, ts_ABP, S_ABP = compute_spectrogram(ABP, fs, fmax=spec_fmax)
    f_CVP, ts_CVP, S_CVP = compute_spectrogram(CVP, fs, fmax=spec_fmax)

    # ── figure layout ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 2, figsize=(14, 14), sharex="row")
    fig.suptitle(os.path.basename(path) if isinstance(path, str) else "recording",
                 fontsize=11)

    row_titles = ["Filtered signal", "Spectrogram", "Detected artefacts",
                  "Slinger HF energy"]
    for row, title in enumerate(row_titles):
        axes[row, 0].set_title(f"ABP — {title}", fontsize=10)
        axes[row, 1].set_title(f"CVP — {title}", fontsize=10)

    # ── row 0 : filtered signals ──────────────────────────────────────────────
    for col, (raw, lp, hp) in enumerate([
        (ABP, lp_ABP, hp_ABP),
        (CVP, lp_CVP, hp_CVP),
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
        (f_ABP, ts_ABP, S_ABP),
        (f_CVP, ts_CVP, S_CVP),
    ]):
        ax = axes[1, col]
        pcm = ax.pcolormesh(ts, f_s, S, shading="gouraud", cmap="inferno")
        plt.colorbar(pcm, ax=ax, label="Power (dB)", pad=0.02)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(f_s[0], f_s[-1])
        ax.set_xlim(ts[0], ts[-1])

    # ── row 2 : detected artefacts ────────────────────────────────────────────
    ax_ABP2 = axes[2, 0]
    ax_CVP2 = axes[2, 1]

    ax_ABP2.plot(t, ABP, color="0.3", lw=0.8)
    ax_ABP2.set_ylabel("mmHg")
    ax_ABP2.set_xlabel("Time (s)")
    ax_ABP2.set_xlim(t[0], t[-1])

    ax_CVP2.plot(t, CVP, color="0.3", lw=0.8)
    ax_CVP2.set_ylabel("mmHg")
    ax_CVP2.set_xlabel("Time (s)")
    ax_CVP2.set_xlim(t[0], t[-1])

    artefact_axes_map = {"abp": ax_ABP2, "cvp": ax_CVP2}
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
        ax_ABP2.legend(handles=legend_patches, loc="upper right", fontsize=7)

    # ── row 3 : slinger HF diagnostic ────────────────────────────────────────
    for col, (diag, slinger_key) in enumerate([
        (slinger_diag_abp, "slinger_abp"),
        (slinger_diag_cvp, "slinger_cvp"),
    ]):
        ax = axes[3, col]
        colour = ARTEFACT_COLOURS[slinger_key]
        if diag is not None:
            ax.plot(diag["ts"], diag["hf_ratio"],  color="0.75", lw=0.8,
                    label="z residual")
            ax.plot(diag["ts"], diag["hf_smooth"], color="C0",   lw=1.2,
                    label="Detection score")
            ax.axhline(diag["threshold"], color="C3", lw=1.0, ls="--",
                       label="Threshold")
            for s, e in artefacts[slinger_key]:
                ax.axvspan(t[s], t[min(e, n - 1)], color=colour, alpha=0.35, lw=0)
            ax.set_ylabel("Detection score")
            ax.set_xlim(diag["ts"][0], diag["ts"][-1])
            ax.legend(loc="upper right", fontsize=7, ncol=3)
        else:
            ax.text(0.5, 0.5, "no slinger run", transform=ax.transAxes,
                    ha="center", va="center", color="0.5")

    plt.tight_layout()
    plt.savefig("Plots.png", dpi=150, bbox_inches="tight")
    plt.show()
