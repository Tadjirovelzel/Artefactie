"""
startagain.py  —  artefact detection decision tree

Decision tree (lazy metric computation):

    START
     │
     ├─ Compute P_total ──► Oscillations collapsed?
     │                          YES ──► Compute LP ──► LP near zero?
     │                                                     YES ──► CALIBRATION
     │                                                     NO  ──► FLUSH
     │                          NO ──┐
     │                               │
     ├─ Compute ringing_ratio ───────► Ringing above 2×cardiac?
     │                          YES ──► SLINGER
     │                          NO ──┐
     │                               │
     ├─ Compute HP_env + P_cardiac ──► Beats damped?
     │                          YES ──► GASBUBBLE
     │                          NO ──┐
     │                               │
     ├─ Compute LP_ABP ──────────────► ABP baseline dropped?
     │                          YES ──► TRANSDUCER HOOG  (restart)
     │                          NO ──┐
     │                               │
     └─ Compute LP_CVP ──────────────► CVP baseline elevated? (ABP normal?)
                                YES ──► INFUUS OP CVD
                                NO  ──► END (no new artefacts)

After each detection the flagged period is excluded and the tree restarts
from the top.  Baselines are always recomputed from the remaining clean signal.
Transducer hoog is checked before infuus: a dropped ABP would satisfy the
infuus ABP guard (ABP ≤ threshold) and cause a false positive if both run
in the same pass.
"""

import tkinter as tk
from tkinter import filedialog
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from signal_helpers import load_signal, lowpass, highpass, compute_spectrogram
from detection import (
    detect_calibration_flush,
    detect_slinger,
    detect_gasbubble,
    detect_transducer_hoog,
    detect_infuus,
)

# ── parameters ────────────────────────────────────────────────────────────────
LP_CUTOFF = 0.5   # low-pass cutoff used for baseline tracking (Hz)
HP_CUTOFF = 1.0    # high-pass cutoff used for beat-amplitude tracking (Hz)
SPEC_FMAX = 15.0   # upper frequency limit for spectrogram features (Hz)
K_THRESH  = 3.0    # deviation from median in IQR units to call an outlier

# ── file picker ───────────────────────────────────────────────────────────────
root = tk.Tk()
root.withdraw()
path = filedialog.askopenfilename(
    title="Select recording (.xlsx)",
    filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
)
root.destroy()

if not path:
    raise SystemExit("No file selected.")

# ── load signal ───────────────────────────────────────────────────────────────
t, abp, cvp, fs = load_signal(path)
n = len(abp)

print(f"File    : {os.path.basename(path)}")
print(f"Duration: {t[-1]:.1f} s  |  fs: {fs:.0f} Hz  |  {n} samples")
print(f"ABP     : {abp.min():.0f} – {abp.max():.0f} mmHg")
print(f"CVP     : {cvp.min():.0f} – {cvp.max():.0f} mmHg")

# ── artefact registry ─────────────────────────────────────────────────────────
# Calibration and flush are tracked per channel (ABP and CVP can be calibrated
# independently).  All other artefact types affect both channels together.
artefacts = {
    "cal_abp"    : [],
    "flush_abp"  : [],
    "cal_cvp"    : [],
    "flush_cvp"  : [],
    "slinger_abp": [],
    "slinger_cvp": [],
    "gasbubble"  : [],
    "transducer" : [],
    "infuus"     : [],
}


def build_exclusion_mask(keys=None):
    """Return a boolean array that is True wherever any artefact was found.

    Parameters
    ----------
    keys : iterable of str, optional
        If supplied, only the listed artefact keys are included.
        Defaults to all keys.
    """
    mask = np.zeros(n, dtype=bool)
    for k, periods in artefacts.items():
        if keys is None or k in keys:
            for start, end in periods:
                mask[start:end] = True
    return mask


def register(key, new_periods):
    """Append newly detected periods to the registry."""
    artefacts[key].extend(new_periods)


# ── decision tree (iterative) ─────────────────────────────────────────────────
iteration = 0

while True:
    iteration += 1
    print(f"\n── iteration {iteration} "
          f"({'clean signal' if iteration == 1 else 'after exclusions'}) ──")

    # Snapshot *before* this pass.  A restart is triggered only when this count
    # grows — guarantees termination and prevents re-registering known periods.
    excl_before = build_exclusion_mask().sum()

    # Helper: register exactly ONE period (the earliest-starting) and restart.
    # Every step below calls _register_first; if it returns True the step found
    # something and the outer loop continues immediately.
    def _register_first(key, periods):
        """Register only the earliest-starting non-overlapping period.

        Periods that overlap with already-excluded samples are skipped so it is
        structurally impossible for any registered period to overlap with a
        previously registered one, regardless of what a detector returned.
        """
        if not periods:
            return False
        excl_now = build_exclusion_mask()
        non_overlapping = [p for p in periods if not excl_now[p[0]:p[1]].any()]
        if not non_overlapping:
            return False
        first = min(non_overlapping, key=lambda p: p[0])
        register(key, [first])
        print(f"  {key:12s}: registered  {first[0]/fs:.1f}–{first[1]/fs:.1f} s"
              f"  (detector found {len(periods)} total, "
              f"{len(non_overlapping)} non-overlapping, using earliest)")
        return True

    # ── step 1 ───────────────────────────────────────────────────────────────
    # Compute P_total from the spectrogram per channel.
    # If it collapses (beats absent), compute LP to distinguish:
    #   LP near zero  → pressure dropped to baseline  → CALIBRATION
    #   LP near max   → pressure clamped at ceiling   → FLUSH
    # ABP and CVP are detected independently; even so, only one period total
    # is registered per iteration so baselines are always recomputed cleanly.
    excl = build_exclusion_mask()
    # Per-channel scan masks: each channel only excludes its own previously
    # detected artefacts plus shared artefacts (gasbubble, transducer, infuus).
    # This prevents an ABP calibration from masking the CVP calibration detector
    # when both occur in the same time window (and vice-versa).
    _shared = ('gasbubble', 'transducer', 'infuus')
    excl_abp = build_exclusion_mask(('cal_abp', 'flush_abp', 'slinger_abp') + _shared)
    excl_cvp = build_exclusion_mask(('cal_cvp', 'flush_cvp', 'slinger_cvp') + _shared)
    try:
        cal_abp, flush_abp, cal_cvp, flush_cvp = detect_calibration_flush(
            abp, cvp, fs,
            excluded=excl,
            lp_cutoff=LP_CUTOFF,
            hp_cutoff=HP_CUTOFF,
            spec_fmax=SPEC_FMAX,
            k=K_THRESH,
            excluded_abp=excl_abp,
            excluded_cvp=excl_cvp,
        )
        # ABP and CVP are channel-specific, so register one period per channel
        # independently — both checked against their own scan mask so a
        # simultaneous cal on both channels is caught in the same iteration.
        for ch_candidates, ch_excl in [
            ([("cal_abp",  p) for p in cal_abp]  + [("flush_abp", p) for p in flush_abp],  excl_abp),
            ([("cal_cvp",  p) for p in cal_cvp]  + [("flush_cvp", p) for p in flush_cvp],  excl_cvp),
        ]:
            non_overlapping = [(k, p) for k, p in ch_candidates
                               if not ch_excl[p[0]:p[1]].any()]
            if non_overlapping:
                key, first = min(non_overlapping, key=lambda x: x[1][0])
                register(key, [first])
                print(f"  {key:12s}: registered  {first[0]/fs:.1f}–{first[1]/fs:.1f} s"
                      f"  ({len(non_overlapping)} non-overlapping, using earliest)")
    except NotImplementedError:
        print("  calibration/flush : not implemented — skipping")

    if build_exclusion_mask().sum() > excl_before:
        continue

    # ── step 2 ───────────────────────────────────────────────────────────────
    # Compute ringing_ratio = P_high / P_total where P_high is power above
    # 2 × dominant cardiac frequency.  A spike in ringing_ratio that exceeds
    # the signal's own baseline by K_THRESH × IQR indicates resonance.
    # ABP and CVP are detected independently; across both lists only the
    # earliest-starting non-overlapping period is registered per iteration.
    excl = build_exclusion_mask()
    try:
        slinger_abp_periods, slinger_cvp_periods = detect_slinger(
            abp, cvp, fs,
            excluded=excl,
            lp_cutoff=LP_CUTOFF,
            hp_cutoff=HP_CUTOFF,
            spec_fmax=SPEC_FMAX,
            k=K_THRESH,
        )
        candidates = (
            [("slinger_abp", p) for p in slinger_abp_periods] +
            [("slinger_cvp", p) for p in slinger_cvp_periods]
        )
        if candidates:
            excl_now = build_exclusion_mask()
            non_overlapping = [(key, p) for key, p in candidates
                               if not excl_now[p[0]:p[1]].any()]
            if non_overlapping:
                key, first = min(non_overlapping, key=lambda x: x[1][0])
                register(key, [first])
                print(f"  {key:12s}: registered  {first[0]/fs:.1f}-{first[1]/fs:.1f} s"
                      f"  (detector found {len(candidates)} total, "
                      f"{len(non_overlapping)} non-overlapping, using earliest)")
    except NotImplementedError:
        print("  slinger     : not implemented — skipping")

    if build_exclusion_mask().sum() > excl_before:
        continue

    # ── step 3 ───────────────────────────────────────────────────────────────
    # Compute HP_env (rolling amplitude of high-passed signal) and P_cardiac
    # (spectrogram power at the dominant cardiac frequency).
    # Both dropping together means the pulse is damped → GASBUBBLE.
    excl = build_exclusion_mask()
    try:
        gasbubble_periods = detect_gasbubble(
            abp, cvp, fs,
            excluded=excl,
            hp_cutoff=HP_CUTOFF,
            spec_fmax=SPEC_FMAX,
            k=K_THRESH,
        )
        _register_first("gasbubble", gasbubble_periods)
    except NotImplementedError:
        print("  gasbubble   : not implemented — skipping")

    if build_exclusion_mask().sum() > excl_before:
        continue

    # ── step 4 ───────────────────────────────────────────────────────────────
    # LP_ABP dropped while oscillations continue → TRANSDUCER HOOG.
    # Must be checked and restarted BEFORE infuus: a transducer drop lowers
    # ABP below its normal baseline, which would satisfy the infuus ABP guard
    # (ABP ≤ threshold) and produce a false infuus positive in the same pass.
    excl = build_exclusion_mask()
    try:
        transducer_periods = detect_transducer_hoog(
            abp, cvp, fs,
            excluded=excl,
            lp_cutoff=LP_CUTOFF,
            hp_cutoff=HP_CUTOFF,
            k=K_THRESH,
        )
        _register_first("transducer", transducer_periods)
    except NotImplementedError:
        print("  transducer  : not implemented — skipping")

    if build_exclusion_mask().sum() > excl_before:
        continue

    # ── step 5 ───────────────────────────────────────────────────────────────
    # LP_CVP elevated while LP_ABP stays normal → INFUUS OP CVD.
    # Uses a fresh exclusion mask so any transducer period found in step 4
    # (on a prior iteration) is already excluded when baselines are estimated.
    excl = build_exclusion_mask()
    try:
        infuus_periods = detect_infuus(
            abp, cvp, fs,
            excluded=excl,
            lp_cutoff=LP_CUTOFF,
            k=K_THRESH,
        )
        _register_first("infuus", infuus_periods)
    except NotImplementedError:
        print("  infuus      : not implemented — skipping")

    if build_exclusion_mask().sum() > excl_before:
        continue

    # ── nothing new found in this pass: tree is complete ─────────────────────
    print("  no new artefacts — done.")
    break

# ── summary ───────────────────────────────────────────────────────────────────
print("\n── summary ─────────────────────────────────────────────────────────")
any_found = False
for key, periods in artefacts.items():
    if periods:
        any_found = True
        spans = ", ".join(
            f"{t[s]:.1f}–{t[min(e, n - 1)]:.1f} s" for s, e in periods
        )
        print(f"  {key:12s}  {len(periods):2d} period(s):  {spans}")

if not any_found:
    print("  no artefacts detected.")

excl_samples = build_exclusion_mask().sum()
print(f"\n  excluded: {excl_samples / fs:.1f} s  "
      f"({excl_samples / n * 100:.1f}% of {n / fs:.1f} s total)")

# ── plot ──────────────────────────────────────────────────────────────────────
# Layout: 2 rows (ABP, CVP) × 3 columns (filtered signal | spectrogram | artefacts)
# Channel-specific artefacts (cal/flush) are only shaded on their own row.
# Shared artefacts (slinger, gasbubble, transducer, infuus) appear on both rows.

ARTEFACT_COLOURS = {
    "cal_abp"    : "#1f77b4",   # blue
    "flush_abp"  : "#ff7f0e",   # orange
    "cal_cvp"    : "#17becf",   # teal
    "flush_cvp"  : "#ffbb78",   # light orange
    "slinger_abp": "#d62728",   # red
    "slinger_cvp": "#e57373",   # light red
    "gasbubble"  : "#2ca02c",   # green
    "transducer" : "#9467bd",   # purple
    "infuus"     : "#8c564b",   # brown
}

ARTEFACT_AXES = {
    "cal_abp"    : ("abp",),
    "flush_abp"  : ("abp",),
    "cal_cvp"    : ("cvp",),
    "flush_cvp"  : ("cvp",),
    "slinger_abp": ("abp",),
    "slinger_cvp": ("cvp",),
    "gasbubble"  : ("abp", "cvp"),
    "transducer" : ("abp", "cvp"),
    "infuus"     : ("cvp",),
}

# ── pre-compute filtered signals and spectrograms ─────────────────────────────
lp_abp = lowpass(abp,  fs, cutoff_hz=LP_CUTOFF)
hp_abp = highpass(abp, fs, cutoff_hz=HP_CUTOFF)
lp_cvp = lowpass(cvp,  fs, cutoff_hz=LP_CUTOFF)
hp_cvp = highpass(cvp, fs, cutoff_hz=HP_CUTOFF)

f_abp, ts_abp, S_abp = compute_spectrogram(abp, fs, fmax=SPEC_FMAX)
f_cvp, ts_cvp, S_cvp = compute_spectrogram(cvp, fs, fmax=SPEC_FMAX)

dlp_abp = np.gradient(lp_abp, 1.0 / fs)
dhp_abp  = np.gradient(hp_abp, 1.0 / fs)
dlp_cvp  = np.gradient(lp_cvp, 1.0 / fs)
dhp_cvp  = np.gradient(hp_cvp, 1.0 / fs)

# ── figure ────────────────────────────────────────────────────────────────────
# Row 0: filtered signals  |  Row 1: spectrograms
# Row 2: detected artefacts|  Row 3: LP & HP derivatives
# Col 0: ABP               |  Col 1: CVP
fig, axes = plt.subplots(5, 2, figsize=(14, 22), sharex="row")
fig.suptitle(os.path.basename(path), fontsize=11)

row_titles = ["Filtered signal", "Spectrogram", "Detected artefacts",
              "d/dt Low-pass", "d/dt High-pass"]
for row, title in enumerate(row_titles):
    axes[row, 0].set_title(f"ABP — {title}", fontsize=10)
    axes[row, 1].set_title(f"CVP — {title}", fontsize=10)

# ── row 0: filtered signals ───────────────────────────────────────────────────
for col, (raw, lp, hp, label) in enumerate([
    (abp, lp_abp, hp_abp, "ABP"),
    (cvp, lp_cvp, hp_cvp, "CVP"),
]):
    ax = axes[0, col]
    ax.plot(t, raw, color="0.75", lw=0.7, label="Raw")
    ax.plot(t, lp,  color="C0",   lw=1.2, label=f"LP ≤{LP_CUTOFF} Hz")
    ax.plot(t, hp,  color="C1",   lw=1.0, label=f"HP ≥{HP_CUTOFF} Hz")
    ax.set_ylabel("mmHg")
    ax.set_xlim(t[0], t[-1])
    ax.legend(loc="upper right", fontsize=7, ncol=3)

# ── row 1: spectrograms ───────────────────────────────────────────────────────
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

# ── row 2: detected artefacts ─────────────────────────────────────────────────
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
    colour  = ARTEFACT_COLOURS[key]
    for ax_name in ARTEFACT_AXES[key]:
        ax = artefact_axes_map[ax_name]
        for s, e in periods:
            ax.axvspan(t[s], t[min(e, n - 1)], color=colour, alpha=0.35, lw=0)
    legend_patches.append(mpatches.Patch(color=colour, alpha=0.6, label=key))

# ── row 3: d/dt low-pass ─────────────────────────────────────────────────────
for col, dlp in enumerate([dlp_abp, dlp_cvp]):
    ax = axes[3, col]
    ax.plot(t, dlp, color="C0", lw=1.0, label=f"d/dt LP ≤{LP_CUTOFF} Hz")
    ax.axhline(0, color="0.6", lw=0.6, ls="--")
    ax.set_ylabel("mmHg/s")
    ax.set_xlim(t[0], t[-1])
    ax.legend(loc="upper right", fontsize=7)

# ── row 4: d/dt high-pass ────────────────────────────────────────────────────
for col, dhp in enumerate([dhp_abp, dhp_cvp]):
    ax = axes[4, col]
    ax.plot(t, dhp, color="C1", lw=0.8, label=f"d/dt HP ≥{HP_CUTOFF} Hz")
    ax.axhline(0, color="0.6", lw=0.6, ls="--")
    ax.set_ylabel("mmHg/s")
    ax.set_xlabel("Time (s)")
    ax.set_xlim(t[0], t[-1])
    ax.legend(loc="upper right", fontsize=7)

if legend_patches:
    fig.legend(handles=legend_patches, loc="lower center",
               fontsize=8, ncol=len(legend_patches),
               bbox_to_anchor=(0.5, 0.0))
    plt.subplots_adjust(bottom=0.05)

plt.tight_layout(rect=[0, 0.04 if legend_patches else 0, 1, 1])
plt.show()
