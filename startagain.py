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
     └─ Compute LP (per channel) ────► Baseline shifted?
                                YES ──► ABP shifted? ──► YES ──► TRANSDUCER HOOG
                                                         NO  ──► INFUUS OP CVD
                                NO  ──► END (no new artefacts)

After each detection the flagged period is excluded and the tree restarts
from the top.  Baselines are always recomputed from the remaining clean signal.
"""

import tkinter as tk
from tkinter import filedialog
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from signal_helpers import load_signal
from detection import (
    detect_calibration_flush,
    detect_slinger,
    detect_gasbubble,
    detect_transducer_hoog,
    detect_infuus,
)

# ── parameters ────────────────────────────────────────────────────────────────
LP_CUTOFF = 0.25   # low-pass cutoff used for baseline tracking (Hz)
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
    "slinger"    : [],
    "gasbubble"  : [],
    "transducer" : [],
    "infuus"     : [],
}


def build_exclusion_mask():
    """Return a boolean array that is True wherever any artefact was found."""
    mask = np.zeros(n, dtype=bool)
    for periods in artefacts.values():
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

    found_this_pass = False
    excl = build_exclusion_mask()

    # ── step 1 ───────────────────────────────────────────────────────────────
    # Compute P_total from the spectrogram per channel.
    # If it collapses (beats absent), compute LP to distinguish:
    #   LP near zero  → pressure dropped to baseline  → CALIBRATION
    #   LP near max   → pressure clamped at ceiling   → FLUSH
    # ABP and CVP are detected independently.
    try:
        cal_abp, flush_abp, cal_cvp, flush_cvp = detect_calibration_flush(
            abp, cvp, fs,
            excluded=excl,
            lp_cutoff=LP_CUTOFF,
            spec_fmax=SPEC_FMAX,
            k=K_THRESH,
        )
        if cal_abp:
            print(f"  cal  ABP    : {len(cal_abp)} period(s) found")
            register("cal_abp", cal_abp)
            found_this_pass = True
        if flush_abp:
            print(f"  flush ABP   : {len(flush_abp)} period(s) found")
            register("flush_abp", flush_abp)
            found_this_pass = True
        if cal_cvp:
            print(f"  cal  CVP    : {len(cal_cvp)} period(s) found")
            register("cal_cvp", cal_cvp)
            found_this_pass = True
        if flush_cvp:
            print(f"  flush CVP   : {len(flush_cvp)} period(s) found")
            register("flush_cvp", flush_cvp)
            found_this_pass = True
    except NotImplementedError:
        print("  calibration/flush : not implemented — skipping")

    if found_this_pass:
        continue   # restart tree with updated exclusions

    # ── step 2 ───────────────────────────────────────────────────────────────
    # Compute ringing_ratio = P_high / P_total where P_high is power above
    # 2 × dominant cardiac frequency.  A spike in ringing_ratio that exceeds
    # the signal's own baseline by K_THRESH × IQR indicates resonance.
    excl = build_exclusion_mask()
    try:
        slinger_periods = detect_slinger(
            abp, cvp, fs,
            excluded=excl,
            lp_cutoff=LP_CUTOFF,
            hp_cutoff=HP_CUTOFF,
            spec_fmax=SPEC_FMAX,
            k=K_THRESH,
        )
        if slinger_periods:
            print(f"  slinger     : {len(slinger_periods)} period(s) found")
            register("slinger", slinger_periods)
            continue
    except NotImplementedError:
        print("  slinger     : not implemented — skipping")

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
        if gasbubble_periods:
            print(f"  gasbubble   : {len(gasbubble_periods)} period(s) found")
            register("gasbubble", gasbubble_periods)
            continue
    except NotImplementedError:
        print("  gasbubble   : not implemented — skipping")

    # ── step 4 ───────────────────────────────────────────────────────────────
    # Compute LP separately for ABP and CVP.
    # A sustained shift away from the channel's own LP baseline signals that
    # the transducer position has changed or that an IV infusion is running.
    #   ABP LP shifted (or both)  → TRANSDUCER HOOG
    #   CVP LP shifted only       → INFUUS OP CVD
    excl = build_exclusion_mask()
    try:
        transducer_periods = detect_transducer_hoog(
            abp, cvp, fs,
            excluded=excl,
            lp_cutoff=LP_CUTOFF,
            k=K_THRESH,
        )
        if transducer_periods:
            print(f"  transducer  : {len(transducer_periods)} period(s) found")
            register("transducer", transducer_periods)
            found_this_pass = True
    except NotImplementedError:
        print("  transducer  : not implemented — skipping")

    try:
        infuus_periods = detect_infuus(
            abp, cvp, fs,
            excluded=excl,
            lp_cutoff=LP_CUTOFF,
            k=K_THRESH,
        )
        if infuus_periods:
            print(f"  infuus      : {len(infuus_periods)} period(s) found")
            register("infuus", infuus_periods)
            found_this_pass = True
    except NotImplementedError:
        print("  infuus      : not implemented — skipping")

    if found_this_pass:
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
# Channel-specific artefacts (cal/flush) are only shaded on their own subplot.
# Shared artefacts (slinger, gasbubble, transducer, infuus) appear on both.
ARTEFACT_COLOURS = {
    "cal_abp"    : "#1f77b4",   # blue
    "flush_abp"  : "#ff7f0e",   # orange
    "cal_cvp"    : "#17becf",   # teal  (lighter blue — same family as cal)
    "flush_cvp"  : "#ffbb78",   # light orange (same family as flush)
    "slinger"    : "#d62728",   # red
    "gasbubble"  : "#2ca02c",   # green
    "transducer" : "#9467bd",   # purple
    "infuus"     : "#8c564b",   # brown
}

# Which axes each artefact type should be shaded on
ARTEFACT_AXES = {
    "cal_abp"    : ("abp",),
    "flush_abp"  : ("abp",),
    "cal_cvp"    : ("cvp",),
    "flush_cvp"  : ("cvp",),
    "slinger"    : ("abp", "cvp"),
    "gasbubble"  : ("abp", "cvp"),
    "transducer" : ("abp", "cvp"),
    "infuus"     : ("abp", "cvp"),
}

fig, (ax_abp, ax_cvp) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
fig.suptitle(os.path.basename(path), fontsize=11)

ax_abp.plot(t, abp, color="0.3", lw=0.8)
ax_abp.set_ylabel("ABP (mmHg)")

ax_cvp.plot(t, cvp, color="0.3", lw=0.8)
ax_cvp.set_ylabel("CVP (mmHg)")
ax_cvp.set_xlabel("Time (s)")

axes_map = {"abp": ax_abp, "cvp": ax_cvp}

legend_patches = []
for key, periods in artefacts.items():
    if not periods:
        continue
    colour   = ARTEFACT_COLOURS[key]
    on_axes  = ARTEFACT_AXES[key]
    for ax_name in on_axes:
        ax = axes_map[ax_name]
        for s, e in periods:
            ax.axvspan(t[s], t[min(e, n - 1)], color=colour, alpha=0.35, lw=0)
    legend_patches.append(
        mpatches.Patch(color=colour, alpha=0.6, label=key)
    )

if legend_patches:
    ax_abp.legend(handles=legend_patches, loc="upper right",
                  fontsize=8, ncol=len(legend_patches))

for ax in (ax_abp, ax_cvp):
    ax.set_xlim(t[0], t[-1])

plt.tight_layout()
plt.show()
