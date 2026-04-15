"""
startagain.py  —  artefact detection decision tree

Decision tree (iterative/lazy):

    START
     │
     ├─ Step 1: Calibration (ABP/CVP) ──► Restart if found
     │
     ├─ Step 2: Flush / Infuus (ABP/CVP) ──► Restart if found
     │
     ├─ Step 3: Transducer Hoog (ABP) ──► LP dropped? 
     │                          YES ──► TRANSDUCER HOOG (restart)
     │                          NO  ──┐
     │                                │
     ├─ Step 4: Gasbubbel (ABP) ──────┴──► HP std low & LP stable?
     │                          YES ──► GASBUBBLE (restart)
     │                          NO  ──┐
     │                                │
     └─ Step 5: Slinger (ABP/CVP) ────► Resonance detected?
                                YES ──► SLINGER (restart)
                                NO  ──► END (no new artefacts)

After each detection the flagged period is excluded and the tree restarts
from the top. This ensures that a baseline for 'clean signal' is always 
maintained and that artefacts are not double-counted.
"""

import tkinter as tk
from tkinter import filedialog
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from signal_helpers import load_signal, lowpass, highpass, compute_spectrogram
from detection import (
    detect_calibration,
    detect_flush_infuus,
    detect_slinger,
    detect_gasbubble,
    detect_transducer_hoog,
)

# ── parameters ────────────────────────────────────────────────────────────────
LP_CUTOFF = 0.5   # low-pass cutoff used for baseline tracking (Hz)
HP_CUTOFF = 1.0    # high-pass cutoff used for beat-amplitude/std tracking (Hz)
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

# ── artefact registry ─────────────────────────────────────────────────────────
artefacts = {
    "cal_abp"    : [],
    "flush_abp"  : [],
    "infuus_abp" : [],
    "cal_cvp"    : [],
    "flush_cvp"  : [],
    "infuus_cvp" : [],
    "slinger_abp": [],
    "slinger_cvp": [],
    "gasbubble"  : [],
    "transducer" : [],
}


def build_exclusion_mask(keys=None):
    """Return a boolean array that is True wherever any artefact was found."""
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
    print(f"\n── iteration {iteration} ──")

    excl_before = build_exclusion_mask().sum()

    def _register_first(key, periods):
        """Register only the earliest-starting non-overlapping period."""
        if not periods:
            return False
        excl_now = build_exclusion_mask()
        non_overlapping = [p for p in periods if not excl_now[p[0]:p[1]].any()]
        if not non_overlapping:
            return False
        first = min(non_overlapping, key=lambda p: p[0])
        register(key, [first])
        print(f"  {key:12s}: registered  {first[0]/fs:.1f}–{first[1]/fs:.1f} s")
        return True

    # ── Step 1 : Calibration ──────────────────────────────────────────────────
    excl = build_exclusion_mask()
    _shared = ('gasbubble', 'transducer')
    excl_abp = build_exclusion_mask(('cal_abp', 'flush_abp', 'infuus_abp', 'slinger_abp') + _shared)
    excl_cvp = build_exclusion_mask(('cal_cvp', 'flush_cvp', 'infuus_cvp', 'slinger_cvp') + _shared)
    
    try:
        cal_abp, cal_cvp = detect_calibration(
            abp, cvp, fs, excluded=excl, 
            lp_cutoff=LP_CUTOFF, hp_cutoff=HP_CUTOFF,
            excluded_abp=excl_abp, excluded_cvp=excl_cvp
        )
        for ch_candidates, ch_excl in [
            ([("cal_abp", p) for p in cal_abp], excl_abp),
            ([("cal_cvp", p) for p in cal_cvp], excl_cvp),
        ]:
            non_overlapping = [(k, p) for k, p in ch_candidates if not ch_excl[p[0]:p[1]].any()]
            if non_overlapping:
                key, first = min(non_overlapping, key=lambda x: x[1][0])
                register(key, [first])
                break
    except Exception as e:
        print(f"  calibration error: {e}")

    if build_exclusion_mask().sum() > excl_before:
        continue

    # ── Step 2 : Flush / Infuus ───────────────────────────────────────────────
    try:
        flush_abp, infuus_abp, flush_cvp, infuus_cvp = detect_flush_infuus(
            abp, cvp, fs, excluded=excl,
            lp_cutoff=LP_CUTOFF, hp_cutoff=HP_CUTOFF,
            excluded_abp=excl_abp, excluded_cvp=excl_cvp
        )
        for ch_candidates, ch_excl in [
            ([("flush_abp", p) for p in flush_abp] + [("infuus_abp", p) for p in infuus_abp], excl_abp),
            ([("flush_cvp", p) for p in flush_cvp] + [("infuus_cvp", p) for p in infuus_cvp], excl_cvp),
        ]:
            non_overlapping = [(k, p) for k, p in ch_candidates if not ch_excl[p[0]:p[1]].any()]
            if non_overlapping:
                key, first = min(non_overlapping, key=lambda x: x[1][0])
                register(key, [first])
                break
    except Exception as e:
        print(f"  flush/infuus error: {e}")

    if build_exclusion_mask().sum() > excl_before:
        continue

    # ── Step 3 : Transducer Hoog ──────────────────────────────────────────────
    # LP_ABP drops. If this is found, the loop restarts.
    try:
        transducer_periods = detect_transducer_hoog(
            abp, cvp, fs, excluded=excl,
            lp_cutoff=LP_CUTOFF, hp_cutoff=HP_CUTOFF
        )
        if _register_first("transducer", transducer_periods):
            continue
    except Exception as e:
        print(f"  transducer error: {e}")

    # ── Step 4 : Gasbubbel (Damping) ──────────────────────────────────────────
    # ONLY checked if transducer was NOT detected as high.
    # Logic: HP std is significantly lower while LP signal is NOT lowered.
    try:
        gasbubble_periods = detect_gasbubble(
            abp, cvp, fs, 
            excluded=excl,
            lp_cutoff=LP_CUTOFF,
            hp_cutoff=HP_CUTOFF,
            spec_fmax=SPEC_FMAX,
            k=K_THRESH
        )
        if _register_first("gasbubble", gasbubble_periods):
            continue
    except Exception as e:
        print(f"  gasbubble error: {e}")

    # ── Step 5 : Slinger ──────────────────────────────────────────────────────
    try:
        slinger_abp_periods, slinger_cvp_periods = detect_slinger(
            abp, cvp, fs, excluded=excl,
            lp_cutoff=LP_CUTOFF, hp_cutoff=HP_CUTOFF,
            spec_fmax=SPEC_FMAX, k=K_THRESH
        )
        candidates = ([("slinger_abp", p) for p in slinger_abp_periods] +
                      [("slinger_cvp", p) for p in slinger_cvp_periods])
        if candidates:
            excl_now = build_exclusion_mask()
            non_overlapping = [(key, p) for key, p in candidates if not excl_now[p[0]:p[1]].any()]
            if non_overlapping:
                key, first = min(non_overlapping, key=lambda x: x[1][0])
                register(key, [first])
                continue
    except Exception as e:
        print(f"  slinger error: {e}")

    # ── Termination ───────────────────────────────────────────────────────────
    print("  no new artefacts — done.")
    break

# ── Summary & Plotting ────────────────────────────────────────────────────────
print("\n── summary ─────────────────────────────────────────────────────────")
for key, periods in artefacts.items():
    if periods:
        spans = ", ".join(f"{t[s]:.1f}–{t[min(e, n-1)]:.1f} s" for s, e in periods)
        print(f"  {key:12s}  {len(periods):2d} period(s):  {spans}")

# (Plotting code remains the same as in your original file)
# ... [Rest of original plotting code] ...
# ── plot ──────────────────────────────────────────────────────────────────────
# Layout: 2 rows (ABP, CVP) × 3 columns (filtered signal | spectrogram | artefacts)
# Channel-specific artefacts (cal/flush/infuus) are only shaded on their own row.
# Shared artefacts (slinger, gasbubble, transducer) appear on both rows.

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

# ── Row 3: DUAL-AXIS DERIVATIVES (LP vs HP) ─────────────────────────────────────
derivative_data = [
    (dlp_abp, dhp_abp, "ABP"),
    (dlp_cvp, dhp_cvp, "CVP")
]

for col, (dlp, dhp, label) in enumerate(derivative_data):
    ax_left = axes[3, col]
    ax_left.set_title(f"{label} - Derivatives (LP left, HP right)")
    
    # Left Axis: Low Pass Derivative (Baseline changes)
    ln1 = ax_left.plot(t, dlp, color="C0", lw=1.2, label="d/dt LP (Baseline)")
    ax_left.set_ylabel("LP Change (mmHg/s)", color="C0")
    ax_left.tick_params(axis='y', labelcolor="C0")
    # Horizontal line at zero to show stability
    ax_left.axhline(0, color="0.6", lw=0.8, ls="--")
    
    # Create the Right Axis for High Pass Derivative (Pulsations)
    ax_right = ax_left.twinx()
    ln2 = ax_right.plot(t, dhp, color="C1", lw=0.8, alpha=0.5, label="d/dt HP (Oscillations)")
    ax_right.set_ylabel("HP Change (mmHg/s)", color="C1")
    ax_right.tick_params(axis='y', labelcolor="C1")
    
    # Merge legends from both axes into one box
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax_left.legend(lns, labs, loc="upper right", fontsize=8)
    
    ax_left.set_xlim(t[0], t[-1])

plt.tight_layout(rect=[0, 0.04 if legend_patches else 0, 1, 1])
plt.show()
