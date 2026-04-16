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
     └─ Step 5: Slinger (ABP/CVP) ────┴──► Only reached when steps 1–4
                                           found nothing new this iteration.
                                           Registers ALL slinger periods at once.
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

from signal_helpers import load_signal
from detection import (
    detect_calibration,
    detect_flush_infuus,
    detect_slinger,
    detect_gasbubble,
    detect_transducer_hoog,
)
from visualise import plot_results

# ── parameters ────────────────────────────────────────────────────────────────
LP_CUTOFF = 0.5   # low-pass cutoff used for baseline tracking (Hz)
HP_CUTOFF = 1.0   # high-pass cutoff used for beat-amplitude/std tracking (Hz)
SPEC_FMAX = 15.0  # upper frequency limit for spectrogram features (Hz)

# Artefact keys that are shared across both channels (used in per-channel masks)
_SHARED_KEYS = ("gasbubble_abp", "gasbubble_cvp", "transducer")

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
    "cal_abp"      : [],
    "flush_abp"    : [],
    "infuus_abp"   : [],
    "cal_cvp"      : [],
    "flush_cvp"    : [],
    "infuus_cvp"   : [],
    "slinger_abp"  : [],
    "slinger_cvp"  : [],
    "gasbubble_abp": [],
    "gasbubble_cvp": [],
    "transducer"   : [],
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


def _register_first(key, periods):
    """Register only the earliest-starting non-overlapping new period."""
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


# ── slinger diagnostic data (captured from the last slinger run) ──────────────
slinger_diag_abp = slinger_diag_cvp = None

# ── decision tree (iterative) ─────────────────────────────────────────────────
iteration = 0

while True:
    iteration += 1
    print(f"\n── iteration {iteration} ──")

    excl_before = build_exclusion_mask().sum()

    # Exclusion masks for this iteration — computed once and shared across steps.
    excl     = build_exclusion_mask()
    excl_abp = build_exclusion_mask(("cal_abp", "flush_abp", "infuus_abp", "slinger_abp") + _SHARED_KEYS)
    excl_cvp = build_exclusion_mask(("cal_cvp", "flush_cvp", "infuus_cvp", "slinger_cvp") + _SHARED_KEYS)

    # ── Step 1 : Calibration ──────────────────────────────────────────────────
    try:
        cal_abp, cal_cvp = detect_calibration(
            abp, cvp, fs, excluded=excl,
            lp_cutoff=LP_CUTOFF, hp_cutoff=HP_CUTOFF,
            excluded_abp=excl_abp, excluded_cvp=excl_cvp,
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
            excluded_abp=excl_abp, excluded_cvp=excl_cvp,
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
    try:
        transducer_periods = detect_transducer_hoog(
            abp, cvp, fs, excluded=excl,
            lp_cutoff=LP_CUTOFF, hp_cutoff=HP_CUTOFF,
        )
        if _register_first("transducer", transducer_periods):
            continue
    except Exception as e:
        print(f"  transducer error: {e}")

    # ── Step 4 : Gasbubbel (Damping) ──────────────────────────────────────────
    try:
        gasbubble_abp_periods, gasbubble_cvp_periods = detect_gasbubble(
            abp, cvp, fs,
            mask=excl,
            lp_cutoff=LP_CUTOFF,
            hp_cutoff=HP_CUTOFF,
        )
        registered = False
        for key, periods in [("gasbubble_abp", gasbubble_abp_periods),
                              ("gasbubble_cvp", gasbubble_cvp_periods)]:
            if _register_first(key, periods):
                registered = True
                break
        if registered:
            continue
    except Exception as e:
        print(f"  gasbubble error: {e}")

    # ── Step 5 : Slinger ──────────────────────────────────────────────────────
    # Only reached when steps 1–4 found nothing new this iteration.
    # All slinger periods are registered in one go — no exclusion between them.
    try:
        slinger_abp_periods, slinger_cvp_periods, diag_abp, diag_cvp = detect_slinger(
            abp, cvp, fs, excluded=excl,
        )
        slinger_diag_abp, slinger_diag_cvp = diag_abp, diag_cvp
        excl_now = build_exclusion_mask()
        new_abp = [p for p in slinger_abp_periods if not excl_now[p[0]:p[1]].any()]
        new_cvp = [p for p in slinger_cvp_periods if not excl_now[p[0]:p[1]].any()]
        if new_abp or new_cvp:
            register("slinger_abp", new_abp)
            register("slinger_cvp", new_cvp)
            for p in new_abp:
                print(f"  slinger_abp  : registered  {p[0]/fs:.1f}–{p[1]/fs:.1f} s")
            for p in new_cvp:
                print(f"  slinger_cvp  : registered  {p[0]/fs:.1f}–{p[1]/fs:.1f} s")
            continue
    except Exception as e:
        print(f"  slinger error: {e}")

    # ── Termination ───────────────────────────────────────────────────────────
    print("  no new artefacts — done.")
    break

# ── summary ───────────────────────────────────────────────────────────────────
print("\n── summary ─────────────────────────────────────────────────────────")
for key, periods in artefacts.items():
    if periods:
        spans = ", ".join(f"{t[s]:.1f}–{t[min(e, n-1)]:.1f} s" for s, e in periods)
        print(f"  {key:12s}  {len(periods):2d} period(s):  {spans}")

# ── plot ──────────────────────────────────────────────────────────────────────
plot_results(
    t, abp, cvp, fs, artefacts,
    slinger_diag_abp, slinger_diag_cvp,
    path, LP_CUTOFF, HP_CUTOFF, SPEC_FMAX,
)
