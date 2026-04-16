"""
beslisboom_helpers.py  —  shared state and helpers for the decision tree
"""

import numpy as np

# ── detection parameters ──────────────────────────────────────────────────────
LP_CUTOFF  = 0.5    # low-pass cutoff for baseline tracking (Hz)
HP_CUTOFF  = 1.0    # high-pass cutoff for beat-amplitude tracking (Hz)
SPEC_FMAX  = 15.0   # upper frequency limit for spectrogram display (Hz)

# Artefact keys that affect both channels (used in per-channel exclusion masks)
SHARED_KEYS = ("gasbubble_abp", "gasbubble_cvp", "transducer")

_ARTEFACT_KEYS = [
    "cal_abp", "flush_abp", "infuus_abp",
    "cal_cvp", "flush_cvp", "infuus_cvp",
    "slinger_abp", "slinger_cvp",
    "gasbubble_abp", "gasbubble_cvp",
    "transducer",
]


def make_artefacts():
    """Return a fresh, empty artefact registry."""
    return {k: [] for k in _ARTEFACT_KEYS}


def build_exclusion_mask(artefacts, n, keys=None):
    """Return a boolean array that is True wherever any artefact was found."""
    mask = np.zeros(n, dtype=bool)
    for k, periods in artefacts.items():
        if keys is None or k in keys:
            for start, end in periods:
                mask[start:end] = True
    return mask


def register(artefacts, key, new_periods):
    """Append newly detected periods to the registry."""
    artefacts[key].extend(new_periods)


def register_first(artefacts, n, fs, key, periods):
    """Register only the earliest-starting non-overlapping new period."""
    if not periods:
        return False
    excl_now = build_exclusion_mask(artefacts, n)
    non_overlapping = [p for p in periods if not excl_now[p[0]:p[1]].any()]
    if not non_overlapping:
        return False
    first = min(non_overlapping, key=lambda p: p[0])
    register(artefacts, key, [first])
    return True
