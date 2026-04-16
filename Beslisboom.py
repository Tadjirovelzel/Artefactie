import numpy as np
import pandas as pd

from readArtefacts import read_Artefacts
from detection import (
    detect_calibration,
    detect_flush_infuus,
    detect_slinger,
    detect_gasbubble,
    detect_transducer_hoog,
)
from visualise import plot_results, ARTEFACT_INFO
from beslisboom_helpers import (
    LP_CUTOFF, HP_CUTOFF, SPEC_FMAX, SHARED_KEYS,
    make_artefacts, build_exclusion_mask, register, register_first,
)

# ── input ─────────────────────────────────────────────────────────────────────
path = r"h:\school\Artefactie\data\KT3401_AFdata_2025"
folder = "Calibratie"
filename = "D01Cal.xlsx"
fs = 100
t, ABP, CVP = read_Artefacts(path, folder, filename, fs)

ABP = np.asarray(ABP, dtype=float)
CVP = np.asarray(CVP, dtype=float)
valid = ~(np.isnan(ABP) | np.isnan(CVP))
t, ABP, CVP = t[valid], ABP[valid], CVP[valid]

n = len(ABP)
artefacts = make_artefacts()
slinger_diag_abp = slinger_diag_cvp = None

# ── decision tree (iterative) ─────────────────────────────────────────────────
while True:
    excl_before = build_exclusion_mask(artefacts, n).sum()
    excl     = build_exclusion_mask(artefacts, n)
    excl_abp = build_exclusion_mask(artefacts, n, ("cal_abp", "flush_abp", "infuus_abp", "slinger_abp") + SHARED_KEYS)
    excl_cvp = build_exclusion_mask(artefacts, n, ("cal_cvp", "flush_cvp", "infuus_cvp", "slinger_cvp") + SHARED_KEYS)

    # ── Step 1 : Calibration ──────────────────────────────────────────────────
    try:
        cal_abp, cal_cvp = detect_calibration(
            ABP, CVP, fs, excluded=excl,
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
                register(artefacts, key, [first])
                break
    except Exception:
        pass

    if build_exclusion_mask(artefacts, n).sum() > excl_before:
        continue

    # ── Step 2 : Flush / Infuus ───────────────────────────────────────────────
    try:
        flush_abp, infuus_abp, flush_cvp, infuus_cvp = detect_flush_infuus(
            ABP, CVP, fs, excluded=excl,
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
                register(artefacts, key, [first])
                break
    except Exception:
        pass

    if build_exclusion_mask(artefacts, n).sum() > excl_before:
        continue

    # ── Step 3 : Transducer Hoog ──────────────────────────────────────────────
    try:
        transducer_periods = detect_transducer_hoog(
            ABP, CVP, fs, excluded=excl,
            lp_cutoff=LP_CUTOFF, hp_cutoff=HP_CUTOFF,
        )
        if register_first(artefacts, n, fs, "transducer", transducer_periods):
            continue
    except Exception:
        pass

    # ── Step 4 : Gasbubbel (Damping) ──────────────────────────────────────────
    try:
        gasbubble_abp_periods, gasbubble_cvp_periods = detect_gasbubble(
            ABP, CVP, fs, mask=excl,
            lp_cutoff=LP_CUTOFF, hp_cutoff=HP_CUTOFF,
        )
        registered = False
        for key, periods in [("gasbubble_abp", gasbubble_abp_periods),
                              ("gasbubble_cvp", gasbubble_cvp_periods)]:
            if register_first(artefacts, n, fs, key, periods):
                registered = True
                break
        if registered:
            continue
    except Exception:
        pass

    # ── Step 5 : Slinger ──────────────────────────────────────────────────────
    try:
        slinger_abp_periods, slinger_cvp_periods, diag_abp, diag_cvp = detect_slinger(
            ABP, CVP, fs, excluded=excl,
        )
        slinger_diag_abp, slinger_diag_cvp = diag_abp, diag_cvp
        excl_now = build_exclusion_mask(artefacts, n)
        new_abp = [p for p in slinger_abp_periods if not excl_now[p[0]:p[1]].any()]
        new_cvp = [p for p in slinger_cvp_periods if not excl_now[p[0]:p[1]].any()]
        if new_abp or new_cvp:
            register(artefacts, "slinger_abp", new_abp)
            register(artefacts, "slinger_cvp", new_cvp)
            continue
    except Exception:
        pass

    break

# ── Output 1 : figure ─────────────────────────────────────────────────────────
plot_results(
    t, ABP, CVP, fs, artefacts,
    slinger_diag_abp, slinger_diag_cvp,
    filename if filename else "recording",
    LP_CUTOFF, HP_CUTOFF, SPEC_FMAX,
)

# ── Output 2 : DataFrame ──────────────────────────────────────────────────────
rows = []
for key, periods in artefacts.items():
    naam, signaal = ARTEFACT_INFO[key]
    for s, e in periods:
        rows.append({
            "Starttijd"            : round(float(t[s]), 2),
            "Eindtijd"             : round(float(t[min(e, n - 1)]), 2),
            "Naam van het artefact": naam,
            "Signaal"              : signaal,
        })

artefacten_uit = (
    pd.DataFrame(rows, columns=["Starttijd", "Eindtijd",
                                 "Naam van het artefact", "Signaal"])
    .sort_values("Starttijd")
    .reset_index(drop=True)
)

print(artefacten_uit.to_string() if not artefacten_uit.empty
      else "geen artefacten gedetecteerd")
