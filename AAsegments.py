import numpy as np
import pandas as pd


def find_runs(mask):
    mask = np.asarray(mask, dtype=bool)
    x = np.r_[False, mask, False]
    starts = np.where(~x[:-1] & x[1:])[0]
    stops = np.where(x[:-1] & ~x[1:])[0] - 1
    return list(zip(starts, stops))


def merge_close_runs(runs, max_gap):
    if not runs:
        return []

    merged = [list(runs[0])]
    for s, e in runs[1:]:
        if s - merged[-1][1] - 1 <= max_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    return [tuple(x) for x in merged]


def rolling_ptp(x, w):
    s = pd.Series(np.asarray(x, dtype=float))
    return (
        s.rolling(w, center=True, min_periods=1).max()
        - s.rolling(w, center=True, min_periods=1).min()
    ).to_numpy()


def rolling_roughness(x, w):
    x = np.asarray(x, dtype=float)
    dx = np.diff(x, prepend=x[0])
    s = pd.Series(dx)
    return s.rolling(w, center=True, min_periods=1).std().fillna(0).to_numpy()


def safe_ratio(a, b, eps=1e-6):
    return a / (b + eps)


def detect_candidate_segments(
    df,
    time_col="Time",
    abp_col="ABP",
    cvp_col="CVP",
    short_win_s=1.0,
    long_win_s=8.0,
    min_duration_s=0.8,
    bridge_gap_s=0.3,
    abp_zero_thr=10.0,
    cvp_zero_thr=3.0,
    abp_shift_thr=15.0,
    cvp_shift_thr=4.0,
    amp_drop_thr=0.6,
    rough_rise_thr=2.5,
):
    t = df[time_col].to_numpy(dtype=float)
    abp = df[abp_col].to_numpy(dtype=float)
    cvp = df[cvp_col].to_numpy(dtype=float)

    fs = 1.0 / np.median(np.diff(t))

    w_short = max(3, int(round(short_win_s * fs)))
    w_long = max(w_short + 2, int(round(long_win_s * fs)))
    min_len = max(1, int(round(min_duration_s * fs)))
    max_gap = int(round(bridge_gap_s * fs))

    med_abp = pd.Series(abp).rolling(w_short, center=True, min_periods=1).median().to_numpy()
    med_cvp = pd.Series(cvp).rolling(w_short, center=True, min_periods=1).median().to_numpy()

    amp_abp = rolling_ptp(abp, w_short)
    amp_cvp = rolling_ptp(cvp, w_short)

    rough_abp = rolling_roughness(abp, w_short)
    rough_cvp = rolling_roughness(cvp, w_short)

    med_abp_base = pd.Series(med_abp).rolling(w_long, center=True, min_periods=1).median().to_numpy()
    med_cvp_base = pd.Series(med_cvp).rolling(w_long, center=True, min_periods=1).median().to_numpy()

    amp_abp_base = pd.Series(amp_abp).rolling(w_long, center=True, min_periods=1).median().to_numpy()
    amp_cvp_base = pd.Series(amp_cvp).rolling(w_long, center=True, min_periods=1).median().to_numpy()

    rough_abp_base = pd.Series(rough_abp).rolling(w_long, center=True, min_periods=1).median().to_numpy()
    rough_cvp_base = pd.Series(rough_cvp).rolling(w_long, center=True, min_periods=1).median().to_numpy()

    near_zero = (
        (np.abs(med_abp) < abp_zero_thr) &
        (np.abs(med_cvp) < cvp_zero_thr)
    )

    baseline_shift = (
        (np.abs(med_abp - med_abp_base) > abp_shift_thr) |
        (np.abs(med_cvp - med_cvp_base) > cvp_shift_thr)
    )

    amp_drop = (
        (safe_ratio(amp_abp, amp_abp_base) < amp_drop_thr) |
        (safe_ratio(amp_cvp, amp_cvp_base) < amp_drop_thr)
    )

    rough_rise = (
        (safe_ratio(rough_abp, rough_abp_base) > rough_rise_thr) |
        (safe_ratio(rough_cvp, rough_cvp_base) > rough_rise_thr)
    )

    candidate_mask = near_zero | baseline_shift | amp_drop | rough_rise

    runs = find_runs(candidate_mask)
    runs = merge_close_runs(runs, max_gap)
    runs = [(s, e) for s, e in runs if (e - s + 1) >= min_len]

    segments = []
    for s, e in runs:
        segments.append({
            "start_idx": int(s),
            "stop_idx": int(e),
            "start_time": float(t[s]),
            "stop_time": float(t[e]),
            "duration_s": float(t[e] - t[s]),
        })

    debug = {
        "time": t,
        "candidate_mask": candidate_mask,
        "near_zero": near_zero,
        "baseline_shift": baseline_shift,
        "amp_drop": amp_drop,
        "rough_rise": rough_rise,
    }

    return segments, debug


def is_border_of_calibration(seg, label, features, core_seg, core_features, max_gap_s=1.0):
    # Alleen twijfelachtige / overgangslabels mogen border worden
    if label not in {"slinger", "geen_artefact", "onbekend_artefact"}:
        return False

    # Border moet kort zijn
    if seg["duration_s"] > 3.0:
        return False

    # Border moet direct naast de calibratie-core liggen
    gap_left = abs(core_seg["start_time"] - seg["stop_time"])
    gap_right = abs(seg["start_time"] - core_seg["stop_time"])
    if min(gap_left, gap_right) > max_gap_s:
        return False

    score = 0

    # Eén van beide signalen begint richting nul te gaan
    if features["zero_frac_abp"] > 0.05:
        score += 1
    if features["zero_frac_cvp"] > 0.05:
        score += 1

    # Pulsatie begint weg te vallen
    if features["amp_ratio_abp"] < 0.85:
        score += 1
    if features["amp_ratio_cvp"] < 0.85:
        score += 1

    # Niveau verschuift richting 0
    abp_toward_zero = abs(features["med_abp_seg"]) < abs(features["med_abp_ref"])
    cvp_toward_zero = abs(features["med_cvp_seg"]) < abs(features["med_cvp_ref"])

    if abp_toward_zero:
        score += 1
    if cvp_toward_zero:
        score += 1

    # Overgang mag wat ruwer zijn, maar niet extreem
    if features["rough_ratio_abp"] > 1.2 or features["rough_ratio_cvp"] > 1.2:
        score += 1

    return score >= 2


def relabel_calibration_borders(results):
    if not results:
        return results

    results = sorted(results, key=lambda r: r["segment"]["start_time"])

    core_indices = [
        i for i, r in enumerate(results)
        if r["label"] == "calibratie_core"
    ]

    for i in core_indices:
        core = results[i]
        core_seg = core["segment"]
        core_features = core["features"]

        # linker buur
        if i - 1 >= 0:
            left = results[i - 1]
            if is_border_of_calibration(
                left["segment"], left["label"], left["features"],
                core_seg, core_features
            ):
                left["label"] = "border_calibratie"

        # rechter buur
        if i + 1 < len(results):
            right = results[i + 1]
            if is_border_of_calibration(
                right["segment"], right["label"], right["features"],
                core_seg, core_features
            ):
                right["label"] = "border_calibratie"

    return results

