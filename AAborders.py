from AAfeatures import compute_artifact_features, compute_border_features, compute_segment_transition_features

def is_left_border_of_calibration(seg, label, border_features, core_seg, max_gap_s=1.0):
    if border_features is None:
        return False

    if label not in {"slinger", "geen_artefact", "onbekend_artefact"}:
        return False

    if seg["duration_s"] > 3.0:
        return False

    gap = abs(core_seg["start_time"] - seg["stop_time"])
    if gap > max_gap_s:
        return False

    toward_zero_abp = abs(border_features["med_abp_seg"]) < 0.95 * abs(border_features["med_abp_ref"])
    toward_zero_cvp = abs(border_features["med_cvp_seg"]) < 0.95 * abs(border_features["med_cvp_ref"])

    amp_drop_abp = border_features["amp_ratio_abp"] < 0.95
    amp_drop_cvp = border_features["amp_ratio_cvp"] < 0.95

    slight_zero_abp = border_features["zero_frac_abp"] > 0.01
    slight_zero_cvp = border_features["zero_frac_cvp"] > 0.01

    return any([
        toward_zero_abp,
        toward_zero_cvp,
        amp_drop_abp,
        amp_drop_cvp,
        slight_zero_abp,
        slight_zero_cvp,
    ])


def is_right_border_of_calibration(seg, label, border_features, transition_features, core_seg, max_gap_s=2.0):
    if border_features is None or transition_features is None:
        return False

    if label not in {"slinger", "geen_artefact", "onbekend_artefact"}:
        return False

    if seg["duration_s"] > 3.0:
        return False

    gap = abs(seg["start_time"] - core_seg["stop_time"])
    if gap > max_gap_s:
        return False

    # t.o.v. normale referentie rechts
    away_from_zero_abp = abs(border_features["med_abp_seg"]) > 1.02 * abs(border_features["med_abp_ref"])
    away_from_zero_cvp = abs(border_features["med_cvp_seg"]) > 1.02 * abs(border_features["med_cvp_ref"])

    # binnen het segment zelf: herstel
    abp_recovers_inside = abs(transition_features["med_abp_second"]) > 1.05 * abs(transition_features["med_abp_first"])
    cvp_recovers_inside = abs(transition_features["med_cvp_second"]) > 1.05 * abs(transition_features["med_cvp_first"])

    abp_amp_recovers = transition_features["amp_abp_second"] > 1.05 * transition_features["amp_abp_first"]
    cvp_amp_recovers = transition_features["amp_cvp_second"] > 1.05 * transition_features["amp_cvp_first"]

    return any([
        away_from_zero_abp,
        away_from_zero_cvp,
        abp_recovers_inside,
        cvp_recovers_inside,
        abp_amp_recovers,
        cvp_amp_recovers,
    ])

def is_border_of_calibration(seg, label, border_features, core_seg, max_gap_s=1.0):
    if border_features is None:
        return False

    if label not in {"slinger", "geen_artefact", "onbekend_artefact"}:
        return False

    if seg["duration_s"] > 3.0:
        return False

    gap_left = abs(core_seg["start_time"] - seg["stop_time"])
    gap_right = abs(seg["start_time"] - core_seg["stop_time"])
    if min(gap_left, gap_right) > max_gap_s:
        return False

    toward_zero_abp = abs(border_features["med_abp_seg"]) < 0.95 * abs(border_features["med_abp_ref"])
    toward_zero_cvp = abs(border_features["med_cvp_seg"]) < 0.95 * abs(border_features["med_cvp_ref"])

    amp_drop_abp = border_features["amp_ratio_abp"] < 0.95
    amp_drop_cvp = border_features["amp_ratio_cvp"] < 0.95

    slight_zero_abp = border_features["zero_frac_abp"] > 0.01
    slight_zero_cvp = border_features["zero_frac_cvp"] > 0.01

    return any([
        toward_zero_abp,
        toward_zero_cvp,
        amp_drop_abp,
        amp_drop_cvp,
        slight_zero_abp,
        slight_zero_cvp,
    ])


def relabel_calibration_borders(results, df):
    if not results:
        return results

    results = sorted(results, key=lambda r: r["segment"]["start_time"])

    for i, r in enumerate(results):
        if r["label"] != "calibratie_core":
            continue

        core_seg = r["segment"]

        # linker buur
        if i - 1 >= 0:
            left = results[i - 1]
            bf_left = compute_border_features(
                df,
                left["segment"]["start_idx"],
                left["segment"]["stop_idx"],
                side="left"
            )
            if is_left_border_of_calibration(left["segment"], left["label"], bf_left, core_seg):
                left["label"] = "border_calibratie"

        # rechter buur
        if i + 1 < len(results):
            right = results[i + 1]
            bf_right = compute_border_features(
                df,
                right["segment"]["start_idx"],
                right["segment"]["stop_idx"],
                side="right"
            )
            tf_right = compute_segment_transition_features(
                df,
                right["segment"]["start_idx"],
                right["segment"]["stop_idx"]
            )
            if is_right_border_of_calibration(right["segment"], right["label"], bf_right, tf_right, core_seg):
                right["label"] = "border_calibratie"

    return results


def merge_calibration_labels(results, df):
    """
    Merge:
        border_calibratie + calibratie_core + border_calibratie
    into one final segment with label 'calibratie'.

    Works on results in this format:
        {
            "segment": ...,
            "label": ...,
            "features": ...
        }
    """
    if not results:
        return results

    results = sorted(results, key=lambda r: r["segment"]["start_time"])
    merged = []
    i = 0

    cal_labels = {"border_calibratie", "calibratie_core"}

    while i < len(results):
        current = results[i]

        if current["label"] not in cal_labels:
            merged.append(current)
            i += 1
            continue

        start_idx = current["segment"]["start_idx"]
        stop_idx = current["segment"]["stop_idx"]

        j = i + 1
        while j < len(results) and results[j]["label"] in cal_labels:
            stop_idx = results[j]["segment"]["stop_idx"]
            j += 1

        new_seg = {
            "start_idx": int(start_idx),
            "stop_idx": int(stop_idx),
            "start_time": float(df["Time"].iloc[start_idx]),
            "stop_time": float(df["Time"].iloc[stop_idx]),
            "duration_s": float(df["Time"].iloc[stop_idx] - df["Time"].iloc[start_idx]),
        }

        new_features = compute_artifact_features(df, start_idx, stop_idx)

        merged.append({
            "segment": new_seg,
            "label": "calibratie",
            "features": new_features,
        })

        i = j

    return merged