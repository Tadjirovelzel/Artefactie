def is_artifact(features):
    return any([
        features["zero_frac_abp"] > 0.7,
        features["zero_frac_cvp"] > 0.7,
        abs(features["delta_med_abp"]) > 15,
        abs(features["delta_med_cvp"]) > 4,
        features["amp_ratio_abp"] < 0.6,
        features["amp_ratio_cvp"] < 0.6,
        features["rough_ratio_abp"] > 2.0,
        features["rough_ratio_cvp"] > 2.0,
    ])

def is_flush_like(features):
    flush_abp = (
        features["delta_med_abp"] > 12 and
        features["high_frac_abp"] > 0.5 and
        features["zero_frac_abp"] < 0.2
    )

    flush_cvp = (
        features["delta_med_cvp"] > 4 and
        features["high_frac_cvp"] > 0.5 and
        features["zero_frac_cvp"] < 0.2
    )

    return flush_abp or flush_cvp

def is_infuus_like(features):
    return (
        features["delta_med_cvp"] > 15 and
        features["high_frac_cvp"] > 0.5 and
        abs(features["delta_med_abp"]) < 20 and
        features["zero_frac_cvp"] < 0.3 and
        features["duration_s"] > 2.0
    )

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

    score = 0

    # één van de signalen begint richting 0 te gaan
    if abs(border_features["med_abp_seg"]) < 0.9 * abs(border_features["med_abp_ref"]):
        score += 1
    if abs(border_features["med_cvp_seg"]) < 0.9 * abs(border_features["med_cvp_ref"]):
        score += 1

    # pulsatie begint weg te vallen
    if border_features["amp_ratio_abp"] < 0.9:
        score += 1
    if border_features["amp_ratio_cvp"] < 0.9:
        score += 1

    # al iets van near-zero gedrag
    if border_features["zero_frac_abp"] > 0.02:
        score += 1
    if border_features["zero_frac_cvp"] > 0.02:
        score += 1

    # beetje extra onrust is toegestaan
    if border_features["rough_ratio_abp"] > 1.1 or border_features["rough_ratio_cvp"] > 1.1:
        score += 1

    return score >= 2


def classify_artifact(features):
    if not is_artifact(features):
        return "geen_artefact"

    if (
        (features["zero_frac_abp"] > 0.8 or features["zero_frac_cvp"] > 0.8) and
        (features["amp_ratio_abp"] < 0.3 or features["amp_ratio_cvp"] < 0.3)
    ):
        return "calibratie_core"

    if is_infuus_like(features):
        return "infuus_op_cvp"

    if is_flush_like(features):
        return "flush"

    if (
        (
            features["delta_med_abp"] < -15 or
            features["delta_med_cvp"] < -4
        ) and
        features["amp_ratio_abp"] > 0.5 and
        features["amp_ratio_cvp"] > 0.5
    ):
        return "transducer_hoog"
    
    if (
        (features["rough_ratio_abp"] > 2.5 or features["rough_ratio_cvp"] > 2.5) and
        abs(features["delta_med_cvp"]) < 10 and
        abs(features["delta_med_abp"]) < 20
    ):
        return "slinger"

    if (
        features["amp_ratio_abp"] < 0.6 and
        abs(features["delta_med_abp"]) < 20 and
        features["rough_ratio_abp"] < 2.5
    ):
        return "gasbel"

    return "onbekend_artefact"