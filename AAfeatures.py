import numpy as np
import pandas as pd


def rolling_puls_amp(x, fs, win_s=1.0):
    x = pd.Series(np.asarray(x, dtype=float))
    w = max(3, int(round(win_s * fs)))
    amp = (
        x.rolling(w, center=True, min_periods=3).max()
        - x.rolling(w, center=True, min_periods=3).min()
    )
    return float(np.nanmedian(amp))


def roughness(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    return float(np.std(np.diff(x)))


def safe_ratio(a, b, eps=1e-6):
    return float(a / (b + eps))


def get_segment_and_reference(df, start_idx, stop_idx, time_col="Time",
                              ref_s=5.0, guard_s=0.5):
    t = df[time_col].to_numpy(dtype=float)
    fs = 1.0 / np.median(np.diff(t))

    ref_n = int(round(ref_s * fs))
    guard_n = int(round(guard_s * fs))

    seg = df.iloc[start_idx:stop_idx + 1].copy()

    before_start = max(0, start_idx - guard_n - ref_n)
    before_stop = max(0, start_idx - guard_n)

    after_start = min(len(df), stop_idx + 1 + guard_n)
    after_stop = min(len(df), stop_idx + 1 + guard_n + ref_n)

    ref_before = df.iloc[before_start:before_stop].copy()
    ref_after = df.iloc[after_start:after_stop].copy()
    ref = pd.concat([ref_before, ref_after], axis=0).reset_index(drop=True)

    return seg, ref, fs


def compute_artifact_features(df, start_idx, stop_idx,
                              time_col="Time", abp_col="ABP", cvp_col="CVP",
                              ref_s=10.0, guard_s=1.0,
                              abp_zero_thr=10.0, cvp_zero_thr=3.0):
    seg, ref, fs = get_segment_and_reference(
        df, start_idx, stop_idx, time_col=time_col,
        ref_s=ref_s, guard_s=guard_s
    )

    abp_seg = seg[abp_col].to_numpy(dtype=float)
    cvp_seg = seg[cvp_col].to_numpy(dtype=float)
    abp_ref = ref[abp_col].to_numpy(dtype=float)
    cvp_ref = ref[cvp_col].to_numpy(dtype=float)

    med_abp_seg = float(np.median(abp_seg))
    med_cvp_seg = float(np.median(cvp_seg))
    med_abp_ref = float(np.median(abp_ref))
    med_cvp_ref = float(np.median(cvp_ref))

    amp_abp_seg = rolling_puls_amp(abp_seg, fs)
    amp_cvp_seg = rolling_puls_amp(cvp_seg, fs)
    amp_abp_ref = rolling_puls_amp(abp_ref, fs)
    amp_cvp_ref = rolling_puls_amp(cvp_ref, fs)

    rough_abp_seg = roughness(abp_seg)
    rough_cvp_seg = roughness(cvp_seg)
    rough_abp_ref = roughness(abp_ref)
    rough_cvp_ref = roughness(cvp_ref)

    return {
        "start_time": float(seg[time_col].iloc[0]),
        "stop_time": float(seg[time_col].iloc[-1]),
        "duration_s": float(seg[time_col].iloc[-1] - seg[time_col].iloc[0]),

        "med_abp_seg": med_abp_seg,
        "med_cvp_seg": med_cvp_seg,
        "med_abp_ref": med_abp_ref,
        "med_cvp_ref": med_cvp_ref,

        "delta_med_abp": med_abp_seg - med_abp_ref,
        "delta_med_cvp": med_cvp_seg - med_cvp_ref,

        "amp_ratio_abp": safe_ratio(amp_abp_seg, amp_abp_ref),
        "amp_ratio_cvp": safe_ratio(amp_cvp_seg, amp_cvp_ref),

        "rough_ratio_abp": safe_ratio(rough_abp_seg, rough_abp_ref),
        "rough_ratio_cvp": safe_ratio(rough_cvp_seg, rough_cvp_ref),

        "zero_frac_abp": float(np.mean(np.abs(abp_seg) < abp_zero_thr)),
        "zero_frac_cvp": float(np.mean(np.abs(cvp_seg) < cvp_zero_thr)),

        "high_frac_abp": float(np.mean(abp_seg > (med_abp_ref + 15))),
        "high_frac_cvp": float(np.mean(cvp_seg > (med_cvp_ref + 4))),
    }


def compute_border_features(
    df, start_idx, stop_idx, side,
    time_col="Time", abp_col="ABP", cvp_col="CVP",
    ref_s=5.0, guard_s=0.5,
    abp_zero_thr=10.0, cvp_zero_thr=3.0
):
    t = df[time_col].to_numpy(dtype=float)
    fs = 1.0 / np.median(np.diff(t))

    ref_n = int(round(ref_s * fs))
    guard_n = int(round(guard_s * fs))

    seg = df.iloc[start_idx:stop_idx + 1].copy()

    if side == "left":
        ref_start = max(0, start_idx - guard_n - ref_n)
        ref_stop = max(0, start_idx - guard_n)
    elif side == "right":
        ref_start = min(len(df), stop_idx + 1 + guard_n)
        ref_stop = min(len(df), stop_idx + 1 + guard_n + ref_n)
    else:
        raise ValueError("side must be 'left' or 'right'")

    ref = df.iloc[ref_start:ref_stop].copy()
    if len(ref) == 0:
        return None

    abp_seg = seg[abp_col].to_numpy(dtype=float)
    cvp_seg = seg[cvp_col].to_numpy(dtype=float)
    abp_ref = ref[abp_col].to_numpy(dtype=float)
    cvp_ref = ref[cvp_col].to_numpy(dtype=float)

    med_abp_seg = float(np.median(abp_seg))
    med_cvp_seg = float(np.median(cvp_seg))
    med_abp_ref = float(np.median(abp_ref))
    med_cvp_ref = float(np.median(cvp_ref))

    amp_abp_seg = rolling_puls_amp(abp_seg, fs)
    amp_cvp_seg = rolling_puls_amp(cvp_seg, fs)
    amp_abp_ref = rolling_puls_amp(abp_ref, fs)
    amp_cvp_ref = rolling_puls_amp(cvp_ref, fs)

    rough_abp_seg = roughness(abp_seg)
    rough_cvp_seg = roughness(cvp_seg)
    rough_abp_ref = roughness(abp_ref)
    rough_cvp_ref = roughness(cvp_ref)

    return {
        "start_time": float(seg[time_col].iloc[0]),
        "stop_time": float(seg[time_col].iloc[-1]),
        "duration_s": float(seg[time_col].iloc[-1] - seg[time_col].iloc[0]),

        "med_abp_seg": med_abp_seg,
        "med_cvp_seg": med_cvp_seg,
        "med_abp_ref": med_abp_ref,
        "med_cvp_ref": med_cvp_ref,

        "amp_ratio_abp": safe_ratio(amp_abp_seg, amp_abp_ref),
        "amp_ratio_cvp": safe_ratio(amp_cvp_seg, amp_cvp_ref),

        "rough_ratio_abp": safe_ratio(rough_abp_seg, rough_abp_ref),
        "rough_ratio_cvp": safe_ratio(rough_cvp_seg, rough_cvp_ref),

        "zero_frac_abp": float(np.mean(np.abs(abp_seg) < abp_zero_thr)),
        "zero_frac_cvp": float(np.mean(np.abs(cvp_seg) < cvp_zero_thr)),
    }

def compute_segment_transition_features(df, start_idx, stop_idx,
                                        time_col="Time", abp_col="ABP", cvp_col="CVP"):
    t = df[time_col].to_numpy(dtype=float)
    fs = 1.0 / np.median(np.diff(t))

    seg = df.iloc[start_idx:stop_idx + 1].copy()
    n = len(seg)

    if n < 6:
        return None

    mid = n // 2
    first = seg.iloc[:mid]
    second = seg.iloc[mid:]

    abp1 = first[abp_col].to_numpy(dtype=float)
    abp2 = second[abp_col].to_numpy(dtype=float)
    cvp1 = first[cvp_col].to_numpy(dtype=float)
    cvp2 = second[cvp_col].to_numpy(dtype=float)

    med_abp_1 = float(np.median(abp1))
    med_abp_2 = float(np.median(abp2))
    med_cvp_1 = float(np.median(cvp1))
    med_cvp_2 = float(np.median(cvp2))

    amp_abp_1 = rolling_puls_amp(abp1, fs)
    amp_abp_2 = rolling_puls_amp(abp2, fs)
    amp_cvp_1 = rolling_puls_amp(cvp1, fs)
    amp_cvp_2 = rolling_puls_amp(cvp2, fs)

    return {
        "med_abp_first": med_abp_1,
        "med_abp_second": med_abp_2,
        "med_cvp_first": med_cvp_1,
        "med_cvp_second": med_cvp_2,
        "amp_abp_first": amp_abp_1,
        "amp_abp_second": amp_abp_2,
        "amp_cvp_first": amp_cvp_1,
        "amp_cvp_second": amp_cvp_2,
    }