"""
detection.py  —  artefact detection functions

Each function receives the raw ABP and CVP arrays, the sample rate, a boolean
exclusion mask (True = already excluded), and tuning parameters.  It returns a
list of (start_idx, end_idx) tuples using numpy-slice (end-exclusive) convention.

Baselines must always be derived from the non-excluded portion of the signal so
that previously detected artefacts do not bias the thresholds.
"""

import numpy as np
import pandas as pd
from signal_helpers import lowpass, highpass, rolling_std


# ── shared helpers ─────────────────────────────────────────────────────────────

def _estimate_cardiac_period(abp, clean, fs, fmin=0.5, fmax=3.0, default_s=1.0):
    """
    Estimate the dominant cardiac period from clean ABP samples via FFT.

    Always uses ABP (not CVP) so a degraded or flat CVP does not corrupt the
    cardiac frequency estimate.  Returns *default_s* when there is not enough
    clean data to produce a reliable estimate.
    """
    clean_abp = abp[clean]
    if len(clean_abp) < int(2 * fs):
        return default_s
    fft_mag = np.abs(np.fft.rfft(clean_abp - clean_abp.mean()))
    freqs   = np.fft.rfftfreq(len(clean_abp), 1.0 / fs)
    band    = (freqs >= fmin) & (freqs <= fmax)
    if not band.any():
        return default_s
    f_dom = freqs[band][np.argmax(fft_mag[band])]
    return 1.0 / max(f_dom, 1e-3)


# ── step 1 : calibration ───────────────────────────────────────────────────────

def detect_calibration(abp, cvp, fs, excluded,
                       lp_cutoff,
                       hp_cutoff,
                       cal_threshold=0.1,
                       hp_max_ratio=0.5,
                       cardiac_fmin=0.5,
                       cardiac_fmax=3.0,
                       abp_phys=(30, 160),
                       cvp_phys=(1,  20),
                       excluded_abp=None,
                       excluded_cvp=None):
    """
    Detect calibration periods, separately per channel.

    Entry : a sample drops below cal_threshold × LP mean AND the mean of the
            raw signal over the next cardiac period is also below that threshold.
    Exit  : a sample rises above the threshold AND the mean over the next cardiac
            period is also above it.  Transient spikes in between are skipped.

    The LP mean is estimated only from physiologically valid samples (ABP: 30–160,
    CVP: 3–20 mmHg) so artefact periods do not bias the baseline.

    The cardiac period is derived from ABP (more reliable) and shared by both
    channels so that a degraded CVP signal does not produce a nonsensical window.

    HP std guard
    ------------
    After detection, each candidate is checked against the rolling HP std.
    True calibration has near-absent cardiac oscillations; if the mean HP std
    inside the period exceeds hp_max_ratio × the clean baseline, the period is
    rejected.  This prevents transducer-hoog events that dip below zero from
    being misclassified as calibration.

    Parameters
    ----------
    hp_cutoff    : high-pass cutoff (Hz) used for the HP std guard
    hp_max_ratio : maximum allowed HP std inside a candidate period as a
                   fraction of the clean baseline (default 0.5)
    excluded_abp / excluded_cvp : optional per-channel scan masks so that an
        ABP calibration does not block CVP calibration detection in the same
        window.  Defaults to the shared `excluded` mask when not supplied.
    """
    clean    = ~excluded
    scan_abp = (~excluded_abp) if excluded_abp is not None else clean
    scan_cvp = (~excluded_cvp) if excluded_cvp is not None else clean

    # Cardiac period estimated from clean ABP only, shared by both channels.
    period_s = _estimate_cardiac_period(abp, clean, fs,
                                         fmin=cardiac_fmin, fmax=cardiac_fmax)
    window   = max(int(period_s * fs), 1)

    def _detect_cal(signal, phys_lo, phys_hi, scan_clean):
        lp        = lowpass(signal, fs, cutoff_hz=lp_cutoff)
        phys_mask = clean & (lp >= phys_lo) & (lp <= phys_hi)
        lp_mean   = (np.mean(lp[phys_mask]) if phys_mask.any() else
                     np.mean(lp[clean])      if clean.any()     else np.mean(lp))
        cal_thr   = cal_threshold * lp_mean
        n         = len(signal)

        # HP std baseline for the oscillation guard
        hp        = highpass(signal, fs, cutoff_hz=hp_cutoff)
        hp_std_r  = rolling_std(hp, fs, window_s=1.0)
        valid_std = clean & ~np.isnan(hp_std_r)
        hp_base   = (float(np.median(hp_std_r[valid_std])) if valid_std.any()
                     else float(np.nanmedian(hp_std_r)))

        def _win_mean(idx):
            return np.mean(signal[idx : min(idx + window, n)])

        # ── scan for below-threshold periods ─────────────────────────────────
        raw_periods = []
        i = 0
        while i < n:
            if signal[i] < cal_thr and scan_clean[i]:
                if _win_mean(i) < cal_thr:
                    start = i
                    j = i + 1
                    while j < n:
                        if signal[j] >= cal_thr:
                            if _win_mean(j) >= cal_thr:
                                if j - start >= window:
                                    raw_periods.append((start, j))
                                i = j
                                break
                            k = j + 1
                            while k < n and signal[k] >= cal_thr:
                                k += 1
                            j = k
                        else:
                            j += 1
                    else:
                        if n - start >= window:
                            raw_periods.append((start, n))
                        i = n
                else:
                    i += 1
            else:
                i += 1

        # ── HP std guard: reject periods where oscillations continue ─────────
        # Use median so a brief noisy transient at the start of calibration
        # does not bias the estimate and cause a false rejection.
        periods = []
        for start, end in raw_periods:
            seg      = hp_std_r[start:end]
            seg      = seg[~np.isnan(seg)]
            median_hp = float(np.median(seg)) if seg.size > 0 else 0.0
            ratio    = median_hp / hp_base if hp_base > 0 else 0.0
            if ratio <= hp_max_ratio:
                periods.append((start, end))

        return periods

    cal_abp = _detect_cal(abp, *abp_phys, scan_abp)
    cal_cvp = _detect_cal(cvp, *cvp_phys, scan_cvp)
    return cal_abp, cal_cvp


# ── step 2 : flush / infuus ────────────────────────────────────────────────────

def detect_flush_infuus(abp, cvp, fs, excluded,
                        lp_cutoff, hp_cutoff,
                        flush_threshold=1.8,
                        min_flush_cycles=1.5,
                        infuus_min_s=5.0,
                        infuus_hp_ratio=0.5,
                        cardiac_fmin=0.5,
                        cardiac_fmax=3.0,
                        abp_phys=(30, 160),
                        cvp_phys=(1,  20),
                        excluded_abp=None,
                        excluded_cvp=None):
    """
    Detect high-pressure periods (raw > flush_threshold × LP mean) and classify
    each as flush or infuus.

    Detection
    ---------
    Same entry/exit confirmation as calibration: a sample triggers entry when it
    exceeds the threshold, confirmed if the mean of the raw signal over the next
    cardiac period also exceeds it.  Transient drops below the threshold are
    skipped unless the window mean confirms the exit.

    Classification
    --------------
    After detecting a high-threshold period:

    * flush  — brief injection: period shorter than infuus_min_s, OR the mean
               rolling HP std inside the period is below infuus_hp_ratio × the
               baseline HP std (cardiac pulsations have been damped / cut off).

    * infuus — sustained IV infusion: period ≥ infuus_min_s AND mean rolling
               HP std ≥ infuus_hp_ratio × baseline (cardiac pulsations continue
               through the elevated pressure).

    Parameters
    ----------
    flush_threshold  : high-threshold as a multiple of LP mean (default 1.5)
    min_flush_cycles : minimum duration of a high-threshold period in cardiac
                       cycles before it is registered at all (default 1)
    infuus_min_s     : minimum duration for the infuus classification (default 5.0 s)
    infuus_hp_ratio  : HP std inside period must be ≥ this fraction of the
                       baseline HP std to classify as infuus (default 0.5)
    excluded_abp / excluded_cvp : per-channel scan masks (see detect_calibration)
    """
    clean    = ~excluded
    scan_abp = (~excluded_abp) if excluded_abp is not None else clean
    scan_cvp = (~excluded_cvp) if excluded_cvp is not None else clean

    # Cardiac period estimated from clean ABP only, shared by both channels.
    period_s       = _estimate_cardiac_period(abp, clean, fs,
                                               fmin=cardiac_fmin, fmax=cardiac_fmax)
    window         = max(int(period_s * fs), 1)
    min_flush_samp = max(int(min_flush_cycles * window), 1)

    def _detect_high(signal, phys_lo, phys_hi, scan_clean):
        lp        = lowpass(signal, fs, cutoff_hz=lp_cutoff)
        phys_mask = clean & (lp >= phys_lo) & (lp <= phys_hi)
        lp_mean   = (np.mean(lp[phys_mask]) if phys_mask.any() else
                     np.mean(lp[clean])      if clean.any()     else np.mean(lp))
        flush_thr = flush_threshold * lp_mean
        n         = len(signal)

        # HP signal for oscillation classification
        hp          = highpass(signal, fs, cutoff_hz=hp_cutoff)
        hp_std_roll = rolling_std(hp, fs, window_s=1.0)

        def _win_mean(idx):
            return np.mean(signal[idx : min(idx + window, n)])

        # ── first pass: find all high-threshold periods ───────────────────────
        high_periods = []
        i = 0
        while i < n:
            if signal[i] > flush_thr and scan_clean[i]:
                if _win_mean(i) > flush_thr:
                    start = i
                    j = i + 1
                    while j < n:
                        if signal[j] <= flush_thr:
                            if _win_mean(j) <= flush_thr:
                                if j - start >= min_flush_samp:
                                    high_periods.append((start, j))
                                i = j
                                break
                            k = j + 1
                            while k < n and signal[k] <= flush_thr:
                                k += 1
                            j = k
                        else:
                            j += 1
                    else:
                        if n - start >= min_flush_samp:
                            high_periods.append((start, n))
                        i = n
                else:
                    i += 1
            else:
                i += 1

        # ── baseline HP std: clean samples outside all high-threshold periods ─
        high_mask = np.zeros(n, dtype=bool)
        for s, e in high_periods:
            high_mask[s:e] = True
        baseline_valid = clean & ~high_mask & ~np.isnan(hp_std_roll)
        hp_std_base    = (float(np.median(hp_std_roll[baseline_valid]))
                          if baseline_valid.any() else float(np.nanmedian(hp_std_roll)))
        # ── classify each period ──────────────────────────────────────────────
        flush_periods  = []
        infuus_periods = []
        for start, end in high_periods:
            duration_s  = (end - start) / fs
            seg_std     = hp_std_roll[start:end]
            seg_std     = seg_std[~np.isnan(seg_std)]
            mean_hp_std = float(seg_std.mean()) if seg_std.size > 0 else 0.0
            hp_ratio    = mean_hp_std / hp_std_base if hp_std_base > 0 else 0.0

            is_infuus = (duration_s >= infuus_min_s and hp_ratio >= infuus_hp_ratio)
            (infuus_periods if is_infuus else flush_periods).append((start, end))

        return flush_periods, infuus_periods

    flush_abp, infuus_abp = _detect_high(abp, *abp_phys, scan_abp)
    flush_cvp, infuus_cvp = _detect_high(cvp, *cvp_phys, scan_cvp)
    return flush_abp, infuus_abp, flush_cvp, infuus_cvp


# ── step 3 : transducer hoog ──────────────────────────────────────────────────

def detect_transducer_hoog(abp, cvp, fs, excluded,
                            lp_cutoff, hp_cutoff,
                            lp_drop_ratio=0.88,
                            hp_lo_ratio=0.3,
                            hp_hi_ratio=2.5,
                            min_event_s=2.0):
    """
    Detect transducer-too-high artefacts.

    Characteristic: ABP LP baseline drops to ≤ lp_drop_ratio × clean median
    while cardiac oscillations (HP std) continue at roughly normal amplitude.

    Algorithm
    ---------
    1. Compute LP-filtered ABP; estimate clean LP median as baseline.
    2. Mark samples below lp_drop_ratio × baseline (requires a genuine LP drop,
       not just normal beat-to-beat variation).
    3. Merge runs separated by less than one cardiac period.
    4. Keep runs ≥ min_event_s; reject via HP-std guard.

    Parameters
    ----------
    lp_drop_ratio : LP must drop to this fraction of the clean median to
                    trigger (default 0.80 — requires a 20 % drop).
    hp_lo_ratio   : reject if mean HP std < hp_lo_ratio × baseline
                    (oscillations absent → likely calibration)
    hp_hi_ratio   : reject if mean HP std > hp_hi_ratio × baseline
                    (oscillations greatly elevated → likely slinger)
    min_event_s   : minimum accepted event duration (s)

    Returns
    -------
    list of (start, end) tuples
    """
    n        = len(abp)
    clean    = ~excluded
    min_samp = max(int(min_event_s * fs), 1)

    lp_abp = lowpass(abp,  fs, cutoff_hz=lp_cutoff)
    hp_abp = highpass(abp, fs, cutoff_hz=hp_cutoff)
    hp_std = rolling_std(hp_abp, fs, window_s=1.0)

    # Use the 90th percentile of clean LP as the baseline.
    # Transducer hoog can only push LP *down*, so the upper tail of the
    # distribution always represents the true normal level — even if the
    # artefact covers the majority of the recording (up to ~90 %).
    # The median would be pulled toward the artefact level in those cases.
    lp_baseline = (float(np.percentile(lp_abp[clean], 90)) if clean.any()
                   else float(np.percentile(lp_abp, 90)))
    threshold   = lp_drop_ratio * lp_baseline

    valid_std   = clean & ~np.isnan(hp_std)
    hp_std_base = (float(np.median(hp_std[valid_std])) if valid_std.any()
                   else float(np.nanmedian(hp_std)))

    period_s = _estimate_cardiac_period(abp, clean, fs)
    gap_samp = max(int(period_s * fs), 1)

    # ── find runs below threshold ─────────────────────────────────────────────
    below = (lp_abp < threshold) & clean

    raw_runs = []
    i = 0
    while i < n:
        if below[i]:
            j = i + 1
            while j < n and below[j]:
                j += 1
            raw_runs.append([i, j])
            i = j
        else:
            i += 1

    # ── merge runs with short gaps (up to one cardiac period) ────────────────
    merged = []
    for run in raw_runs:
        if merged and run[0] - merged[-1][1] <= gap_samp:
            merged[-1][1] = run[1]
        else:
            merged.append(run)

    # ── duration filter and HP std guard ─────────────────────────────────────
    periods = []
    for start, end in merged:
        if end - start < min_samp:
            continue

        seg_std  = hp_std[start:end]
        seg_std  = seg_std[~np.isnan(seg_std)]
        mean_std = float(seg_std.mean()) if seg_std.size > 0 else 0.0
        ratio    = mean_std / hp_std_base if hp_std_base > 0 else 0.0

        if ratio < hp_lo_ratio or ratio > hp_hi_ratio:
            continue

        periods.append((start, end))

    return periods



# ── step 4 : gas bubble ───────────────────────────────────────────────────────

def detect_gasbubble(abp, cvp, fs, mask,
                     lp_cutoff=0.5,
                     hp_cutoff=1.0,
                     std_reduction_ratio=0.5,
                     lp_stability_threshold=0.15,
                     min_event_s=3.0,
                     cardiac_fmin=0.5,
                     cardiac_fmax=3.0):
    """
    Detects gas bubble (damping) artefacts per channel: a sustained drop in
    pulse pressure (HP std) while the LP baseline remains stable.

    Detection runs independently on ABP and CVP so that damping visible on
    only one channel is not incorrectly attributed to the other.

    Returns
    -------
    gasbubble_abp, gasbubble_cvp : two lists of (start_idx, end_idx) tuples
    """
    n     = len(abp)
    clean = ~mask
    if np.sum(clean) < int(10 * fs):
        return [], []

    # Cardiac period estimated from ABP (more reliable), shared by both channels.
    period_s = _estimate_cardiac_period(abp, clean, fs,
                                         fmin=cardiac_fmin, fmax=cardiac_fmax)
    window   = max(int(period_s * fs), 1)
    min_samp = max(int(min_event_s * fs), 1)

    def _detect_one(signal):
        lp       = lowpass(signal,  fs, cutoff_hz=lp_cutoff)
        hp       = highpass(signal, fs, cutoff_hz=hp_cutoff)

        baseline_lp_mean = float(np.mean(lp[clean]))
        hp_std_r         = rolling_std(hp, fs, window_s=1.0)
        valid_std        = clean & ~np.isnan(hp_std_r)
        baseline_hp_std  = float(np.median(hp_std_r[valid_std]) if valid_std.any()
                                 else np.nanmedian(hp_std_r))

        if baseline_hp_std == 0:
            return []

        # LP stability is NOT checked sample-by-sample: MAP can drift over a
        # long damping event, creating repeated gaps that split one period into
        # many.  LP is checked as a guard over the full detected period instead.
        is_damped = hp_std_r < (baseline_hp_std * std_reduction_ratio)
        candidate = is_damped & clean

        def _win_frac(idx):
            seg = candidate[idx : min(idx + window, n)]
            return float(seg.mean()) if len(seg) > 0 else 0.0

        # ── scan: entry and exit both confirmed over one cardiac period ───────
        raw_periods = []
        i = 0
        while i < n:
            if candidate[i] and _win_frac(i) >= 0.5:
                start = i
                j = i + 1
                while j < n:
                    if not candidate[j]:
                        if _win_frac(j) < 0.5:
                            if j - start >= min_samp:
                                raw_periods.append((start, j))
                            i = j
                            break
                        # Transient gap — skip past this non-candidate run
                        k = j + 1
                        while k < n and not candidate[k]:
                            k += 1
                        j = k
                    else:
                        j += 1
                else:
                    if n - start >= min_samp:
                        raw_periods.append((start, n))
                    i = n
            else:
                i += 1

        # ── LP stability guard ────────────────────────────────────────────────
        periods = []
        for start, end in raw_periods:
            mean_lp = float(np.mean(lp[start:end]))
            lp_dev  = abs(mean_lp - baseline_lp_mean) / baseline_lp_mean
            if lp_dev <= lp_stability_threshold:
                periods.append((start, end))

        return periods

    gasbubble_abp = _detect_one(abp)
    gasbubble_cvp = _detect_one(cvp)
    return gasbubble_abp, gasbubble_cvp

# ── step 5 : slinger (last resort — only runs when no other artefacts remain) ──

def _rolling_mean(x: np.ndarray, win: int) -> np.ndarray:
    return (
        pd.Series(x)
        .rolling(win, center=True, min_periods=max(1, win // 3))
        .mean()
        .to_numpy()
    )


def _rolling_median(x: np.ndarray, win: int) -> np.ndarray:
    return (
        pd.Series(x)
        .rolling(win, center=True, min_periods=max(1, win // 3))
        .median()
        .to_numpy()
    )


def _mask_to_periods(mask: np.ndarray, min_region_s: float,
                     merge_gap_s: float, fs: float) -> list:
    """
    Convert a boolean sample mask to a list of (start_idx, end_idx) tuples.

    Small False gaps between True runs are filled (merge_gap_s), then runs
    shorter than min_region_s are removed.
    """
    mask = mask.copy()
    n          = len(mask)
    merge_samp = max(1, int(round(merge_gap_s * fs)))
    min_samp   = max(1, int(round(min_region_s * fs)))

    # Fill short gaps
    i = 0
    while i < n:
        if not mask[i]:
            j = i
            while j < n and not mask[j]:
                j += 1
            if i > 0 and j < n and (j - i) <= merge_samp:
                mask[i:j] = True
            i = j
        else:
            i += 1

    # Remove short runs
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if (j - i) < min_samp:
                mask[i:j] = False
            i = j
        else:
            i += 1

    # Collect (start, end) pairs
    periods = []
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            periods.append((i, j))
            i = j
        else:
            i += 1
    return periods


def detect_slinger(abp, cvp, fs, excluded,
                   # ABP channel settings
                   abp_baseline_window_s=0.25,
                   abp_feature_window_s=0.50,
                   abp_trend_window_s=1.00,
                   abp_z_threshold=4.0,
                   abp_score_threshold=10.0,
                   abp_min_region_s=0.40,
                   abp_merge_gap_s=1.20,
                   # CVP channel settings
                   cvp_baseline_window_s=0.25,
                   cvp_feature_window_s=0.60,
                   cvp_trend_window_s=1.20,
                   cvp_z_threshold=3.5,
                   cvp_score_threshold=8.0,
                   cvp_min_region_s=0.30,
                   cvp_merge_gap_s=0.80):
    """
    Detect resonance / ringing artefacts (slinger), separately per channel.

    Uses a time-domain approach: robust z-scores of residual RMS (signal minus
    local median baseline), absolute slope, and trend deviation are combined
    into a detection score.  Regions where the score exceeds the threshold are
    flagged as slinger.  All statistics are computed from non-excluded samples
    only, so previously detected artefacts do not bias the detection.

    Algorithm (per channel)
    -----------------------
    1. Baseline  = rolling median (short window) → captures the local mean.
    2. Residual  = signal − baseline → isolates fast oscillations.
    3. Features  = residual RMS, absolute slope, and trend deviation, each
                   smoothed over feature_window_s.
    4. z-scores  = robust (median/MAD) z-scores computed on the clean (non-
                   excluded) samples only, then clipped to zero.
    5. Score     = 1.0×z_residual + 1.0×z_slope + 0.4×z_deviation.
    6. Mask      = (z_residual > z_threshold AND z_slope > z_threshold)
                   OR score > score_threshold, restricted to clean samples.
    7. Post-processing: merge short gaps (merge_gap_s), drop short regions
                   (min_region_s).

    Returns
    -------
    slinger_abp, slinger_cvp : lists of (start_idx, end_idx) sample-index tuples
    diag_abp, diag_cvp       : dicts with 'ts', 'hf_ratio' (z_residual),
                               'hf_smooth' (combined score), 'threshold'
    """
    n     = len(abp)
    clean = ~excluded
    t     = np.arange(n, dtype=float) / fs

    def _detect_one(signal, ch_name, cfg):
        # Odd window sizes in samples
        def _win(seconds):
            w = max(5, int(round(seconds * fs)))
            return w if w % 2 == 1 else w + 1

        baseline_win = _win(cfg["baseline_window_s"])
        feature_win  = _win(cfg["feature_window_s"])
        trend_win    = _win(cfg["trend_window_s"])

        baseline   = _rolling_median(signal, baseline_win)
        slow_trend = _rolling_mean(signal,   trend_win)
        residual   = signal - baseline
        slope      = np.abs(np.gradient(signal, t))

        residual_rms = np.sqrt(np.maximum(
            _rolling_mean(residual ** 2, feature_win), 0.0))
        slope_mean   = _rolling_mean(slope, feature_win)
        trend_dev    = np.abs(signal - slow_trend)

        # Robust z-scores: statistics derived from clean samples only
        def _zsc(x):
            ref = x[clean]
            if ref.size == 0:
                return np.zeros(n)
            med = float(np.nanmedian(ref))
            mad = float(np.nanmedian(np.abs(ref - med)))
            if not np.isfinite(mad) or mad < 1e-12:
                return np.zeros(n)
            return np.clip(0.6745 * (x - med) / mad, 0.0, None)

        z_residual  = _zsc(residual_rms)
        z_slope     = _zsc(slope_mean)
        z_deviation = _zsc(trend_dev)
        score       = 1.0 * z_residual + 1.0 * z_slope + 0.4 * z_deviation

        z_thr = cfg["z_threshold"]
        s_thr = cfg["score_threshold"]

        # Floor: threshold must be at least 15 % of the score range on clean
        # samples, so that a uniformly quiet signal cannot produce a threshold
        # that sits at or below the score median and flag half the recording.
        if clean.any():
            clean_score = score[clean]
            score_range = float(clean_score.max() - clean_score.min())
            score_floor = 0.15 * score_range
            if score_floor > s_thr:
                s_thr = score_floor

        raw_mask = (
            ((z_residual > z_thr) & (z_slope > z_thr))
            | (score > s_thr)
        ) & clean

        # Suppress detections within one trend-window of each boundary.
        # Rolling statistics have fewer neighbours there (even with min_periods)
        # and np.gradient uses one-sided differences at the edges, both of which
        # can produce spurious high scores.
        margin = trend_win
        raw_mask[:margin]  = False
        raw_mask[-margin:] = False

        periods = _mask_to_periods(
            raw_mask, cfg["min_region_s"], cfg["merge_gap_s"], fs)

        diag = {"ts": t, "hf_ratio": z_residual, "hf_smooth": score,
                "threshold": s_thr}
        return periods, diag

    abp_cfg = dict(
        baseline_window_s=abp_baseline_window_s,
        feature_window_s=abp_feature_window_s,
        trend_window_s=abp_trend_window_s,
        z_threshold=abp_z_threshold,
        score_threshold=abp_score_threshold,
        min_region_s=abp_min_region_s,
        merge_gap_s=abp_merge_gap_s,
    )
    cvp_cfg = dict(
        baseline_window_s=cvp_baseline_window_s,
        feature_window_s=cvp_feature_window_s,
        trend_window_s=cvp_trend_window_s,
        z_threshold=cvp_z_threshold,
        score_threshold=cvp_score_threshold,
        min_region_s=cvp_min_region_s,
        merge_gap_s=cvp_merge_gap_s,
    )

    slinger_abp, diag_abp = _detect_one(abp, "abp", abp_cfg)
    slinger_cvp, diag_cvp = _detect_one(cvp, "cvp", cvp_cfg)
    return slinger_abp, slinger_cvp, diag_abp, diag_cvp