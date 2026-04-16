"""
detection.py  —  artefact detection functions

Each function receives the raw ABP and CVP arrays, the sample rate, a boolean
exclusion mask (True = already excluded), and tuning parameters.  It returns a
list of (start_idx, end_idx) tuples using numpy-slice (end-exclusive) convention.

Baselines must always be derived from the non-excluded portion of the signal so
that previously detected artefacts do not bias the thresholds.
"""

import numpy as np
from scipy.signal import spectrogram as _spectrogram, butter, filtfilt
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
    print(f"  [calibration] cardiac period={period_s:.2f} s  window={window} smp")

    def _detect_cal(signal, ch_name, phys_lo, phys_hi, scan_clean):
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

        print(f"    [{ch_name}] LP mean={lp_mean:.1f} mmHg  cal_thr={cal_thr:.2f} mmHg  "
              f"HP std base={hp_base:.2f} mmHg  max={hp_max_ratio:.1f}×")

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
            if ratio > hp_max_ratio:
                print(f"    [{ch_name}] cal {start/fs:.1f}–{end/fs:.1f} s rejected: "
                      f"HP std median {median_hp:.2f} ({ratio:.2f}x base) — oscillations "
                      f"continue, likely transducer hoog")
            else:
                periods.append((start, end))

        print(f"    [{ch_name}] cal: {len(periods)} period(s)  "
              f"({len(raw_periods) - len(periods)} rejected by HP guard)")
        return periods

    cal_abp = _detect_cal(abp, "ABP", *abp_phys, scan_abp)
    cal_cvp = _detect_cal(cvp, "CVP", *cvp_phys, scan_cvp)
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
    print(f"  [flush/infuus] cardiac period={period_s:.2f} s  window={window} smp  "
          f"min_flush={min_flush_cycles} cycle(s)={min_flush_samp} smp")

    def _detect_high(signal, ch_name, phys_lo, phys_hi, scan_clean):
        lp        = lowpass(signal, fs, cutoff_hz=lp_cutoff)
        phys_mask = clean & (lp >= phys_lo) & (lp <= phys_hi)
        lp_mean   = (np.mean(lp[phys_mask]) if phys_mask.any() else
                     np.mean(lp[clean])      if clean.any()     else np.mean(lp))
        flush_thr = flush_threshold * lp_mean
        n         = len(signal)

        # HP signal for oscillation classification
        hp          = highpass(signal, fs, cutoff_hz=hp_cutoff)
        hp_std_roll = rolling_std(hp, fs, window_s=1.0)

        print(f"    [{ch_name}] LP mean={lp_mean:.1f} mmHg  "
              f"flush_thr={flush_thr:.1f} mmHg  "
              f"infuus: dur≥{infuus_min_s:.0f}s AND HP≥{infuus_hp_ratio:.1f}×base")

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
        print(f"    [{ch_name}] HP std baseline={hp_std_base:.2f} mmHg")

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
            label = "infuus" if is_infuus else "flush"
            (infuus_periods if is_infuus else flush_periods).append((start, end))
            print(f"    [{ch_name}] → {label:6s}  {start/fs:.1f}–{end/fs:.1f} s  "
                  f"dur={duration_s:.1f}s  HP std={mean_hp_std:.2f} "
                  f"({hp_ratio:.2f}× baseline)")

        print(f"    [{ch_name}] flush: {len(flush_periods)}  infuus: {len(infuus_periods)}")
        return flush_periods, infuus_periods

    flush_abp, infuus_abp = _detect_high(abp, "ABP", *abp_phys, scan_abp)
    flush_cvp, infuus_cvp = _detect_high(cvp, "CVP", *cvp_phys, scan_cvp)
    return flush_abp, infuus_abp, flush_cvp, infuus_cvp


# ── step 3 : transducer hoog ──────────────────────────────────────────────────

def detect_transducer_hoog(abp, cvp, fs, excluded,
                            lp_cutoff, hp_cutoff,
                            lp_drop_ratio=0.85,
                            lp_exit_ratio=0.9,
                            hp_lo_ratio=0.3,
                            hp_hi_ratio=2.5,
                            cardiac_fmin=0.5,
                            cardiac_fmax=3.0,
                            min_event_s=2.0,
                            tail_guard_s=3.0):
    """
    Detect transducer-too-high artefacts.

    Characteristic: the ABP LP baseline drops (transducer raised above the
    phlebostatic axis → hydrostatic under-read) while cardiac oscillations
    on the HP signal continue at roughly their normal amplitude.

    Detection
    ---------
    The LP-filtered ABP is scanned for a sustained drop below
    lp_drop_ratio × the clean LP median.  Entry and exit are confirmed
    with the same window-mean approach as detect_calibration and
    detect_flush_infuus: the mean of the LP over the next cardiac period
    must also cross the threshold before the transition is registered.
    Transient recoveries that are not window-confirmed are skipped.

    HP std guard
    ------------
    After detection the mean rolling HP std inside each period must lie
    within [hp_lo_ratio, hp_hi_ratio] × the clean HP std baseline:
      • below hp_lo_ratio → oscillations absent; likely calibration residue
      • above hp_hi_ratio → oscillations greatly elevated; likely slinger

    Tail guard
    ----------
    A genuine transducer artefact must show a confirmed LP recovery before
    the recording ends.  Any candidate whose exit falls within tail_guard_s
    of the end of the signal is rejected (no confirmed recovery observed).

    Hysteresis
    ----------
    Entry and exit use different thresholds to prevent the detector from
    toggling off and on when the LP hovers near the boundary:
      entry : LP drops below  lp_drop_ratio × baseline  (default 80 %)
      exit  : LP rises above  lp_exit_ratio × baseline  (default 90 %)
    The exit threshold must be higher than the entry threshold.  A genuine
    recovery from a transducer-hoog event requires the LP to return well
    toward the normal baseline, not merely cross the entry line.

    Returns
    -------
    list of (start, end) tuples
    """
    print("    [transducer] running transducer-hoog detection")

    n          = len(abp)
    clean      = ~excluded
    min_samp   = max(int(min_event_s * fs), 1)
    tail_guard = max(int(tail_guard_s * fs), 0)

    lp_abp = lowpass(abp,  fs, cutoff_hz=lp_cutoff)
    hp_abp = highpass(abp, fs, cutoff_hz=hp_cutoff)
    hp_std = rolling_std(hp_abp, fs, window_s=1.0)

    lp_baseline = (float(np.median(lp_abp[clean])) if clean.any()
                   else float(np.median(lp_abp)))
    valid_std   = clean & ~np.isnan(hp_std)
    hp_std_base = (float(np.median(hp_std[valid_std])) if valid_std.any()
                   else float(np.nanmedian(hp_std)))
    entry_thr = lp_drop_ratio * lp_baseline
    exit_thr  = lp_exit_ratio * lp_baseline

    period_s = _estimate_cardiac_period(abp, clean, fs,
                                         fmin=cardiac_fmin, fmax=cardiac_fmax)
    window = max(int(period_s * fs), 1)

    print(f"    [transducer] LP baseline={lp_baseline:.1f} mmHg  "
          f"entry<{entry_thr:.1f} ({lp_drop_ratio:.0%})  "
          f"exit>{exit_thr:.1f} ({lp_exit_ratio:.0%})")
    print(f"    [transducer] HP-std baseline={hp_std_base:.2f} mmHg  "
          f"accept range: [{hp_lo_ratio:.1f}x, {hp_hi_ratio:.1f}x]")
    print(f"    [transducer] cardiac period={period_s:.2f} s  window={window} smp")

    def _win_mean_lp(idx):
        return float(np.mean(lp_abp[idx : min(idx + window, n)]))

    # ── scan LP for sustained drops ───────────────────────────────────────────
    # Entry fires when LP drops below entry_thr; exit requires LP to recover
    # above exit_thr (hysteresis).  A signal that hovers between the two
    # thresholds is treated as still inside the period.
    raw_periods = []
    i = 0
    while i < n:
        if lp_abp[i] < entry_thr and not excluded[i]:
            if _win_mean_lp(i) < entry_thr:
                start = i
                j = i + 1
                while j < n:
                    if lp_abp[j] >= exit_thr:
                        if _win_mean_lp(j) >= exit_thr:
                            # Confirmed exit — apply tail guard
                            if j >= n - tail_guard:
                                print(f"    [transducer] {start/fs:.1f}–{j/fs:.1f} s: "
                                      f"exit within tail guard — rejected")
                                i = j
                                break
                            if j - start >= min_samp:
                                raw_periods.append((start, j))
                            i = j
                            break
                        # Transient LP recovery above exit_thr — skip past it
                        k = j + 1
                        while k < n and lp_abp[k] >= exit_thr:
                            k += 1
                        j = k
                    else:
                        j += 1
                else:
                    # Reached end without confirmed recovery
                    print(f"    [transducer] {start/fs:.1f}–end: "
                          f"no confirmed LP recovery — rejected")
                    i = n
            else:
                i += 1
        else:
            i += 1

    # ── HP std guard ──────────────────────────────────────────────────────────
    periods = []
    for start, end in raw_periods:
        seg_std  = hp_std[start:end]
        seg_std  = seg_std[~np.isnan(seg_std)]
        mean_std = float(seg_std.mean()) if seg_std.size > 0 else 0.0
        ratio    = mean_std / hp_std_base if hp_std_base > 0 else 0.0

        if ratio < hp_lo_ratio:
            print(f"    [transducer] {start/fs:.1f}–{end/fs:.1f} s: "
                  f"oscillations absent (HP std {mean_std:.2f}, {ratio:.2f}× base) "
                  f"— not transducer hoog")
            continue
        if ratio > hp_hi_ratio:
            print(f"    [transducer] {start/fs:.1f}–{end/fs:.1f} s: "
                  f"oscillations greatly elevated (HP std {mean_std:.2f}, {ratio:.2f}× base) "
                  f"— likely slinger")
            continue

        mean_lp = float(lp_abp[start:end].mean())
        periods.append((start, end))
        print(f"    [transducer] period  {start/fs:.1f}–{end/fs:.1f} s  "
              f"(LP {mean_lp:.1f} mmHg  HP std {mean_std:.2f} ({ratio:.2f}× base))")

    print(f"    [transducer] result: {len(periods)} period(s)")
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
    Detects gas bubble (damping) artefacts: a sustained drop in pulse pressure
    (HP std) while the LP baseline remains stable.

    Entry is confirmed when both conditions hold over a full cardiac period.
    Exit is confirmed the same way — brief transients within a damped period
    do not break it prematurely.
    """
    n = len(abp)
    clean = ~mask
    if np.sum(clean) < int(10 * fs):
        return []

    # ── filtered signals ──────────────────────────────────────────────────────
    lp = lowpass(abp,  fs, cutoff_hz=lp_cutoff)
    hp = highpass(abp, fs, cutoff_hz=hp_cutoff)

    # ── baselines from clean samples ──────────────────────────────────────────
    baseline_lp_mean = float(np.mean(lp[clean]))
    hp_std_r         = rolling_std(hp, fs, window_s=1.0)
    valid_std        = clean & ~np.isnan(hp_std_r)
    baseline_hp_std  = float(np.median(hp_std_r[valid_std]) if valid_std.any()
                             else np.nanmedian(hp_std_r))

    if baseline_hp_std == 0:
        return []

    # ── per-sample damping mask ───────────────────────────────────────────────
    # LP stability is NOT checked sample-by-sample: the patient's MAP can drift
    # over a long damping event, creating repeated gaps that split one period
    # into many.  LP is checked as a guard over the full detected period instead.
    is_damped = hp_std_r < (baseline_hp_std * std_reduction_ratio)
    candidate = is_damped & clean

    print(f"    [gasbubble] LP baseline={baseline_lp_mean:.1f} mmHg  "
          f"HP-std baseline={baseline_hp_std:.2f} mmHg  "
          f"damping thr={baseline_hp_std * std_reduction_ratio:.2f} mmHg")

    # ── confirmation window (one cardiac period) ──────────────────────────────
    period_s = _estimate_cardiac_period(abp, clean, fs,
                                         fmin=cardiac_fmin, fmax=cardiac_fmax)
    window   = max(int(period_s * fs), 1)
    min_samp = max(int(min_event_s * fs), 1)

    def _win_frac(idx):
        seg = candidate[idx : min(idx + window, n)]
        return float(seg.mean()) if len(seg) > 0 else 0.0

    # ── scan: entry and exit both confirmed over one cardiac period ───────────
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

    # ── LP stability guard ────────────────────────────────────────────────────
    # Reject periods where the mean LP over the whole period deviates too far
    # from baseline — distinguishes gas bubble (stable MAP) from transducer
    # hoog (LP drops) or flush (LP spikes).
    periods = []
    for start, end in raw_periods:
        mean_lp = float(np.mean(lp[start:end]))
        lp_dev  = abs(mean_lp - baseline_lp_mean) / baseline_lp_mean
        if lp_dev > lp_stability_threshold:
            print(f"    [gasbubble] {start/fs:.1f}–{end/fs:.1f} s rejected: "
                  f"LP mean {mean_lp:.1f} mmHg deviates {lp_dev:.0%} from baseline")
        else:
            periods.append((start, end))
            print(f"    [gasbubble] period  {start/fs:.1f}–{end/fs:.1f} s  "
                  f"({(end-start)/fs:.1f} s)")

    print(f"    [gasbubble] result: {len(periods)} period(s)")
    return periods

# ── step 5 : slinger (last resort — only runs when no other artefacts remain) ──

def detect_slinger(abp, cvp, fs, excluded,
                   hf_low=8.0,
                   hf_high=20.0,
                   baseline_percentile=40,
                   k=18.0,
                   smooth_window=5,
                   fill_gap_bins=2,
                   min_event_s=1.0):
    """
    Detect resonance / ringing artefacts (slinger), separately per channel.

    Characteristic: a burst of high-frequency energy (hf_low–hf_high Hz) that
    stands out above the baseline HF level.

    For each channel:
    1. Compute the spectrogram (1 s window, 0.5 s overlap, 2 s nfft) with
       reflect-padding to avoid edge artefacts.
    2. Sum power in the HF band; log-transform then smooth with a rolling mean.
    3. Estimate the baseline from non-excluded bins at the bottom
       baseline_percentile of the distribution (robust when slinger fills most
       of the recording).
    4. Threshold = baseline median + k × baseline std.
    5. Fill small inter-bin gaps (fill_gap_bins), mask excluded bins, then keep
       runs that meet min_event_s.

    Parameters
    ----------
    hf_low / hf_high    : HF band limits for ringing energy (Hz)
    baseline_percentile : upper percentile cut for baseline estimation
    k                   : threshold multiplier  (baseline_median + k × std)
    smooth_window       : rolling-mean window in spectrogram bins
    fill_gap_bins       : dilation width to bridge short inter-segment gaps
    min_event_s         : minimum accepted event duration (s)

    Returns
    -------
    slinger_abp, slinger_cvp : two lists of (start_idx, end_idx) tuples
    """
    n        = len(abp)
    nperseg  = max(int(1.0 * fs), 32)
    noverlap = int(0.5 * fs)
    nfft     = int(2.0 * fs)
    pad      = nperseg // 2
    half_win_s = nperseg / (2.0 * fs)

    def _detect_one(signal, ch_name):
        print(f"    [slinger/{ch_name}] running slinger detection")

        # ── spectrogram with reflect-padding to suppress edge effects ─────────
        sig_pad = np.pad(signal, pad_width=pad, mode="reflect")
        f, ts_pad, Sxx = _spectrogram(
            sig_pad, fs=fs, nperseg=nperseg, noverlap=noverlap,
            nfft=nfft, scaling="density",
        )
        ts = ts_pad - pad / fs

        # Trim to bins fully inside the original signal
        duration = (n - 1) / fs
        valid    = (ts >= half_win_s) & (ts <= duration - half_win_s)
        ts  = ts[valid]
        Sxx = Sxx[:, valid]

        # ── HF band energy: log-transformed and smoothed ──────────────────────
        hf_band   = (f >= hf_low) & (f <= hf_high)
        hf_energy = np.sum(Sxx[hf_band, :], axis=0)
        hf_log    = np.log1p(hf_energy)

        if smooth_window > 1:
            kernel    = np.ones(smooth_window) / smooth_window
            hf_smooth = np.convolve(hf_log, kernel, mode="same")
        else:
            hf_smooth = hf_log.copy()

        # ── classify bins: clean (baseline) vs scan (detection) ──────────────
        def _excl_frac(ts_i):
            s = max(0, int((ts_i - half_win_s) * fs))
            e = min(n, int((ts_i + half_win_s) * fs))
            return float(excluded[s:e].mean()) if e > s else 1.0

        excl_frac  = np.array([_excl_frac(ti) for ti in ts])
        clean_bins = excl_frac < 0.5   # for baseline estimation
        scan_bins  = excl_frac < 0.1   # for candidate detection

        # ── baseline and threshold ────────────────────────────────────────────
        ref      = hf_smooth[clean_bins] if clean_bins.any() else hf_smooth
        bl_limit = np.percentile(ref, baseline_percentile)
        bl       = ref[ref <= bl_limit]
        med      = float(np.median(bl))
        std      = float(np.std(bl))
        thr      = med + k * std

        print(f"    [slinger/{ch_name}] HF {hf_low:.0f}-{hf_high:.0f} Hz  "
              f"baseline med={med:.3f}  std={std:.4f}  thr={thr:.3f}  (k={k})")

        # ── candidate bins ────────────────────────────────────────────────────
        candidate = hf_smooth > thr

        # Fill small gaps between candidate bins
        if fill_gap_bins > 1:
            half_d = fill_gap_bins // 2
            expanded = candidate.copy()
            for shift in range(1, half_d + 1):
                expanded[shift:]  |= candidate[:-shift]
                expanded[:-shift] |= candidate[shift:]
            candidate = expanded

        # Mask out excluded bins (fill before masking so gaps don't bleed in)
        candidate = candidate & scan_bins

        # ── find contiguous segments above threshold ───────────────────────────
        periods = []
        i = 0
        m = len(ts)
        while i < m:
            if candidate[i]:
                j = i + 1
                while j < m and candidate[j]:
                    j += 1
                # Convert spectrogram bin centres to sample indices
                s_samp = max(0, int(round(ts[i]     * fs)))
                e_samp = min(n, int(round(ts[j - 1] * fs)) + 1)
                dur_s  = (e_samp - s_samp) / fs
                if dur_s >= min_event_s:
                    periods.append((s_samp, e_samp))
                    print(f"    [slinger/{ch_name}] period  "
                          f"{s_samp/fs:.1f}-{e_samp/fs:.1f} s  "
                          f"({dur_s:.1f} s  bins={j - i})")
                i = j
            else:
                i += 1

        print(f"    [slinger/{ch_name}] result: {len(periods)} period(s)")
        diag = {"ts": ts, "hf_log": hf_log, "hf_smooth": hf_smooth, "threshold": thr}
        return periods, diag

    slinger_abp, diag_abp = _detect_one(abp, "abp")
    slinger_cvp, diag_cvp = _detect_one(cvp, "cvp")
    return slinger_abp, slinger_cvp, diag_abp, diag_cvp