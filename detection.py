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
                       cvp_phys=(3,  20),
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
        periods = []
        for start, end in raw_periods:
            seg      = hp_std_r[start:end]
            seg      = seg[~np.isnan(seg)]
            mean_hp  = float(seg.mean()) if seg.size > 0 else 0.0
            ratio    = mean_hp / hp_base if hp_base > 0 else 0.0
            if ratio > hp_max_ratio:
                print(f"    [{ch_name}] cal {start/fs:.1f}–{end/fs:.1f} s rejected: "
                      f"HP std {mean_hp:.2f} ({ratio:.2f}× base) — oscillations "
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
                        cvp_phys=(3,  20),
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


# ── step 4 : slinger ───────────────────────────────────────────────────────────

def detect_slinger(abp, cvp, fs, excluded,
                   lp_cutoff, hp_cutoff, spec_fmax, k,
                   # ── sensitivity / bound parameters ────────────────────────
                   ringing_k=2.0,
                   high_band_factor=1.5,
                   cardiac_fmin=0.5,
                   cardiac_fmax=2.0,
                   min_iqr_floor=0.05,
                   lp_drop_ratio=0.6,
                   lp_spike_offset=15.0,
                   min_event_s=2.0,
                   spec_overlap=0.75,
                   boundary_hp_ratio=1.5,
                   min_ringy_windows=2,
                   confirm_hp_ratio=1.1):
    """
    Detect resonance / ringing artefacts (slinger), separately per channel.

    Characteristic: a sudden burst of high-frequency content above a multiple
    of the dominant cardiac frequency, faster oscillation periods.  Mean
    arterial pressure (LP) stays near normal.

    ABP and CVP are detected independently because a slinger in one line (e.g.
    catheter resonance in the arterial tubing) does not necessarily affect the
    other channel.  Both channels use the same algorithm and parameters.

    Parameters
    ----------
    abp, cvp          : 1D float arrays
    fs                : sample rate (Hz)
    excluded          : bool array, True = already excluded
    lp_cutoff         : low-pass cutoff (Hz)
    hp_cutoff         : high-pass cutoff (Hz)
    spec_fmax         : upper frequency limit for spectrogram (Hz)
    k                 : global outlier multiplier (not used for ringing_ratio)
    ringing_k         : IQR multiplier for the ringing_ratio threshold;
                        kept separate from k so slinger sensitivity can be
                        tuned independently.  Lower = more sensitive.
    high_band_factor  : ringing power = spectrogram power above
                        high_band_factor × f_dom.  Lower = more sensitive
                        (captures ringing closer to the cardiac frequency).
    cardiac_fmin      : lower bound (Hz) of the cardiac frequency search band
    cardiac_fmax      : upper bound (Hz) of the cardiac frequency search band.
                        Keep at or below 2.0 Hz (120 bpm) so slinger ringing
                        frequencies are not mistaken for the cardiac peak.
    min_iqr_floor     : minimum scale for the ringing_ratio threshold; prevents
                        the threshold from collapsing when slinger occupies more
                        than half the recording and the lower-half IQR is tiny.
    lp_drop_ratio     : reject candidate if LP mean < lp_drop_ratio × LP baseline
                        (indicates transducer repositioning, not slinger)
    lp_spike_offset   : reject candidate if LP mean > LP baseline + this value (mmHg)
                        (indicates flush, not slinger)
    min_event_s       : minimum accepted event duration (s)
    spec_overlap      : fractional overlap between successive spectrogram windows
    boundary_hp_ratio : HP-std threshold (× baseline) used to refine rough start/end
                        boundaries; a causal-lag correction of window_s/2 is applied
                        to the start boundary.  Increase to widen, decrease to narrow.
    min_ringy_windows : minimum number of consecutive ringy spectrogram windows to
                        form a candidate; filters single-window marginal noise.
    confirm_hp_ratio  : HP-std confirmation threshold (× baseline); candidate is
                        rejected if the mean HP std over the rough period is below
                        this ratio.  Set low (e.g. 1.1) so genuine slingers
                        (HP std typically 2× baseline) pass while noise (HP std at
                        or below baseline) is rejected.

    Returns
    -------
    slinger_abp, slinger_cvp : two lists of (start, end) tuples
    """
    n        = len(abp)
    clean    = ~excluded
    window_s = 2.0
    nperseg  = max(int(window_s * fs), 32)
    noverlap = int(nperseg * spec_overlap)
    min_dur  = max(int(min_event_s * fs), 1)

    def _detect_one(signal, ch_name):
        print(f"    [slinger/{ch_name}] running slinger detection")

        # ── spectrogram ───────────────────────────────────────────────────────
        freqs, times, Sxx = _spectrogram(
            signal, fs=fs, nperseg=nperseg, noverlap=noverlap,
            nfft=nperseg, scaling="density",
        )

        half_win = nperseg // 2
        def _ef(ti):
            s = max(0, int(ti * fs) - half_win)
            e = min(n, int(ti * fs) + half_win)
            return excluded[s:e].mean() if e > s else 1.0

        excl_frac_win = np.array([_ef(ti) for ti in times])
        clean_win = excl_frac_win < 0.5   # used for f_dom and rr baseline
        scan_win  = excl_frac_win < 0.1   # used for ringy flagging — near-zero overlap
                                           # with excluded regions prevents ghost re-detections

        # ── dominant cardiac frequency ─────────────────────────────────────────
        cardiac_band = (freqs >= cardiac_fmin) & (freqs <= cardiac_fmax)
        if clean_win.sum() >= 4 and cardiac_band.any():
            P_mean = Sxx[cardiac_band][:, clean_win].mean(axis=1)
            f_dom  = float(freqs[cardiac_band][np.argmax(P_mean)])
        else:
            f_dom = 1.2   # fallback: 72 bpm
        print(f"    [slinger/{ch_name}] dominant cardiac frequency: {f_dom:.2f} Hz  "
              f"ringing band: >{high_band_factor:.1f}xf_dom = "
              f">{high_band_factor * f_dom:.2f} Hz")

        # ── ringing ratio ──────────────────────────────────────────────────────
        high_band  = (freqs > high_band_factor * f_dom) & (freqs <= spec_fmax)
        total_band = (freqs >= 0.5) & (freqs <= spec_fmax)

        if not high_band.any():
            print(f"    [slinger/{ch_name}] no frequencies above "
                  f"{high_band_factor * f_dom:.1f} Hz in spectrogram — skipped")
            return []

        P_high  = Sxx[high_band,  :].sum(axis=0)
        P_total = Sxx[total_band, :].sum(axis=0)
        ringing_ratio = np.where(P_total > 0, P_high / P_total, 0.0)

        # ── outlier threshold ──────────────────────────────────────────────────
        # Use p25 as the location estimator and the IQR of the lower half of the
        # distribution as the scale estimator.  This is robust when slinger
        # occupies more than 50 % of the recording: in that case the global
        # median falls in the slinger zone and inflates the threshold to an
        # unreachable level.  The lower-half IQR captures only the spread within
        # the clean mode, clamped to min_iqr_floor so the threshold is never
        # trivially close to the baseline.
        rr_ref     = ringing_ratio[clean_win] if clean_win.sum() >= 4 else ringing_ratio
        rr_p25     = float(np.percentile(rr_ref, 25))
        rr_p50     = float(np.median(rr_ref))
        lower_half = rr_ref[rr_ref <= rr_p50]
        lh_iqr     = float(max(np.subtract(*np.percentile(lower_half, [75, 25])), 1e-6)
                           if lower_half.size >= 4
                           else max(rr_p50 - rr_p25, 1e-6))
        rr_thr  = rr_p25 + ringing_k * max(lh_iqr, min_iqr_floor)
        ringy   = (ringing_ratio > rr_thr) & scan_win

        print(f"    [slinger/{ch_name}] ringing_ratio  p25={rr_p25:.4f}  "
              f"lh_iqr={lh_iqr:.4f}  threshold={rr_thr:.4f}  "
              f"(ringing_k={ringing_k})  flagged windows={ringy.sum()}")

        # ── LP baseline and HP std ─────────────────────────────────────────────
        lp_sig      = lowpass(signal,  fs, cutoff_hz=lp_cutoff)
        hp_sig      = highpass(signal, fs, cutoff_hz=hp_cutoff)
        hp_std      = rolling_std(hp_sig, fs, window_s=1.0)
        lp_norm     = float(np.median(lp_sig[clean]) if clean.any() else np.median(lp_sig))
        valid_std   = clean & ~np.isnan(hp_std)
        hp_std_norm = (float(np.median(hp_std[valid_std])) if valid_std.any()
                       else float(np.nanmedian(hp_std)))

        print(f"    [slinger/{ch_name}] LP baseline: {lp_norm:.1f} mmHg  "
              f"HP-std baseline: {hp_std_norm:.2f} mmHg")

        # ── HP-std boundary refinement ─────────────────────────────────────────
        # rolling_std is causal: out[i] covers signal[i-window+1..i].  The start
        # boundary therefore lags the true onset by up to window_s/2; compensate
        # by shifting the refined start back by that amount.
        _hp_std_thr = hp_std_norm * boundary_hp_ratio
        _lag        = int(0.5 * fs)          # half the 1-s rolling-std window

        def _hp_boundaries(s_rough, e_rough):
            extend = int(window_s * fs)
            s_srch = max(0, s_rough - extend)
            e_srch = min(n, e_rough + extend)
            seg    = hp_std[s_srch:e_srch]
            above  = (seg > _hp_std_thr) & (~np.isnan(seg))
            if not above.any():
                return s_rough, e_rough
            idx   = np.where(above)[0]
            s_idx = max(0, s_srch + int(idx[0])  - _lag)   # lag-compensated start
            e_idx = s_srch + int(idx[-1]) + 1               # last above-threshold + 1
            return s_idx, e_idx

        # ── scan ringy runs ────────────────────────────────────────────────────
        periods = []
        i = 0
        while i < len(times):
            if ringy[i]:
                j = i + 1
                while j < len(times) and ringy[j]:
                    j += 1

                n_ringy = j - i
                s_rough = max(0, int((times[i]     - window_s / 2) * fs))
                e_rough = min(n, int((times[j - 1] + window_s / 2) * fs))

                # ── guard 1: require minimum run of consecutive ringy windows ──
                if n_ringy < min_ringy_windows:
                    print(f"    [slinger/{ch_name}] {s_rough/fs:.1f}-{e_rough/fs:.1f} s: "
                          f"only {n_ringy} ringy window(s) < min {min_ringy_windows} "
                          f"— likely noise, skipping")
                    i = j
                    continue

                if e_rough - s_rough >= min_dur:
                    lp_mean = float(lp_sig[s_rough:e_rough].mean())

                    if lp_mean < lp_norm * lp_drop_ratio:
                        print(f"    [slinger/{ch_name}] {s_rough/fs:.1f}-{e_rough/fs:.1f} s: "
                              f"LP dropped ({lp_mean:.1f} vs {lp_norm:.1f} mmHg) "
                              f"— not slinger")
                        i = j
                        continue
                    if lp_mean > lp_norm + lp_spike_offset:
                        print(f"    [slinger/{ch_name}] {s_rough/fs:.1f}-{e_rough/fs:.1f} s: "
                              f"LP elevated ({lp_mean:.1f} vs {lp_norm:.1f} mmHg) "
                              f"— not slinger")
                        i = j
                        continue

                    # ── guard 2: HP std must be elevated (lenient confirmation) ─
                    seg_std  = hp_std[s_rough:e_rough]
                    seg_std  = seg_std[~np.isnan(seg_std)]
                    mean_std = float(seg_std.mean()) if seg_std.size > 0 else 0.0
                    if mean_std < hp_std_norm * confirm_hp_ratio:
                        print(f"    [slinger/{ch_name}] {s_rough/fs:.1f}-{e_rough/fs:.1f} s: "
                              f"HP std {mean_std:.2f} below {confirm_hp_ratio:.1f}x "
                              f"baseline {hp_std_norm:.2f} — not slinger")
                        i = j
                        continue

                    s_final, e_final = _hp_boundaries(s_rough, e_rough)
                    seg_std2  = hp_std[s_final:e_final]
                    seg_std2  = seg_std2[~np.isnan(seg_std2)]
                    rr_mean   = float(ringing_ratio[i:j].mean())
                    mean_std2 = float(seg_std2.mean()) if seg_std2.size > 0 else float("nan")
                    periods.append((s_final, e_final))
                    print(f"    [slinger/{ch_name}] period  "
                          f"{s_final/fs:.1f}-{e_final/fs:.1f} s  "
                          f"(rough {s_rough/fs:.1f}-{e_rough/fs:.1f} s)  "
                          f"ringing_ratio {rr_mean:.3f}  HP std {mean_std2:.2f} mmHg")
                else:
                    print(f"    [slinger/{ch_name}] window at {times[i]:.1f} s "
                          f"too short — ignored")

                i = j
            else:
                i += 1

        # ── merge overlapping refined periods ──────────────────────────────────
        # _hp_boundaries can produce overlapping results for adjacent ringy runs
        # when the HP std is elevated throughout a single slinger event.  Merge
        # so the detector always returns non-overlapping, contiguous spans.
        if periods:
            n_before = len(periods)
            periods.sort(key=lambda p: p[0])
            merged = [list(periods[0])]
            for s, e in periods[1:]:
                if s <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])
            periods = [tuple(p) for p in merged]
            if len(periods) < n_before:
                print(f"    [slinger/{ch_name}] merged {n_before} overlapping "
                      f"periods -> {len(periods)}")

        print(f"    [slinger/{ch_name}] result: {len(periods)} period(s)")
        return periods

    slinger_abp = _detect_one(abp, "abp")
    slinger_cvp = _detect_one(cvp, "cvp")
    return slinger_abp, slinger_cvp


# ── step 5 : gas bubble ────────────────────────────────────────────────────────

def detect_gasbubble(abp, cvp, fs, excluded,
                     hp_cutoff, spec_fmax, k,
                     # ── sensitivity / bound parameters ────────────────────
                     min_event_s=2.0,
                     lp_boundary_hz=0.5,
                     spec_overlap=0.75,
                     lp_lo_ratio=0.5,
                     lp_hi_ratio=1.5,
                     slinger_hp_ratio=1.5):
    """
    Detect gas-bubble artefacts (damped pulse, mean pressure near normal).

    Strategy:
      Same spectrogram-collapse detection as Path A of detect_calibration_flush,
      but with the LP condition flipped to the positive: a collapsed window whose
      LP mean stays between lp_lo_ratio and lp_hi_ratio of the clean baseline is
      classified as a gas bubble.  An HP-std guard rejects slinger false positives.

    Parameters
    ----------
    abp, cvp          : 1D float arrays
    fs                : sample rate (Hz)
    excluded          : bool array, True = already excluded
    hp_cutoff         : high-pass cutoff (Hz) for oscillation amplitude check
    spec_fmax         : upper frequency limit for P_total computation (Hz)
    k                 : collapse threshold — collapsed if P_total < p75 / k
                        (linear, not quadratic, because gas bubble damps rather
                        than fully silences the signal)
    min_event_s       : minimum accepted event duration (s)
    lp_boundary_hz    : cutoff (Hz) for the LP used as mean-pressure estimator
    spec_overlap      : fractional overlap between successive spectrogram windows
    lp_lo_ratio       : lower LP fraction — LP mean must be ≥ lp_lo_ratio × baseline
    lp_hi_ratio       : upper LP fraction — LP mean must be ≤ lp_hi_ratio × baseline
    slinger_hp_ratio  : reject if HP std during candidate > slinger_hp_ratio × baseline;
                        gas bubble damps oscillations, slinger amplifies them

    Returns
    -------
    list of (start, end) tuples
    """
    print("    [gasbubble] running gas-bubble detection")

    n        = len(abp)
    window_s = 2.0
    nperseg  = max(int(window_s * fs), 32)
    noverlap = int(nperseg * spec_overlap)
    clean    = ~excluded
    min_dur  = max(int(min_event_s * fs), 1)

    # ── LP baseline (ABP) ────────────────────────────────────────────────────
    nyq  = 0.5 * fs
    b, a = butter(2, max(lp_boundary_hz / nyq, 1e-3), btype="low")
    lp   = filtfilt(b, a, abp)
    lp_norm = np.median(lp[clean]) if clean.any() else np.median(lp)
    print(f"    [gasbubble] ABP LP baseline: {lp_norm:.1f} mmHg  "
          f"gas-bubble LP window: {lp_norm * lp_lo_ratio:.1f}–"
          f"{lp_norm * lp_hi_ratio:.1f} mmHg")

    # ── HP std baseline — used to reject slinger false positives ─────────────
    hp_abp      = highpass(abp, fs, cutoff_hz=hp_cutoff)
    hp_std      = rolling_std(hp_abp, fs, window_s=1.0)
    valid_std   = clean & ~np.isnan(hp_std)
    hp_std_norm = (float(np.median(hp_std[valid_std])) if valid_std.any()
                   else float(np.nanmedian(hp_std)))
    print(f"    [gasbubble] HP-std baseline: {hp_std_norm:.2f} mmHg  "
          f"slinger reject above: {slinger_hp_ratio:.1f}×")

    # ── spectrogram P_total ──────────────────────────────────────────────────
    freqs, times, Sxx = _spectrogram(
        abp, fs=fs, nperseg=nperseg, noverlap=noverlap,
        nfft=nperseg, scaling="density",
    )
    band    = (freqs >= 0.5) & (freqs <= spec_fmax)
    P_total = Sxx[band, :].sum(axis=0)

    half_win = nperseg // 2
    def _ef(ti):
        s = max(0, int(ti * fs) - half_win)
        e = min(n, int(ti * fs) + half_win)
        return excluded[s:e].mean() if e > s else 1.0

    excl_frac = np.array([_ef(ti) for ti in times])
    clean_win = excl_frac < 0.5
    scan_win  = excl_frac < 0.9

    periods = []

    if clean_win.sum() < 4:
        print("    [gasbubble] not enough clean windows — skipped")
        return periods

    p75          = np.percentile(P_total[clean_win], 75)
    collapse_thr = p75 / k
    collapsed    = (P_total < collapse_thr) & scan_win
    print(f"    [gasbubble] P_total p75={p75:.2e}  "
          f"collapse threshold={collapse_thr:.2e}  "
          f"collapsed windows={collapsed.sum()}")

    # ── scan collapsed runs ──────────────────────────────────────────────────
    i = 0
    while i < len(times):
        if collapsed[i]:
            j = i + 1
            while j < len(times) and collapsed[j]:
                j += 1

            s_rough = max(0, int((times[i]     - window_s / 2) * fs))
            e_rough = min(n, int((times[j - 1] + window_s / 2) * fs))

            lp_mean = lp[s_rough:e_rough].mean()

            if lp_norm * lp_lo_ratio <= lp_mean <= lp_norm * lp_hi_ratio:
                seg_std = hp_std[s_rough:e_rough]
                seg_std = seg_std[~np.isnan(seg_std)]
                if seg_std.size > 0 and seg_std.mean() > hp_std_norm * slinger_hp_ratio:
                    print(f"    [gasbubble] {s_rough/fs:.1f}–{e_rough/fs:.1f} s: "
                          f"HP std elevated ({seg_std.mean():.2f} vs baseline "
                          f"{hp_std_norm:.2f}) — likely slinger, not gas bubble")
                elif e_rough - s_rough >= min_dur:
                    periods.append((s_rough, e_rough))
                    mean_std = float(seg_std.mean()) if seg_std.size > 0 else float("nan")
                    print(f"    [gasbubble] period  "
                          f"{s_rough/fs:.1f}–{e_rough/fs:.1f} s  "
                          f"(LP {lp_mean:.1f} mmHg  HP std {mean_std:.2f} mmHg)")
                else:
                    print(f"    [gasbubble] too short "
                          f"({(e_rough - s_rough)/fs:.2f} s) — skipped")
            else:
                print(f"    [gasbubble] LP={lp_mean:.1f} outside gas-bubble range "
                      f"— not gas bubble (cal or flush)")

            i = j
        else:
            i += 1

    print(f"    [gasbubble] result: {len(periods)} period(s)")
    return periods

