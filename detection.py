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


# ── step 1 ─────────────────────────────────────────────────────────────────────

def detect_calibration_flush(abp, cvp, fs, excluded,
                              lp_cutoff, hp_cutoff, spec_fmax, k,
                              # ── sensitivity / bound parameters ────────────
                              flush_offset=15.0,
                              min_event_s=2.0,
                              lp_boundary_hz=0.5,
                              spec_overlap=0.75,
                              cal_lp_ratio=0.5,
                              flush_hp_lo=0.3,
                              flush_hp_hi=2.0):
    """
    Detect calibration and flush periods, separately per channel.

    Two detection paths are combined:
      Path A — spectrogram collapse → LP below baseline  →  calibration
                                    → LP above baseline  →  flush (flat type)
      Path B — LP spike alone (> baseline + flush_offset mmHg, ≥ min_event_s)
                                                         →  flush (oscillating type)
    Path B catches flushes where the signal never goes quiet (e.g. turbulent
    CVP flush with large oscillations) that Path A would miss entirely.

    Parameters
    ----------
    abp, cvp         : 1D float arrays
    fs               : sample rate (Hz)
    excluded         : bool array, True = already excluded
    lp_cutoff        : low-pass cutoff (Hz) for cal-vs-flush classification
    hp_cutoff        : high-pass cutoff (Hz) for oscillation amplitude check
    spec_fmax        : upper frequency limit for P_total computation (Hz)
    k                : collapse threshold — collapsed if P_total < p75 / k²
    flush_offset     : mmHg above baseline LP needed to call a Path-B flush spike
    min_event_s      : minimum accepted event duration (s)
    lp_boundary_hz   : cutoff (Hz) for the fast LP used in boundary refinement
    spec_overlap     : fractional overlap between successive spectrogram windows
    cal_lp_ratio     : LP must be below this fraction of baseline to call calibration
                       (vs gas-bubble); 0.5 means LP < 50 % of baseline
    flush_hp_lo      : HP-std lower bound (× baseline) for Path-B flush acceptance;
                       ratios below this are "flat" (oscillations suppressed) → flush
    flush_hp_hi      : HP-std upper bound (× baseline) for Path-B flush acceptance;
                       ratios above this are "turbulent" → flush.
                       Ratios inside [flush_hp_lo, flush_hp_hi] are "normal cardiac"
                       amplitude → infuus, not flush.

    Returns
    -------
    cal_abp, flush_abp, cal_cvp, flush_cvp : four lists of (start, end) tuples
    """
    n             = len(abp)
    window_s      = 2.0
    nperseg       = max(int(window_s * fs), 32)
    noverlap      = int(nperseg * spec_overlap)
    min_samples   = max(int(min_event_s * fs), 1)
    clean_samples = ~excluded

    # Zero-phase LP for boundary detection
    nyq  = 0.5 * fs
    b, a = butter(2, max(lp_boundary_hz / nyq, 1e-3), btype="low")

    def _detect_one(signal, ch_name):
        print(f"    [{ch_name}] running cal/flush detection")

        lp_fast = filtfilt(b, a, signal)
        lp_slow = lowpass(signal, fs, cutoff_hz=lp_cutoff)

        def _med(arr):
            return np.median(arr[clean_samples]) if clean_samples.any() else np.median(arr)

        lp_fast_norm = _med(lp_fast)
        lp_slow_norm = _med(lp_slow)
        print(f"    [{ch_name}] LP baseline: {lp_fast_norm:.1f} mmHg  "
              f"LP slow baseline: {lp_slow_norm:.1f} mmHg")

        hp_sig      = highpass(signal, fs, cutoff_hz=hp_cutoff)
        hp_std      = rolling_std(hp_sig, fs, window_s=1.0)
        valid       = clean_samples & ~np.isnan(hp_std)
        hp_std_norm = (np.median(hp_std[valid]) if valid.any()
                       else np.median(hp_std[~np.isnan(hp_std)]))
        print(f"    [{ch_name}] HP-std baseline: {hp_std_norm:.2f} mmHg")

        def _lp_boundaries(s_rough, e_rough, is_flush):
            extend = int(window_s * fs)
            s_srch = max(0, s_rough - extend)
            e_srch = min(n, e_rough + extend)
            seg    = lp_fast[s_srch:e_srch]
            if len(seg) < 4:
                return s_rough, e_rough
            deriv = np.gradient(seg)
            r_e   = e_rough - s_srch
            if is_flush:
                sub = deriv[:r_e]
                if sub.size == 0: return s_rough, e_rough
                start_off = int(np.argmax(sub))
                sub2 = deriv[start_off:]
                if sub2.size == 0: return s_rough, e_rough
                end_off = start_off + int(np.argmin(sub2))
            else:
                sub = deriv[:r_e]
                if sub.size == 0: return s_rough, e_rough
                start_off = int(np.argmin(sub))
                sub2 = deriv[start_off:]
                if sub2.size == 0: return s_rough, e_rough
                end_off = start_off + int(np.argmax(sub2))
            if start_off >= end_off:
                return s_rough, e_rough
            return s_srch + start_off, s_srch + end_off + 1

        # ── Path A: spectrogram collapse ──────────────────────────────────────
        print(f"    [{ch_name}] Path A: computing spectrogram (window {window_s} s)")
        freqs, times, Sxx = _spectrogram(
            signal, fs=fs, nperseg=nperseg, noverlap=noverlap,
            nfft=nperseg, scaling="density",
        )
        band    = (freqs >= 0.5) & (freqs <= spec_fmax)
        P_total = Sxx[band, :].sum(axis=0)

        half_win = nperseg // 2
        def _ef(ti):
            s = max(0, int(ti * fs) - half_win)
            e = min(n, int(ti * fs) + half_win)
            return excluded[s:e].mean() if e > s else 1.0

        clean_win = np.array([_ef(ti) for ti in times]) < 0.5
        cal_p   = []
        flush_a = []

        if clean_win.sum() >= 4:
            p75          = np.percentile(P_total[clean_win], 75)
            collapse_thr = p75 / (k ** 2)
            collapsed    = (P_total < collapse_thr) & clean_win
            print(f"    [{ch_name}] Path A: P_total p75={p75:.2e}  "
                  f"collapse threshold={collapse_thr:.2e}  "
                  f"collapsed windows={collapsed.sum()}")

            i = 0
            while i < len(times):
                if collapsed[i]:
                    j = i + 1
                    while j < len(times) and collapsed[j]:
                        j += 1

                    s_rough = max(0, int((times[i]     - window_s / 2) * fs))
                    e_rough = min(n, int((times[j - 1] + window_s / 2) * fs))

                    lp_mean = lp_slow[s_rough:e_rough].mean()
                    if lp_mean > lp_slow_norm:
                        is_flush = True
                    elif lp_mean < lp_slow_norm * cal_lp_ratio:
                        is_flush = False
                    else:
                        print(f"    [{ch_name}] Path A: LP near baseline "
                              f"({lp_mean:.1f} vs {lp_slow_norm:.1f} mmHg) "
                              f"— not cal/flush, skipping (gas bubble?)")
                        i = j
                        continue

                    s_final, e_final = _lp_boundaries(s_rough, e_rough, is_flush)
                    kind = "flush" if is_flush else "cal"

                    if e_final - s_final < min_samples:
                        print(f"    [{ch_name}] Path A: {kind} too short "
                              f"({(e_final - s_final) / fs:.2f} s) — skipped")
                        i = j
                        continue

                    print(f"    [{ch_name}] Path A: {kind} period  "
                          f"{s_final/fs:.1f}–{e_final/fs:.1f} s")
                    (flush_a if is_flush else cal_p).append((s_final, e_final))
                    i = j
                else:
                    i += 1
        else:
            print(f"    [{ch_name}] Path A: not enough clean windows — skipped")

        # ── Path B: LP spike → oscillating flush ─────────────────────────────
        print(f"    [{ch_name}] Path B: LP spike scan  "
              f"(threshold = baseline + {flush_offset} mmHg = "
              f"{lp_slow_norm + flush_offset:.1f} mmHg)")
        flush_thr       = lp_slow_norm + flush_offset
        above           = lp_slow > flush_thr
        above[excluded] = False

        flush_b = []
        i = 0
        while i < n:
            if above[i]:
                j = i + 1
                while j < n and above[j]:
                    j += 1
                if j - i >= min_samples:
                    s_f, e_f = _lp_boundaries(i, j, is_flush=True)
                    seg = hp_std[s_f:e_f]
                    seg = seg[~np.isnan(seg)]
                    if seg.size > 0:
                        ratio = seg.mean() / hp_std_norm if hp_std_norm > 0 else 0.0
                        if flush_hp_lo <= ratio <= flush_hp_hi:
                            print(f"    [{ch_name}] Path B: LP spike but oscillations "
                                  f"at normal amplitude (HP std {seg.mean():.2f}, "
                                  f"{ratio:.2f}× baseline {hp_std_norm:.2f}) "
                                  f"— not flush (infuus?)")
                            i = j
                            continue
                        else:
                            print(f"    [{ch_name}] Path B: flush confirmed — "
                                  f"oscillations {'flat' if ratio < flush_hp_lo else 'turbulent'} "
                                  f"(HP std {seg.mean():.2f}, {ratio:.2f}× baseline "
                                  f"{hp_std_norm:.2f})")
                    flush_b.append((s_f, e_f))
                    print(f"    [{ch_name}] Path B: flush period  "
                          f"{s_f/fs:.1f}–{e_f/fs:.1f} s")
                else:
                    print(f"    [{ch_name}] Path B: spike at {i/fs:.1f} s too short "
                          f"({(j - i)/fs:.2f} s) — ignored")
                i = j
            else:
                i += 1

        def _merge(periods):
            if not periods:
                return []
            srt = sorted(periods)
            merged = [list(srt[0])]
            for s, e in srt[1:]:
                if s <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])
            return [tuple(p) for p in merged]

        flush_p = _merge(flush_a + flush_b)
        print(f"    [{ch_name}] result: {len(cal_p)} cal, {len(flush_p)} flush period(s)")
        return cal_p, flush_p

    cal_abp,  flush_abp  = _detect_one(abp, "ABP")
    cal_cvp,  flush_cvp  = _detect_one(cvp, "CVP")
    return cal_abp, flush_abp, cal_cvp, flush_cvp


# ── step 2 ─────────────────────────────────────────────────────────────────────

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


# ── step 3 ─────────────────────────────────────────────────────────────────────

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


# ── step 4 ─────────────────────────────────────────────────────────────────────

def detect_transducer_hoog(abp, cvp, fs, excluded,
                            lp_cutoff, hp_cutoff, k,
                            # ── sensitivity / bound parameters ────────────
                            transducer_offset=10.0,
                            baseline_pct=90,
                            hp_lo_ratio=0.3,
                            hp_hi_ratio=2.5,
                            min_event_s=2.0,
                            tail_guard_s=3.0):
    """
    Detect transducer-repositioning artefacts (transducer too high).

    Characteristic: the ABP (and usually CVP) LP baseline drops while
    oscillations continue near-normally.  The transducer has been placed above
    the phlebostatic axis, causing a hydrostatic under-read of ~15–20 mmHg.

    Parameters
    ----------
    abp, cvp           : 1D float arrays
    fs                 : sample rate (Hz)
    excluded           : bool array, True = already excluded
    lp_cutoff          : low-pass cutoff (Hz)
    hp_cutoff          : high-pass cutoff (Hz)
    k                  : not used directly; kept for API consistency
    transducer_offset  : mmHg subtracted from the baseline percentile to form
                         the detection threshold.  Observed drops are 15–22 mmHg
                         so 10 mmHg gives a safe margin above LP variability.
    baseline_pct       : percentile of clean LP_ABP used as the "normal"
                         upper reference.  p90 is robust to drop contamination.
    hp_lo_ratio        : reject if HP std < hp_lo_ratio × baseline
                         (oscillations absent → calibration, not transducer)
    hp_hi_ratio        : reject if HP std > hp_hi_ratio × baseline
                         (oscillations very elevated → slinger, not transducer)
    min_event_s        : minimum accepted event duration (s)
    tail_guard_s       : reject any period whose end falls within the last
                         tail_guard_s seconds of the recording; a genuine
                         transducer artefact must show a confirmed recovery
                         (LP returns to baseline) before the signal ends.

    Returns
    -------
    list of (start, end) tuples
    """
    print("    [transducer] running transducer-hoog detection")

    n           = len(abp)
    clean       = ~excluded
    min_samples = max(int(min_event_s * fs), 1)

    lp_abp = lowpass(abp, fs, cutoff_hz=lp_cutoff)
    hp_abp = highpass(abp, fs, cutoff_hz=hp_cutoff)
    hp_std = rolling_std(hp_abp, fs, window_s=1.0)

    abp_vals = lp_abp[clean] if clean.any() else lp_abp

    p_baseline = float(np.percentile(abp_vals, baseline_pct))
    thr_abp    = p_baseline - transducer_offset

    valid_std   = clean & ~np.isnan(hp_std)
    hp_std_norm = (float(np.median(hp_std[valid_std])) if valid_std.any()
                   else float(np.nanmedian(hp_std)))

    print(f"    [transducer] ABP LP  p{baseline_pct}={p_baseline:.1f}  "
          f"threshold={thr_abp:.1f} mmHg  (offset={transducer_offset} mmHg)")
    print(f"    [transducer] HP-std baseline: {hp_std_norm:.2f} mmHg  "
          f"accept range: [{hp_lo_ratio:.1f}×, {hp_hi_ratio:.1f}×]")

    dropped           = lp_abp < thr_abp
    dropped[excluded] = False

    tail_guard = max(int(tail_guard_s * fs), 0)

    periods = []
    i = 0
    while i < n:
        if dropped[i]:
            j = i + 1
            while j < n and dropped[j]:
                j += 1

            if j >= n - tail_guard:
                print(f"    [transducer] {i/fs:.1f}–{j/fs:.1f} s: "
                      f"period reaches end of recording (no confirmed recovery) "
                      f"— skipping")
                i = j
                continue

            if j - i >= min_samples:
                s_f, e_f = i, j

                seg_std = hp_std[s_f:e_f]
                seg_std = seg_std[~np.isnan(seg_std)]
                if seg_std.size > 0:
                    ratio = seg_std.mean() / hp_std_norm if hp_std_norm > 0 else 0.0
                    if ratio < hp_lo_ratio:
                        print(f"    [transducer] {i/fs:.1f}–{j/fs:.1f} s: "
                              f"oscillations absent ({seg_std.mean():.2f}, "
                              f"{ratio:.2f}× baseline) — not transducer hoog, skipped")
                        i = j
                        continue
                    if ratio > hp_hi_ratio:
                        print(f"    [transducer] {i/fs:.1f}–{j/fs:.1f} s: "
                              f"oscillations very elevated ({seg_std.mean():.2f}, "
                              f"{ratio:.2f}× baseline) — likely slinger, skipped")
                        i = j
                        continue

                mean_lp  = float(lp_abp[i:j].mean())
                mean_std = float(seg_std.mean()) if seg_std.size > 0 else float("nan")
                periods.append((s_f, e_f))
                print(f"    [transducer] period  {s_f/fs:.1f}–{e_f/fs:.1f} s  "
                      f"(ABP LP {mean_lp:.1f} mmHg  HP std {mean_std:.2f} mmHg)")
            else:
                print(f"    [transducer] drop at {i/fs:.1f} s too short "
                      f"({(j - i)/fs:.2f} s) — ignored")

            i = j
        else:
            i += 1

    print(f"    [transducer] result: {len(periods)} period(s)")
    return periods


# ── step 5 ─────────────────────────────────────────────────────────────────────

def detect_infuus(abp, cvp, fs, excluded,
                  lp_cutoff, k,
                  # ── sensitivity / bound parameters ──────────────────────
                  infuus_offset=5.0,
                  min_cvp_mean=20.0,
                  baseline_pct=10,
                  min_event_s=2.0,
                  min_iqr=1.0):
    """
    Detect IV-infusion artefacts on the CVP channel.

    Characteristic: CVP LP baseline rises while oscillations continue
    (HP amplitude stays normal).  ABP remains near its own baseline.

    Parameters
    ----------
    abp, cvp        : 1D float arrays
    fs              : sample rate (Hz)
    excluded        : bool array, True = already excluded
    lp_cutoff       : low-pass cutoff (Hz)
    k               : IQR multiplier for the ABP upper bound
    infuus_offset   : mmHg added to the CVP baseline percentile to form the
                      elevated-CVP threshold
    min_cvp_mean    : minimum absolute CVP LP mean (mmHg) during the candidate
                      period; physiological sanity check — a real infuus always
                      pushes CVP well above normal baseline
    baseline_pct    : percentile of clean LP_CVP used as the "floor" reference;
                      p10 is robust even when the infuus covers >50 % of the
                      recording (elevated samples stay in the upper tail)
    min_event_s     : minimum accepted event duration (s)
    min_iqr         : floor for the ABP IQR to avoid division instability on
                      very flat signals

    Returns
    -------
    list of (start, end) tuples
    """
    print("    [infuus] running infuus detection")

    n           = len(cvp)
    clean       = ~excluded
    min_samples = max(int(min_event_s * fs), 1)

    lp_cvp = lowpass(cvp, fs, cutoff_hz=lp_cutoff)
    lp_abp = lowpass(abp, fs, cutoff_hz=lp_cutoff)

    cvp_vals = lp_cvp[clean] if clean.any() else lp_cvp
    abp_vals = lp_abp[clean] if clean.any() else lp_abp

    p_cvp   = float(np.percentile(cvp_vals, baseline_pct))
    thr_cvp = p_cvp + infuus_offset

    med_abp = float(np.median(abp_vals))
    iqr_abp = float(max(np.subtract(*np.percentile(abp_vals, [75, 25])), min_iqr))
    thr_abp = med_abp + k * iqr_abp

    print(f"    [infuus] CVP LP  p{baseline_pct}={p_cvp:.1f}  "
          f"threshold={thr_cvp:.1f} mmHg  (offset={infuus_offset} mmHg)  "
          f"min_mean={min_cvp_mean:.0f} mmHg")
    print(f"    [infuus] ABP LP  median={med_abp:.1f}  IQR={iqr_abp:.2f}  "
          f"threshold={thr_abp:.1f} mmHg")

    elevated           = lp_cvp > thr_cvp
    elevated[excluded] = False

    periods = []
    i = 0
    while i < n:
        if elevated[i]:
            j = i + 1
            while j < n and elevated[j]:
                j += 1

            if j - i >= min_samples:
                abp_mean = lp_abp[i:j].mean()
                cvp_mean = lp_cvp[i:j].mean()
                if abp_mean > thr_abp:
                    print(f"    [infuus] period {i/fs:.1f}–{j/fs:.1f} s: "
                          f"ABP also elevated ({abp_mean:.1f} mmHg) — not infuus")
                elif cvp_mean < min_cvp_mean:
                    print(f"    [infuus] period {i/fs:.1f}–{j/fs:.1f} s: "
                          f"CVP mean {cvp_mean:.1f} mmHg below minimum "
                          f"{min_cvp_mean:.0f} mmHg — not infuus")
                else:
                    periods.append((i, j))
                    print(f"    [infuus] period  {i/fs:.1f}–{j/fs:.1f} s  "
                          f"CVP LP {cvp_mean:.1f}  ABP LP {abp_mean:.1f} mmHg")
            i = j
        else:
            i += 1

    print(f"    [infuus] result: {len(periods)} period(s)")
    return periods
