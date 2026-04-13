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
                              flush_offset=15.0):
    """
    Detect calibration and flush periods, separately per channel.

    Two detection paths are combined:
      Path A — spectrogram collapse → LP below baseline  →  calibration
                                    → LP above baseline  →  flush (flat type)
      Path B — LP spike alone (> baseline + flush_offset mmHg, ≥ 2 s)
                                                         →  flush (oscillating type)
    Path B catches flushes where the signal never goes quiet (e.g. turbulent
    CVP flush with large oscillations) that Path A would miss entirely.

    Parameters
    ----------
    abp, cvp      : 1D float arrays
    fs            : sample rate (Hz)
    excluded      : bool array, True = already excluded
    lp_cutoff     : low-pass cutoff (Hz) used for cal-vs-flush classification
    spec_fmax     : upper frequency limit for P_total computation (Hz)
    k             : outlier threshold — collapsed if P_total < p75 / k²
    flush_offset  : fixed mmHg above baseline LP to declare a flush spike (Path B)

    Returns
    -------
    cal_abp, flush_abp, cal_cvp, flush_cvp : four lists of (start, end) tuples
    """
    n             = len(abp)
    window_s      = 2.0
    nperseg       = max(int(window_s * fs), 32)
    noverlap      = int(nperseg * 0.75)
    clean_samples = ~excluded

    # 0.5 Hz zero-phase LP for boundary detection (suppresses cardiac, fast response)
    nyq  = 0.5 * fs
    b, a = butter(2, max(0.5 / nyq, 1e-3), btype="low")

    def _detect_one(signal, ch_name):
        """Run the cal/flush pipeline on a single channel signal.
        Returns (cal_periods, flush_periods) as lists of (start, end) tuples.

        Single detection path:
          Spectrogram collapse → LP below baseline  →  calibration
                               → LP above baseline  →  flush
        Both artefacts require oscillations to collapse; they are then told
        apart by whether the slow LP sits below or above its clean median.
        """
        print(f"    [{ch_name}] running cal/flush detection")

        # ── LP signals ────────────────────────────────────────────────────────
        lp_fast = filtfilt(b, a, signal)         # 0.5 Hz — precise boundaries
        lp_slow = lowpass(signal, fs, cutoff_hz=lp_cutoff)  # classification

        def _med(arr):
            return np.median(arr[clean_samples]) if clean_samples.any() else np.median(arr)

        lp_fast_norm = _med(lp_fast)
        lp_slow_norm = _med(lp_slow)
        print(f"    [{ch_name}] LP baseline: {lp_fast_norm:.1f} mmHg  "
              f"LP slow baseline: {lp_slow_norm:.1f} mmHg")

        # ── HP rolling std — used by Path B to confirm oscillations stop ──────
        hp_sig  = highpass(signal, fs, cutoff_hz=hp_cutoff)
        hp_std  = rolling_std(hp_sig, fs, window_s=1.0)
        valid   = clean_samples & ~np.isnan(hp_std)
        hp_std_norm = (np.median(hp_std[valid]) if valid.any()
                       else np.median(hp_std[~np.isnan(hp_std)]))
        print(f"    [{ch_name}] HP-std baseline: {hp_std_norm:.2f} mmHg")

        # Helper: given a rough (s, e), refine boundaries via LP gradient.
        # filtfilt produces a zero-phase sigmoid at each transition; the
        # inflection point (steepest slope) falls exactly at the true edge,
        # so argmax/argmin of the gradient recovers sub-second accuracy.
        #
        # Searches are constrained to avoid picking the wrong transition when
        # the extended window contains multiple edges (e.g. flush preceding cal):
        #   cal  — start: steepest fall  WITHIN rough window (not outside it)
        #                  end : steepest rise  AFTER the start
        #   flush — start: steepest rise  up to the rough window end
        #                  end : steepest fall  AFTER the start
        def _lp_boundaries(s_rough, e_rough, is_flush):
            extend = int(window_s * fs)
            s_srch = max(0, s_rough - extend)
            e_srch = min(n, e_rough + extend)
            seg    = lp_fast[s_srch:e_srch]
            if len(seg) < 4:
                return s_rough, e_rough

            deriv = np.gradient(seg)
            r_s   = s_rough - s_srch   # rough-start offset inside seg
            r_e   = e_rough - s_srch   # rough-end   offset inside seg

            if is_flush:
                sub = deriv[:r_e]                      # up to rough end
                if sub.size == 0: return s_rough, e_rough
                start_off = int(np.argmax(sub))        # steepest rise → flush start
                sub2 = deriv[start_off:]               # after flush start
                if sub2.size == 0: return s_rough, e_rough
                end_off = start_off + int(np.argmin(sub2))   # steepest fall → flush end
            else:
                # Search from s_srch (full backward extension) up to r_e.
                # Using r_s as the start was too conservative: the spectrogram's
                # first collapsed window centre can land slightly after the actual
                # step, placing the true lp_fast inflection just before r_s.
                sub = deriv[:r_e]
                if sub.size == 0: return s_rough, e_rough
                start_off = int(np.argmin(sub))        # steepest fall → cal start
                sub2 = deriv[start_off:]               # after cal start
                if sub2.size == 0: return s_rough, e_rough
                end_off = start_off + int(np.argmax(sub2))   # steepest rise → cal end

            if start_off >= end_off:
                return s_rough, e_rough                # degenerate — keep rough bounds

            return s_srch + start_off, s_srch + end_off + 1

        # ── Path A: spectrogram collapse ──────────────────────────────────────
        print(f"    [{ch_name}] Path A: computing spectrogram (window {window_s} s)")
        freqs, times, Sxx = _spectrogram(
            signal, fs=fs, nperseg=nperseg, noverlap=noverlap,
            nfft=nperseg, scaling="density",
        )
        band    = (freqs >= 0.5) & (freqs <= spec_fmax)
        P_total = Sxx[band, :].sum(axis=0)

        half_win  = nperseg // 2
        def _ef(ti):
            s = max(0, int(ti * fs) - half_win)
            e = min(n, int(ti * fs) + half_win)
            return excluded[s:e].mean() if e > s else 1.0

        clean_win = np.array([_ef(ti) for ti in times]) < 0.5
        cal_p  = []
        flush_a = []   # flush from path A

        if clean_win.sum() >= 4:
            p75       = np.percentile(P_total[clean_win], 75)
            collapse_thr = p75 / (k ** 2)
            collapsed = (P_total < collapse_thr) & clean_win
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
                        is_flush = True    # LP elevated → flush
                    elif lp_mean < lp_slow_norm * 0.5:
                        is_flush = False   # LP dropped to near zero → calibration
                    else:
                        # LP stays near normal baseline despite oscillation collapse
                        # → gas bubble candidate; not a cal/flush event.
                        print(f"    [{ch_name}] Path A: LP near baseline "
                              f"({lp_mean:.1f} vs {lp_slow_norm:.1f} mmHg) "
                              f"— not cal/flush, skipping (gas bubble?)")
                        i = j
                        continue

                    s_final, e_final  = _lp_boundaries(s_rough, e_rough, is_flush)
                    kind = "flush" if is_flush else "cal"

                    # Ignore micro-detections shorter than 2 s — these are
                    # artefacts of baseline re-estimation, not real events.
                    if e_final - s_final < int(2.0 * fs):
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
        flush_thr    = lp_slow_norm + flush_offset
        min_samples  = max(int(2.0 * fs), 1)
        above        = lp_slow > flush_thr
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

                    # Flush must either suppress oscillations (beats flat) or
                    # produce clearly turbulent oscillations (>> normal amplitude).
                    # Infuus keeps HP std near the normal cardiac baseline (0.3–2×).
                    # Reject only when HP std falls in that "normal cardiac" band.
                    seg = hp_std[s_f:e_f]
                    seg = seg[~np.isnan(seg)]
                    if seg.size > 0:
                        ratio = seg.mean() / hp_std_norm if hp_std_norm > 0 else 0.0
                        if hp_std_norm * 0.3 <= seg.mean() <= hp_std_norm * 2.0:
                            print(f"    [{ch_name}] Path B: LP spike but oscillations "
                                  f"at normal amplitude (HP std {seg.mean():.2f}, "
                                  f"{ratio:.2f}× baseline {hp_std_norm:.2f}) "
                                  f"— not flush (infuus?)")
                            i = j
                            continue
                        else:
                            print(f"    [{ch_name}] Path B: flush confirmed — "
                                  f"oscillations {'flat' if ratio < 0.3 else 'turbulent'} "
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

        # ── merge flush from both paths, remove overlaps ──────────────────────
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
                   lp_cutoff, hp_cutoff, spec_fmax, k):
    """
    Detect resonance / ringing artefacts (slinger).

    Characteristic: a sudden burst of high-frequency content above 2 × the
    dominant cardiac frequency, faster oscillation periods, and elevated HP
    standard deviation.  Mean arterial pressure (LP) stays near normal.

    Strategy:
      1. Compute ABP spectrogram.  Find dominant cardiac frequency f_dom as
         the spectral peak in the 0.5–4 Hz band averaged over clean windows.
      2. ringing_ratio(t) = P_high(t) / P_total(t)
           P_high  = spectrogram power above 2 × f_dom  (ringing content)
           P_total = spectrogram power in 0.5 Hz – spec_fmax
      3. Flag windows where ringing_ratio > median + k × IQR (clean baseline).
      4. Confirm LP stays near normal: drop or spike indicates a different
         artefact (transducer / flush) that should have been caught earlier.
      5. Confirm HP std ≥ 1.5 × baseline: genuine ringing produces clearly
         elevated oscillation amplitude, not just a spectral redistribution.

    Returns
    -------
    list of (start, end) tuples
    """
    print("    [slinger] running slinger detection")

    n        = len(abp)
    clean    = ~excluded
    window_s = 2.0
    nperseg  = max(int(window_s * fs), 32)
    noverlap = int(nperseg * 0.75)
    min_dur  = max(int(2.0 * fs), 1)

    # ── spectrogram ───────────────────────────────────────────────────────────
    freqs, times, Sxx = _spectrogram(
        abp, fs=fs, nperseg=nperseg, noverlap=noverlap,
        nfft=nperseg, scaling="density",
    )

    half_win = nperseg // 2
    def _ef(ti):
        s = max(0, int(ti * fs) - half_win)
        e = min(n, int(ti * fs) + half_win)
        return excluded[s:e].mean() if e > s else 1.0

    clean_win = np.array([_ef(ti) for ti in times]) < 0.5

    # ── dominant cardiac frequency ────────────────────────────────────────────
    cardiac_band = (freqs >= 0.5) & (freqs <= 4.0)
    if clean_win.sum() >= 4 and cardiac_band.any():
        # Average power spectrum over clean windows, find the dominant peak
        P_mean = Sxx[cardiac_band][:, clean_win].mean(axis=1)
        f_dom  = float(freqs[cardiac_band][np.argmax(P_mean)])
    else:
        f_dom = 1.2   # fallback: 72 bpm
    print(f"    [slinger] dominant cardiac frequency: {f_dom:.2f} Hz")

    # ── ringing ratio ─────────────────────────────────────────────────────────
    # Ringing power lives above 2 × f_dom; total power is the full 0.5–fmax band.
    high_band  = (freqs > 1.6 * f_dom) & (freqs <= spec_fmax)
    total_band = (freqs >= 0.5)        & (freqs <= spec_fmax)

    if not high_band.any():
        print(f"    [slinger] no frequencies above {2 * f_dom:.1f} Hz in spectrogram — skipped")
        return []

    P_high  = Sxx[high_band,  :].sum(axis=0)
    P_total = Sxx[total_band, :].sum(axis=0)
    ringing_ratio = np.where(P_total > 0, P_high / P_total, 0.0)

    # ── outlier threshold (k × IQR above clean median) ───────────────────────
    rr_ref = ringing_ratio[clean_win] if clean_win.sum() >= 4 else ringing_ratio
    rr_med = float(np.median(rr_ref))
    rr_iqr = float(max(np.subtract(*np.percentile(rr_ref, [75, 25])), 1e-6))
    rr_thr = rr_med + k * rr_iqr
    ringy  = ringing_ratio > rr_thr

    print(f"    [slinger] ringing_ratio  median={rr_med:.4f}  IQR={rr_iqr:.4f}  "
          f"threshold={rr_thr:.4f}  flagged windows={ringy.sum()}")

    # ── LP baseline and HP std baseline ──────────────────────────────────────
    lp_abp  = lowpass(abp,  fs, cutoff_hz=lp_cutoff)
    hp_abp  = highpass(abp, fs, cutoff_hz=hp_cutoff)
    hp_std  = rolling_std(hp_abp, fs, window_s=1.0)

    lp_norm = float(np.median(lp_abp[clean]) if clean.any() else np.median(lp_abp))
    valid_std   = clean & ~np.isnan(hp_std)
    hp_std_norm = (float(np.median(hp_std[valid_std])) if valid_std.any()
                   else float(np.nanmedian(hp_std)))

    print(f"    [slinger] LP baseline: {lp_norm:.1f} mmHg  "
          f"HP-std baseline: {hp_std_norm:.2f} mmHg")

    # ── scan ringy runs ───────────────────────────────────────────────────────
    periods = []
    i = 0
    while i < len(times):
        if ringy[i]:
            j = i + 1
            while j < len(times) and ringy[j]:
                j += 1

            s_rough = max(0, int((times[i]     - window_s / 2) * fs))
            e_rough = min(n, int((times[j - 1] + window_s / 2) * fs))

            if e_rough - s_rough >= min_dur:
                lp_mean = float(lp_abp[s_rough:e_rough].mean())

                # LP must stay near normal — a drop indicates transducer,
                # a large spike indicates flush (both caught in earlier steps)
                if lp_mean < lp_norm * 0.6:
                    print(f"    [slinger] {s_rough/fs:.1f}–{e_rough/fs:.1f} s: "
                          f"LP dropped ({lp_mean:.1f} vs {lp_norm:.1f} mmHg) "
                          f"— not slinger")
                    i = j
                    continue
                if lp_mean > lp_norm + 15.0:
                    print(f"    [slinger] {s_rough/fs:.1f}–{e_rough/fs:.1f} s: "
                          f"LP elevated ({lp_mean:.1f} vs {lp_norm:.1f} mmHg) "
                          f"— not slinger")
                    i = j
                    continue

                # HP std check removed as confirmation: ringing can be
                # concentrated at a narrow frequency band that shifts the
                # ringing_ratio clearly but does not raise broadband HP std
                # by the 1.5× threshold.  ringing_ratio + LP guard are
                # sufficient discriminators.
                seg_std  = hp_std[s_rough:e_rough]
                seg_std  = seg_std[~np.isnan(seg_std)]
                rr_mean  = float(ringing_ratio[i:j].mean())
                mean_std = float(seg_std.mean()) if seg_std.size > 0 else float("nan")
                periods.append((s_rough, e_rough))
                print(f"    [slinger] period  {s_rough/fs:.1f}–{e_rough/fs:.1f} s  "
                      f"ringing_ratio {rr_mean:.3f}  HP std {mean_std:.2f} mmHg")
            else:
                print(f"    [slinger] window at {times[i]:.1f} s too short — ignored")

            i = j
        else:
            i += 1

    print(f"    [slinger] result: {len(periods)} period(s)")
    return periods


# ── step 3 ─────────────────────────────────────────────────────────────────────

def detect_gasbubble(abp, cvp, fs, excluded,
                     hp_cutoff, spec_fmax, k):
    """
    Detect gas-bubble artefacts (damped pulse, mean pressure near normal).

    Strategy:
      Same spectrogram-collapse detection as Path A of detect_calibration_flush,
      but with the LP condition flipped to the positive: a collapsed window whose
      LP mean stays between 50 % and 150 % of the clean baseline is classified as
      a gas bubble rather than calibration (LP < 50 %) or flush (LP > baseline).

    This works because gas bubbles damp the pulsatile component (P_total drops)
    while the mean arterial pressure is largely unaffected.

    Returns
    -------
    list of (start, end) tuples
    """
    print("    [gasbubble] running gas-bubble detection")

    n         = len(abp)
    window_s  = 2.0
    nperseg   = max(int(window_s * fs), 32)
    noverlap  = int(nperseg * 0.75)
    clean     = ~excluded
    min_dur   = max(int(2.0 * fs), 1)

    # ── LP baseline (ABP) ────────────────────────────────────────────────────
    nyq  = 0.5 * fs
    b, a = butter(2, max(0.5 / nyq, 1e-3), btype="low")
    lp   = filtfilt(b, a, abp)
    lp_norm = np.median(lp[clean]) if clean.any() else np.median(lp)
    print(f"    [gasbubble] ABP LP baseline: {lp_norm:.1f} mmHg  "
          f"gas-bubble LP window: {lp_norm * 0.5:.1f}–{lp_norm * 1.5:.1f} mmHg")

    # ── HP std baseline — used to reject slinger false positives ─────────────
    # A slinger can make P_total collapse in the 0.5–spec_fmax band (if ringing
    # energy sits above spec_fmax), mimicking a gas bubble.  The key difference:
    # gas bubble damps oscillations (HP std drops), slinger amplifies them.
    hp_abp      = highpass(abp, fs, cutoff_hz=hp_cutoff)
    hp_std      = rolling_std(hp_abp, fs, window_s=1.0)
    valid_std   = clean & ~np.isnan(hp_std)
    hp_std_norm = (float(np.median(hp_std[valid_std])) if valid_std.any()
                   else float(np.nanmedian(hp_std)))
    print(f"    [gasbubble] HP-std baseline: {hp_std_norm:.2f} mmHg")

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
    # Strict threshold for baseline p75 only (keeps artefact windows out).
    clean_win = excl_frac < 0.5
    # Looser threshold for the collapse scan — windows that merely clip a flush
    # boundary are included; the LP check rejects any that land inside a flush.
    scan_win  = excl_frac < 0.9

    periods = []

    if clean_win.sum() < 4:
        print("    [gasbubble] not enough clean windows — skipped")
        return periods

    p75          = np.percentile(P_total[clean_win], 75)
    # Gas bubble damps beats but does not flatten the signal completely, so use
    # a linear (k×) drop rather than the quadratic (k²×) used for calibration.
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

            if lp_norm * 0.5 <= lp_mean <= lp_norm * 1.5:
                # LP near baseline — could be gas bubble, but check HP std
                # first: a slinger can collapse P_total by pushing ringing
                # energy above spec_fmax while HP std is elevated.
                seg_std = hp_std[s_rough:e_rough]
                seg_std = seg_std[~np.isnan(seg_std)]
                if seg_std.size > 0 and seg_std.mean() > hp_std_norm * 1.5:
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
                            transducer_offset=10.0):
    """
    Detect transducer-repositioning artefacts (transducer too high).

    Characteristic: the ABP (and usually CVP) LP baseline drops while
    oscillations continue unchanged.  The transducer has been placed above
    the phlebostatic axis, causing a hydrostatic under-read of ~15–20 mmHg.

    Strategy:
      1. Compute LP_ABP and HP_ABP rolling std.
      2. ABP threshold = p90(LP_ABP_clean) − transducer_offset.
         p90 is contamination-robust: even if the drop covers >50 % of the
         recording the upper tail (normal-pressure samples) is unaffected.
      3. Flag sustained periods (≥ 2 s) where LP_ABP < threshold.
      4. Refine start/end via the steepest LP gradient (same approach as
         cal/flush boundary refinement).
      5. Confirm oscillations continue: HP std must be ≥ 0.3 × baseline.
         Collapsed oscillations indicate calibration, not a repositioning.

    Returns
    -------
    list of (start, end) tuples
    """
    print("    [transducer] running transducer-hoog detection")

    n           = len(abp)
    clean       = ~excluded
    min_samples = max(int(2.0 * fs), 1)

    lp_abp = lowpass(abp, fs, cutoff_hz=lp_cutoff)
    hp_abp = highpass(abp, fs, cutoff_hz=hp_cutoff)
    hp_std = rolling_std(hp_abp, fs, window_s=1.0)

    abp_vals = lp_abp[clean] if clean.any() else lp_abp

    # p90: contamination-robust upper baseline.
    # The drop period deflates lower percentiles; p90 stays anchored to the
    # normal-pressure portion of the signal regardless of drop duration.
    p90_abp = float(np.percentile(abp_vals, 90))
    thr_abp = p90_abp - transducer_offset

    # HP std baseline
    valid_std   = clean & ~np.isnan(hp_std)
    hp_std_norm = (float(np.median(hp_std[valid_std])) if valid_std.any()
                   else float(np.nanmedian(hp_std)))

    print(f"    [transducer] ABP LP  p90={p90_abp:.1f}  "
          f"threshold={thr_abp:.1f} mmHg  (offset={transducer_offset} mmHg)")
    print(f"    [transducer] HP-std baseline: {hp_std_norm:.2f} mmHg")

    dropped           = lp_abp < thr_abp
    dropped[excluded] = False

    periods = []
    i = 0
    while i < n:
        if dropped[i]:
            j = i + 1
            while j < n and dropped[j]:
                j += 1

            if j - i >= min_samples:
                # No gradient-based boundary refinement here.
                # Unlike flush/calibration (clean step transitions), the
                # transducer_hoog LP has a brief post-drop bounce that
                # confuses gradient argmax.  zero-phase filtfilt means
                # the threshold crossings are already accurately placed.
                s_f, e_f = i, j

                # Confirm oscillations are present but not wildly elevated.
                # < 0.3× baseline: oscillations absent → calibration, not transducer.
                # > 2.5× baseline: oscillations very elevated → slinger, not transducer.
                # Transducer repositioning leaves cardiac oscillations near normal.
                seg_std = hp_std[s_f:e_f]
                seg_std = seg_std[~np.isnan(seg_std)]
                if seg_std.size > 0:
                    ratio = seg_std.mean() / hp_std_norm if hp_std_norm > 0 else 0.0
                    if ratio < 0.3:
                        print(f"    [transducer] {i/fs:.1f}–{j/fs:.1f} s: "
                              f"oscillations absent ({seg_std.mean():.2f}, "
                              f"{ratio:.2f}× baseline) — not transducer hoog, skipped")
                        i = j
                        continue
                    if ratio > 2.5:
                        print(f"    [transducer] {i/fs:.1f}–{j/fs:.1f} s: "
                              f"oscillations very elevated ({seg_std.mean():.2f}, "
                              f"{ratio:.2f}× baseline) — likely slinger, not transducer hoog, skipped")
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


def detect_infuus(abp, cvp, fs, excluded,
                  lp_cutoff, k,
                  infuus_offset=5.0,
                  min_cvp_mean=20.0):
    """
    Detect IV-infusion artefacts on the CVP channel.

    Characteristic: CVP LP baseline rises while oscillations continue
    (HP amplitude stays normal).  ABP remains near its own baseline.

    Strategy:
      1. Compute LP_CVP and LP_ABP.
      2. CVP threshold = p10(LP_CVP) + infuus_offset.
         Using p10 instead of the median makes the baseline robust to
         contamination: even if the infuus covers 50 %+ of the recording
         the elevated samples are in the upper tail and cannot inflate p10.
      3. ABP threshold = median(LP_ABP) + k × IQR(LP_ABP).
         ABP is unaffected by an infuus so the ordinary IQR approach works.
      4. Flag sustained periods (≥ 2 s) where LP_CVP > CVP threshold
         AND LP_ABP ≤ ABP threshold (CVP-only shift)
         AND mean LP_CVP ≥ min_cvp_mean (physiological sanity check).
         A relative threshold alone can fire at low absolute CVP values
         (e.g. 10 mmHg) that can never represent a real infuus elevation.

    Returns
    -------
    list of (start, end) tuples
    """
    print("    [infuus] running infuus detection")

    n           = len(cvp)
    clean       = ~excluded
    min_samples = max(int(2.0 * fs), 1)

    lp_cvp = lowpass(cvp, fs, cutoff_hz=lp_cutoff)
    lp_abp = lowpass(abp, fs, cutoff_hz=lp_cutoff)

    cvp_vals = lp_cvp[clean] if clean.any() else lp_cvp
    abp_vals = lp_abp[clean] if clean.any() else lp_abp

    # CVP: p10 as contamination-robust baseline + fixed offset
    p10_cvp = float(np.percentile(cvp_vals, 10))
    thr_cvp = p10_cvp + infuus_offset

    # ABP: standard median ± IQR (uncontaminated)
    med_abp = float(np.median(abp_vals))
    iqr_abp = float(max(np.subtract(*np.percentile(abp_vals, [75, 25])), 1.0))
    thr_abp = med_abp + k * iqr_abp

    print(f"    [infuus] CVP LP  p10={p10_cvp:.1f}  "
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
