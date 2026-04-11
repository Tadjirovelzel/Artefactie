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
from signal_helpers import lowpass


# ── step 1 ─────────────────────────────────────────────────────────────────────

def detect_calibration_flush(abp, cvp, fs, excluded,
                              lp_cutoff, spec_fmax, k):
    """
    Detect periods where oscillations have collapsed entirely, separately for
    each channel so that ABP and CVP calibration/flush events are independent.

    Strategy (per channel):
      1. Spectrogram → P_total(t): flag windows where power << p75 / k²
         (collapsed oscillations).
      2. LP at 0.5 Hz (zero-phase) for boundary detection: midpoint threshold
         between the normal level and the artefact level in each collapsed run.
      3. Classify each collapsed run as calibration (LP below normal) or flush
         (LP above normal) using a slow LP at lp_cutoff Hz.

    Parameters
    ----------
    abp, cvp  : 1D float arrays
    fs        : sample rate (Hz)
    excluded  : bool array, True = already excluded
    lp_cutoff : low-pass cutoff (Hz) used for cal-vs-flush classification
    spec_fmax : upper frequency limit for P_total computation (Hz)
    k         : outlier threshold — collapsed if P_total < p75 / k²

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

    def _detect_one(signal):
        """Run the full cal/flush pipeline on a single channel signal.
        Returns (cal_periods, flush_periods) as lists of (start, end) tuples.

        Two detection paths are combined:
          Path A — spectrogram collapse → LP drop  →  calibration
                                        → LP spike →  flush (flat-at-high type)
          Path B — LP spike alone       →           →  flush (oscillating type)
        Path B catches flushes where the signal never goes flat (e.g. turbulent
        CVP flush) that Path A would miss entirely.
        """

        # ── LP signals (needed by both paths) ────────────────────────────────
        lp_fast = filtfilt(b, a, signal)         # 0.5 Hz — precise boundaries
        lp_slow = lowpass(signal, fs, cutoff_hz=lp_cutoff)  # classification

        def _med(arr):
            return np.median(arr[clean_samples]) if clean_samples.any() else np.median(arr)
        def _iqr(arr):
            v = arr[clean_samples] if clean_samples.any() else arr
            return float(np.percentile(v, 75) - np.percentile(v, 25)) or 1.0

        lp_fast_norm = _med(lp_fast)
        lp_slow_norm = _med(lp_slow)
        lp_iqr       = _iqr(lp_fast)

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
                sub = deriv[r_s:r_e]                   # within rough window only
                if sub.size == 0: return s_rough, e_rough
                start_off = r_s + int(np.argmin(sub))  # steepest fall → cal start
                sub2 = deriv[start_off:]               # after cal start
                if sub2.size == 0: return s_rough, e_rough
                end_off = start_off + int(np.argmax(sub2))   # steepest rise → cal end

            if start_off >= end_off:
                return s_rough, e_rough                # degenerate — keep rough bounds

            return s_srch + start_off, s_srch + end_off + 1

        # ── Path A: spectrogram collapse ──────────────────────────────────────
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
            collapsed = (P_total < p75 / (k ** 2)) & clean_win

            i = 0
            while i < len(times):
                if collapsed[i]:
                    j = i + 1
                    while j < len(times) and collapsed[j]:
                        j += 1

                    s_rough = max(0, int((times[i]     - window_s / 2) * fs))
                    e_rough = min(n, int((times[j - 1] + window_s / 2) * fs))

                    is_flush          = lp_slow[s_rough:e_rough].mean() > lp_slow_norm
                    s_final, e_final  = _lp_boundaries(s_rough, e_rough, is_flush)

                    (flush_a if is_flush else cal_p).append((s_final, e_final))
                    i = j
                else:
                    i += 1

        # ── Path B: LP spike → flush ──────────────────────────────────────────
        # Detects flush periods where LP rises far above baseline, regardless
        # of whether oscillations collapse.  Uses the same k threshold.
        flush_thr = lp_fast_norm + k * lp_iqr
        min_flush_samples = max(int(2.0 * fs), 1)   # require at least 2 s

        above = lp_fast > flush_thr
        above[excluded] = False   # ignore already-excluded samples

        flush_b = []
        i = 0
        while i < n:
            if above[i]:
                j = i + 1
                while j < n and above[j]:
                    j += 1
                if j - i >= min_flush_samples:
                    s_f, e_f = _lp_boundaries(i, j, is_flush=True)
                    flush_b.append((s_f, e_f))
                i = j
            else:
                i += 1

        # ── Merge flush from both paths, remove overlaps ─────────────────────
        def _merge(periods):
            if not periods:
                return []
            srt = sorted(periods)
            merged = [srt[0]]
            for s, e in srt[1:]:
                if s <= merged[-1][1]:          # overlapping or adjacent
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                else:
                    merged.append((s, e))
            return merged

        flush_p = _merge(flush_a + flush_b)
        return cal_p, flush_p

    cal_abp,  flush_abp  = _detect_one(abp)
    cal_cvp,  flush_cvp  = _detect_one(cvp)
    return cal_abp, flush_abp, cal_cvp, flush_cvp


# ── step 2 ─────────────────────────────────────────────────────────────────────

def detect_slinger(abp, cvp, fs, excluded,
                   lp_cutoff, hp_cutoff, spec_fmax, k):
    """
    Detect resonance / ringing artefacts (slinger).

    Strategy (lazy):
      1. Estimate dominant cardiac frequency f_dom from non-excluded peaks.
      2. Compute ringing_ratio(t) = P_high(t) / P_total(t) where P_high is
         spectrogram power above 2 × f_dom.
      3. Flag windows where ringing_ratio >> baseline + k × IQR.

    Returns
    -------
    list of (start, end) tuples
    """
    raise NotImplementedError


# ── step 3 ─────────────────────────────────────────────────────────────────────

def detect_gasbubble(abp, cvp, fs, excluded,
                     hp_cutoff, spec_fmax, k):
    """
    Detect gas-bubble artefacts (damped pulse with elevated diastolic).

    Strategy (lazy):
      1. Compute HP_env(t) = rolling std of the high-passed signal.
      2. Compute P_cardiac(t) from the spectrogram at f_dom.
      3. Flag windows where both HP_env and P_cardiac drop below their
         baselines by more than k × IQR.

    Returns
    -------
    list of (start, end) tuples
    """
    raise NotImplementedError


# ── step 4 ─────────────────────────────────────────────────────────────────────

def detect_transducer_hoog(abp, cvp, fs, excluded,
                            lp_cutoff, k):
    """
    Detect transducer-repositioning artefacts (ABP baseline shift).

    Strategy (lazy):
      1. Compute LP_ABP(t) and LP_CVP(t).
      2. Flag periods where LP_ABP deviates from its baseline by > k × IQR
         while LP_CVP remains within its baseline.

    Returns
    -------
    list of (start, end) tuples
    """
    raise NotImplementedError


def detect_infuus(abp, cvp, fs, excluded,
                  lp_cutoff, k):
    """
    Detect IV-infusion artefacts on the CVP channel (infuus op CVD).

    Strategy (lazy):
      1. Compute LP_CVP(t) and LP_ABP(t).
      2. Flag periods where LP_CVP deviates from its baseline by > k × IQR
         while LP_ABP remains within its baseline.

    Returns
    -------
    list of (start, end) tuples
    """
    raise NotImplementedError
