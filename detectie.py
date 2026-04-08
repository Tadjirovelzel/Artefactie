"""
detectie.py — Artefact detection functions for arterial and venous pressure signals.

Provides FFT-based frequency analysis, systolic peak detection, and detection of
flush, calibration, and gas bubble artefacts. Written for KT3401 - Artefact Detection.
"""

import numpy as np
from scipy.signal import find_peaks, butter, filtfilt, welch
from scipy.integrate import trapezoid




# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_periods(mask, min_samples):
    """
    Return contiguous True-runs in a boolean mask that are at least min_samples long.
    """
    changes = np.diff(mask.astype(int), prepend=0, append=0)
    starts  = np.where(changes ==  1)[0]
    ends    = np.where(changes == -1)[0]
    return [(s, e) for s, e in zip(starts, ends) if e - s >= min_samples]


def _merge_periods(periods, max_gap):
    """
    Merge consecutive periods whose gap is smaller than max_gap samples.
    """
    if not periods:
        return periods
    merged = [list(periods[0])]
    for s, e in periods[1:]:
        if s - merged[-1][1] <= max_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [tuple(p) for p in merged]


def _build_artifact_mask(n_samples, artifact_periods):
    """
    Build a boolean mask that is True inside every artifact period.
    """
    mask = np.zeros(n_samples, dtype=bool)
    for s, e in artifact_periods:
        mask[s:min(e, n_samples)] = True
    return mask


# ---------------------------------------------------------------------------
# Public analysis functions
# ---------------------------------------------------------------------------

def compute_fft(signal, fs, frange=(0.5, 10)):
    """
    Compute the FFT of a signal and return the dominant frequency within a
    physiologically relevant frequency band.
    """
    signal = np.asarray(signal, dtype=float)
    N = len(signal)

    freqs      = np.fft.rfftfreq(N, d=1 / fs)
    magnitudes = np.abs(np.fft.rfft(signal)) / N
 
    band = (freqs >= frange[0]) & (freqs <= frange[1])
    freqs      = freqs[band]
    magnitudes = magnitudes[band]

    dominant_freq = freqs[np.argmax(magnitudes)]
    return freqs, magnitudes, dominant_freq


def compute_rms(signal):
    """
    Compute the Root Mean Square (RMS) of a signal.
    """
    return np.sqrt(np.mean(np.asarray(signal, dtype=float) ** 2))


def detect_peaks_abp(signal, fs, rms_factor=0.5, hr_range=(0.5, 3.0),
                     physiological_max=None):
    """
    Detect systolic peaks in an ABP (or CVP) signal.

    The dominant cardiac frequency from the FFT sets the minimum inter-peak
    distance (60 % of the expected cycle length). The RMS amplitude scaled by
    rms_factor sets the minimum peak prominence.
    """
    signal = np.asarray(signal, dtype=float)
    dominant_freq = compute_fft(signal, fs, frange=hr_range)[2]

    min_distance = int(fs / dominant_freq * 0.6)

    if physiological_max is not None:
        normal_samples = signal[signal <= physiological_max]
        rms_signal = normal_samples if len(normal_samples) > 1 else signal
    else:
        rms_signal = signal
    prominence_threshold = compute_rms(rms_signal) * rms_factor

    peaks = find_peaks(signal, distance=min_distance, prominence=prominence_threshold)[0]
    return peaks, dominant_freq, prominence_threshold


def detect_artifacts(signal, fs, dominant_freq, peaks, physiological_max=None):
    """
    Detect flush-type and calibration artefacts based on signal level and duration.

    Flush artefact      : signal stays ABOVE the average peak height for more
                          than 2× the dominant cardiac interval.
    Calibration artefact: signal stays BELOW 25 % of the average peak height
                          for more than 2× the dominant cardiac interval.
                          (During calibration the transducer is open to air,
                          driving the signal to ~0 mmHg.)
    """
    signal      = np.asarray(signal, dtype=float)
    min_samples = int(2 * fs / dominant_freq)

    peak_heights = signal[peaks]
    if physiological_max is not None:
        normal_peaks = peak_heights[peak_heights <= physiological_max]
        peak_heights = normal_peaks if len(normal_peaks) > 0 else peak_heights
    flush_threshold = np.mean(peak_heights)
    cal_threshold   = flush_threshold * 0.25

    flush_periods = _find_periods(signal > flush_threshold, min_samples)
    cal_periods   = _find_periods(signal < cal_threshold,   min_samples)

    return flush_periods, cal_periods, flush_threshold, cal_threshold


def redetect_peaks_clean(signal, fs, dominant_freq, artifact_periods,
                         physiological_max=None):
    """
    Re-run systolic peak detection on the signal sections that fall outside
    all artifact periods.

    Each clean section gets its own prominence threshold (std of that section)
    so that low-amplitude sections (e.g. after a flush) are handled correctly.

    For signals with a known physiological ceiling (e.g. CVP ≤ 25 mmHg), the
    optional physiological_max parameter restricts the std calculation to
    samples below that value. This prevents long elevated artefact periods
    from inflating the prominence threshold and causing missed peaks.
    """
    signal       = np.asarray(signal, dtype=float)
    min_distance = int(fs / dominant_freq * 0.6)

    artifact_mask  = _build_artifact_mask(len(signal), artifact_periods)
    clean_sections = _find_periods(~artifact_mask, min_samples=min_distance * 2)

    all_peaks = []
    for start, end in clean_sections:
        section = signal[start:end]
        if physiological_max is not None:
            normal_samples = section[section <= physiological_max]
            prominence = np.std(normal_samples) if len(normal_samples) > 1 else np.std(section)
        else:
            prominence = np.std(section)
        local_peaks = find_peaks(section, distance=min_distance, prominence=prominence)[0]
        all_peaks.extend(local_peaks + start)

    return np.array(all_peaks, dtype=int)


def detect_infuus_cvp(signal, fs, dominant_freq, flush_threshold, hp_cutoff=0.3):
    """
    Detect infuus (IV infusion) artefacts on a CVP signal.

    An infuus artefact causes the CVP baseline to rise above the flush threshold
    while cardiac oscillations continue on top. This contrasts with a true flush,
    which also elevates the signal but suppresses oscillations.

    Method
    ------
    1. Find all periods where the signal exceeds flush_threshold for more than
       2× the dominant cardiac interval (same criterion as flush detection).
    2. Apply a high-pass Butterworth filter (default cutoff 0.3 Hz) to each
       elevated period to remove the slow baseline rise.
    3. Attempt peak detection on the high-pass filtered section. If at least
       two cardiac peaks are found, the period is classified as infuus;
       otherwise it is classified as a flush.
    """
    signal      = np.asarray(signal, dtype=float)
    min_samples = int(fs / dominant_freq)        # 1 cardiac cycle minimum duration
    min_distance = int(fs / dominant_freq * 0.6)

    # High-pass filter to isolate cardiac oscillations
    b, a      = butter(2, hp_cutoff / (fs / 2), btype="high")
    signal_hp = filtfilt(b, a, signal)

    elevated_periods = _find_periods(signal > flush_threshold, min_samples)
    window_samples   = int(fs / dominant_freq * 5)  # 3-cycle analysis window

    infuus_periods = []
    flush_periods  = []

    for start, end in elevated_periods:
        section_hp = signal_hp[start:end]
        n = end - start

        # Sliding-window oscillation mask: True where cardiac peaks are present.
        # Using 50 % overlap so that transitions between infuus and flush are
        # captured at roughly half-window resolution (~1.5 cardiac cycles).
        osc_mask = np.zeros(n, dtype=bool)
        step = max(window_samples // 2, 1)
        for i in range(0, n, step):
            win = section_hp[i : min(i + window_samples, n)]
            if len(win) < min_distance * 2:
                continue
            prom = np.std(win) * 0.5
            if prom == 0:
                continue
            peaks_in_win, _ = find_peaks(win, distance=min_distance, prominence=prom)
            if len(peaks_in_win) >= 1:
                osc_mask[i : i + len(win)] = True

        for sub_start, sub_end in _find_periods(osc_mask,  min_samples):
            infuus_periods.append((start + sub_start, start + sub_end))
        for sub_start, sub_end in _find_periods(~osc_mask, min_samples):
            flush_periods.append((start + sub_start, start + sub_end))

    return infuus_periods, flush_periods, signal_hp


def detect_gasbubble(signal, fs, dominant_freq, peaks, artifact_periods=None):
    """
    Detect gas-bubble artefacts by tracking the beat-to-beat systolic and
    diastolic envelopes.

    A gas-bubble artefact is flagged when all three conditions hold simultaneously
    for more than 3× the dominant cardiac interval:

      1. Systolic envelope  < average systolic pressure
      2. Diastolic envelope > average diastolic pressure
      3. Instantaneous pulse pressure < 60 % of average pulse pressure
         (the pulse-pressure collapse is the primary discriminator)

    Flush and calibration periods supplied in artifact_periods are excluded from
    both the diastolic baseline and the artefact search, and neighbouring
    gasbubble segments within 6 dominant intervals are merged into one.
    Detected periods are snapped back to the end of any immediately preceding
    artefact period.
    """
    signal           = np.asarray(signal, dtype=float)
    artifact_periods = artifact_periods or []

    min_samples  = int(3 * fs / dominant_freq)
    min_distance = int(fs / dominant_freq * 0.6)
    max_gap      = int(6 * fs / dominant_freq)

    # Exclude artefact regions from diastolic trough baseline
    artifact_mask  = _build_artifact_mask(len(signal), artifact_periods)
    all_troughs, _ = find_peaks(-signal, distance=min_distance)
    clean_troughs  = all_troughs[~artifact_mask[all_troughs]]

    avg_systolic  = np.mean(signal[peaks])
    avg_diastolic = np.mean(signal[clean_troughs]) if len(clean_troughs) > 0 else 0.0
    avg_pp        = avg_systolic - avg_diastolic

    # Interpolate continuous systolic and diastolic envelopes
    t_idx          = np.arange(len(signal))
    upper_envelope = np.interp(t_idx, peaks,         signal[peaks])         if len(peaks)         >= 2 else np.full(len(signal), avg_systolic)
    lower_envelope = np.interp(t_idx, clean_troughs, signal[clean_troughs]) if len(clean_troughs) >= 2 else np.full(len(signal), avg_diastolic)
    inst_pp        = upper_envelope - lower_envelope

    # Detect gasbubble mask: all three conditions AND outside known artefacts
    gasbubble_mask = (
        (upper_envelope < avg_systolic) &
        (lower_envelope > avg_diastolic) &
        (inst_pp        < 0.60 * avg_pp) &
        (~artifact_mask)
    )

    gasbubble_periods = _merge_periods(_find_periods(gasbubble_mask, min_samples), max_gap)

    # Snap period starts to the end of any immediately preceding artefact
    artifact_ends = [end for _, end in artifact_periods]
    snapped = []
    for start, end in gasbubble_periods:
        new_start = next((art_end for art_end in artifact_ends if 0 < start - art_end <= max_gap), start  - max_gap)
        snapped.append((new_start, end))

    return snapped, avg_systolic, avg_diastolic





# def detect_transducer_shift(signal_abp, signal_cvp, fs,
#                                 lp_cutoff=0.05,
#                                 k_abp=1.0,
#                                 k_cvp=1.0,
#                                 min_duration=2.0):
#     """
#     Detecteer transducer-hoog gebied op basis van:
#     - sterke low-pass baseline
#     - baseline-daling t.o.v. (median - k * std)
#     - ABP én CVP moeten beide laag zijn

#     Parameters
#     ----------
#     signal_abp, signal_cvp : arrays
#         Druksignalen
#     fs : float
#         Sampling rate
#     lp_cutoff : float
#         Low-pass cutoff (Hz)
#     k_abp, k_cvp : float
#         Aantal standaarddeviaties onder normaal
#     min_duration : float
#         Minimale duur van artefact (s)

#     Returns
#     -------
#     periods : list of (start, end)
#     base_abp, base_cvp : arrays
#         Low-pass baselines
#     """

#     # --- 1. Sterke low-pass baseline ---
#     b, a = butter(4, lp_cutoff / (fs/2), btype="low")
#     base_abp = filtfilt(b, a, signal_abp)
#     base_cvp = filtfilt(b, a, signal_cvp)

#     # --- 2. Normale baseline + variatie ---
#     med_abp = np.median(base_abp)
#     med_cvp = np.median(base_cvp)

#     sd_abp = np.std(base_abp)
#     sd_cvp = np.std(base_cvp)

#     # --- 3. SD-gebaseerde daling ---
#     mask_abp = base_abp < (med_abp - k_abp * sd_abp)
#     mask_cvp = base_cvp < (med_cvp - k_cvp * sd_cvp)

#     # --- 4. Combineer: ABP én CVP moeten laag zijn ---
#     final_mask = mask_abp & mask_cvp

#     # --- 5. Vind periodes ---
#     min_samples = int(min_duration * fs)
#     periods = _find_periods(final_mask, min_samples)

#     return periods, base_abp, base_cvp

# def detect_transducer_shift(abp, cvp, fs,
#                             lp_cutoff=0.05,
#                             k_baseline=0.4,
#                             min_duration=2.0,
#                             band_width=0.4,
#                             power_ratio_min=0.05,
#                             recovery_time=10.0):

#     # --- 1. Low-pass baseline ---
#     b, a = butter(4, lp_cutoff / (fs/2), btype="low")
#     base_abp = filtfilt(b, a, abp)
#     base_cvp = filtfilt(b, a, cvp)

#     # --- 2. Baseline-daling ---
#     abp_low = base_abp < (np.median(base_abp) - k_baseline * np.std(base_abp))
#     cvp_low = base_cvp < (np.median(base_cvp) - k_baseline * np.std(base_cvp))

#     # --- 3. Dominante hartslagfrequentie ---
#     f, Pxx = welch(abp, fs, nperseg=4*fs)
#     dom = f[np.argmax(Pxx)]

#     # --- 4. Sliding bandpower ---
#     win = int(2 * fs)
#     bp = np.zeros_like(abp)
#     for i in range(len(abp) - win):
#         seg = abp[i:i+win]
#         f2, P2 = welch(seg, fs, nperseg=win)
#         mask = (f2 > dom - band_width) & (f2 < dom + band_width)
#         bp[i:i+win] = trapezoid(P2[mask], f2[mask])

#     # normale bandpower
#     bp_norm = np.median(bp)
#     bp_ok = bp > (power_ratio_min * bp_norm)

#     # --- 5. State machine: calibratie-blokkade ---
#     state = np.zeros(len(abp), dtype=int)  # 0=NORMAL, 1=CALIB
#     state[~bp_ok] = 1
#     print("abp_low true:", np.mean(abp_low))
#     print("cvp_low true:", np.mean(cvp_low))
#     print("bp_ok true:", np.mean(bp_ok))

#     # herstel pas na X seconden stabiel bp_ok
#     rec = int(recovery_time * fs)
#     for i in range(1, len(state)):
#         if state[i] == 0 and state[i-1] == 1:
#             if not np.all(bp_ok[i:i+rec]):
#                 state[i] = 1

#     # --- 6. Transducer shift alleen in NORMAL state ---
#     candidate = abp_low & cvp_low & bp_ok
#     final_mask = candidate & (state == 0)

#     # --- 7. Periodes ---
#     min_samples = int(min_duration * fs)
#     periods = _find_periods(final_mask, min_samples)

#     return periods, base_abp, base_cvp

def detect_transducer_shift(abp, cvp, fs,
                            lp_cutoff=0.05,
                            k_baseline=0.5,
                            min_duration=2.0,
                            block_after_calib=10.0,
                            calibration_periods=None):

    # --- 1. Low-pass baseline ---
    b, a = butter(4, lp_cutoff / (fs/2), btype="low")
    base_abp = filtfilt(b, a, abp)
    base_cvp = filtfilt(b, a, cvp)

    # --- 2. Baseline-daling ---
    abp_low = base_abp < (np.median(base_abp) - k_baseline * np.std(base_abp))
    cvp_low = base_cvp < (np.median(base_cvp) - k_baseline * np.std(base_cvp))

    # Kandidaten: ABP én CVP moeten dalen
    candidate = abp_low & cvp_low

    # --- 3. Calibratie-blokkade ---
    if calibration_periods is not None:
        block_mask = np.zeros(len(abp), dtype=bool)
        extra = int(block_after_calib * fs)

        for start, end in calibration_periods:
            i0 = int(start * fs)
            i1 = min(len(abp), int(end * fs) + extra)
            block_mask[i0:i1] = True

        # blokkeer alles binnen de calibratie + marge
        candidate = candidate & ~block_mask

    # --- 4. Periodes ---
    min_samples = int(min_duration * fs)
    periods = _find_periods(candidate, min_samples)

    return periods, base_abp, base_cvp
