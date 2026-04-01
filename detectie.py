"""
detectie.py — Artefact detection functions for arterial and venous pressure signals.

Provides FFT-based frequency analysis, systolic peak detection, and detection of
flush, calibration, and gas bubble artefacts. Written for KT3401 - Artefact Detection.
"""

import numpy as np
from scipy.signal import find_peaks, butter, filtfilt


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_periods(mask, min_samples):
    """
    Return contiguous True-runs in a boolean mask that are at least min_samples long.

    Parameters
    ----------
    mask : 1D bool array
    min_samples : int  minimum duration in samples

    Returns
    -------
    list of (start_idx, end_idx) tuples (end is exclusive)
    """
    changes = np.diff(mask.astype(int), prepend=0, append=0)
    starts  = np.where(changes ==  1)[0]
    ends    = np.where(changes == -1)[0]
    return [(s, e) for s, e in zip(starts, ends) if e - s >= min_samples]


def _merge_periods(periods, max_gap):
    """
    Merge consecutive periods whose gap is smaller than max_gap samples.

    Parameters
    ----------
    periods : list of (start_idx, end_idx) tuples
    max_gap : int  maximum gap in samples to bridge

    Returns
    -------
    list of merged (start_idx, end_idx) tuples
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

    Parameters
    ----------
    n_samples       : int  total signal length
    artifact_periods: list of (start_idx, end_idx) tuples

    Returns
    -------
    1D bool array of length n_samples
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

    Parameters
    ----------
    signal : 1D array-like
    fs     : float  sampling rate (Hz)
    frange : (float, float)  frequency band of interest (Hz), default 0.5–10 Hz

    Returns
    -------
    freqs        : frequency bins clipped to frange (Hz)
    magnitudes   : FFT magnitudes at each frequency bin
    dominant_freq: frequency with highest magnitude within frange (Hz)
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

    Parameters
    ----------
    signal : 1D array-like

    Returns
    -------
    rms : float
    """
    return np.sqrt(np.mean(np.asarray(signal, dtype=float) ** 2))


def detect_peaks_abp(signal, fs, rms_factor=0.5, hr_range=(0.5, 3.0),
                     physiological_max=None):
    """
    Detect systolic peaks in an ABP (or CVP) signal.

    The dominant cardiac frequency from the FFT sets the minimum inter-peak
    distance (60 % of the expected cycle length). The RMS amplitude scaled by
    rms_factor sets the minimum peak prominence.

    Parameters
    ----------
    signal            : 1D array-like  pressure values
    fs                : float          sampling rate (Hz)
    rms_factor        : float          prominence = rms_factor * RMS (default 0.5)
    hr_range          : (float, float) heart-rate search band in Hz (default 0.5–3.0)
    physiological_max : float or None  if given, only samples below this value are
                        used to compute the RMS prominence threshold (prevents large
                        artefact spikes from inflating the threshold and masking
                        normal-amplitude peaks)

    Returns
    -------
    peaks               : indices of detected systolic peaks
    dominant_freq       : dominant cardiac frequency (Hz)
    prominence_threshold: prominence value used (mmHg)
    """
    signal = np.asarray(signal, dtype=float)
    _, _, dominant_freq = compute_fft(signal, fs, frange=hr_range)

    min_distance = int(fs / dominant_freq * 0.6)

    if physiological_max is not None:
        normal_samples = signal[signal <= physiological_max]
        rms_signal = normal_samples if len(normal_samples) > 1 else signal
    else:
        rms_signal = signal
    prominence_threshold = compute_rms(rms_signal) * rms_factor

    peaks, _ = find_peaks(signal, distance=min_distance, prominence=prominence_threshold)
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

    Parameters
    ----------
    signal           : 1D array-like
    fs               : float  sampling rate (Hz)
    dominant_freq    : float  dominant cardiac frequency (Hz)
    peaks            : array  indices of detected systolic peaks
    physiological_max: float or None  if given, only peaks below this value are
                       used to compute the flush threshold (prevents long elevated
                       artefact periods from inflating the baseline estimate)

    Returns
    -------
    flush_periods   : list of (start_idx, end_idx) tuples
    cal_periods     : list of (start_idx, end_idx) tuples
    flush_threshold : float  average peak height (mmHg)
    cal_threshold   : float  0.25 × average peak height (mmHg)
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

    Parameters
    ----------
    signal             : 1D array-like
    fs                 : float  sampling rate (Hz)
    dominant_freq      : float  dominant cardiac frequency (Hz)
    artifact_periods   : list of (start_idx, end_idx) tuples to exclude
    physiological_max  : float or None  if given, only samples below this
                         value are used to compute the prominence threshold

    Returns
    -------
    peaks : array of peak indices in the original signal coordinate system
    """
    signal       = np.asarray(signal, dtype=float)
    min_distance = int(fs / dominant_freq * 0.6)

    artifact_mask  = _build_artifact_mask(len(signal), artifact_periods)
    clean_sections = _find_periods(~artifact_mask, min_samples=min_distance * 2)

    all_peaks = []
    for s, e in clean_sections:
        section = signal[s:e]
        if physiological_max is not None:
            normal_samples = section[section <= physiological_max]
            prominence = np.std(normal_samples) if len(normal_samples) > 1 else np.std(section)
        else:
            prominence = np.std(section)
        local_peaks, _ = find_peaks(section, distance=min_distance, prominence=prominence)
        all_peaks.extend(local_peaks + s)

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

    Parameters
    ----------
    signal          : 1D array-like  CVP values
    fs              : float          sampling rate (Hz)
    dominant_freq   : float          dominant cardiac frequency (Hz)
    flush_threshold : float          signal level above which a period is
                                     considered elevated (same as flush_threshold
                                     returned by detect_artifacts)
    hp_cutoff       : float          high-pass filter cutoff frequency (Hz),
                                     default 0.3 Hz

    Returns
    -------
    infuus_periods : list of (start_idx, end_idx) tuples
    flush_periods  : list of (start_idx, end_idx) tuples
    signal_hp      : 1D array  high-pass filtered signal (full length)
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

    for s, e in elevated_periods:
        section_hp = signal_hp[s:e]
        n = e - s

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

        for sub_s, sub_e in _find_periods(osc_mask,  min_samples):
            infuus_periods.append((s + sub_s, s + sub_e))
        for sub_s, sub_e in _find_periods(~osc_mask, min_samples):
            flush_periods.append((s + sub_s, s + sub_e))

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

    Parameters
    ----------
    signal           : 1D array-like
    fs               : float  sampling rate (Hz)
    dominant_freq    : float  dominant cardiac frequency (Hz)
    peaks            : array  clean systolic peak indices (from redetect_peaks_clean)
    artifact_periods : list of (start_idx, end_idx) tuples to exclude (optional)

    Returns
    -------
    gasbubble_periods : list of (start_idx, end_idx) tuples
    avg_systolic      : float  mean systolic level used as threshold (mmHg)
    avg_diastolic     : float  mean diastolic level used as threshold (mmHg)
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
    artifact_ends = [e for _, e in artifact_periods]
    snapped = []
    for s, e in gasbubble_periods:
        new_s = next((art_e for art_e in artifact_ends if 0 < s - art_e <= max_gap), s)
        snapped.append((new_s, e))

    return snapped, avg_systolic, avg_diastolic
