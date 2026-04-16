from pathlib import Path
import signal
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_segments(mask):
    idx = np.where(mask)[0]
    segments = []

    if len(idx) == 0:
        return segments

    start = idx[0]
    prev = idx[0]

    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            segments.append((start, prev))
            start = i
            prev = i

    segments.append((start, prev))
    return segments

def detect_slinger_segments(
    t,
    signal,
    fs,
    hf_low=8,
    hf_high=20,
    baseline_percentile=40,
    k=50.0,
    smooth_window=3,
    fill_gap_bins=2,
    min_duration_sec=1.0,
):
    nperseg = int(1 * fs)
    noverlap = int(0.5 * fs)
    nfft = int(2 * fs)

    pad = nperseg // 2
    signal_padded = np.pad(signal, pad_width=pad, mode="reflect")

    f, ts_pad, Sxx = spectrogram(
        signal_padded,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft
    )

    ts = ts_pad - pad / fs
    half_window = nperseg / (2 * fs)
    duration = t[-1] - t[0]

    valid = (ts >= half_window) & (ts <= duration - half_window)
    ts = ts[valid]
    Sxx = Sxx[:, valid]

    hf_band = (f >= hf_low) & (f <= hf_high)
    hf_energy = np.sum(Sxx[hf_band, :], axis=0)
    hf_log = np.log1p(hf_energy)

    hf_smooth = (
        pd.Series(hf_log)
        .rolling(window=smooth_window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )

    baseline_limit = np.percentile(hf_smooth, baseline_percentile)
    baseline = hf_smooth[hf_smooth <= baseline_limit]

    std = np.std(baseline)
    med = np.median(baseline)
    mad = np.median(np.abs(baseline - med))

    # Robust threshold
    threshold = med + k * std
    candidate = hf_smooth > threshold

    if fill_gap_bins > 1:
        candidate = (
            pd.Series(candidate.astype(int))
            .rolling(window=fill_gap_bins, center=True, min_periods=1)
            .max()
            .to_numpy()
            .astype(bool)
        )

    segments_spec = find_segments(candidate)

    valid_segments_spec = []
    for s, e in segments_spec:
        duration_seg = ts[e] - ts[s]
        if duration_seg >= min_duration_sec:
            valid_segments_spec.append((s, e))

    segments_time = [(t[0] + ts[s], t[0] + ts[e]) for s, e in valid_segments_spec]

    return {
        "f": f,
        "ts": ts,
        "Sxx": Sxx,
        "hf_log": hf_log,
        "hf_smooth": hf_smooth,
        "threshold": threshold,
        "segments_spec": valid_segments_spec,
        "segments_time": segments_time,
    }


# -----------------------------
# 1. Read data
# -----------------------------
base_path = Path(__file__).resolve().parent
file_path = base_path / "data" / "KT3401_AFdata_2025" / "Meerdere_artefacten\D01Transd_hoog_Slinger.xlsx"

print("File:", file_path)
print("Exists:", file_path.exists())

df = pd.read_excel(file_path)
df = df.iloc[1:].reset_index(drop=True)

df["Time"] = pd.to_numeric(df["Time"])
df["ABP"] = pd.to_numeric(df["ABP"])
df["CVP"] = pd.to_numeric(df["CVP"])

t = df["Time"].to_numpy()
abp = df["ABP"].to_numpy()
cvp = df["CVP"].to_numpy()

fs = 1 / np.median(np.diff(t))
print("fs =", fs)

# -----------------------------
# 2. Detect in ABP and CVP
# -----------------------------
res_abp = detect_slinger_segments(
    t, abp, fs,
    k=4.0,
    min_duration_sec=1.0,
    fill_gap_bins=2,
)

res_cvp = detect_slinger_segments(
    t, cvp, fs,
    k=4.0,
    min_duration_sec=1.0,
    fill_gap_bins=2,
)

print("ABP slinger segments:")
for s, e in res_abp["segments_time"]:
    print(f"  {s:.2f} s  ->  {e:.2f} s")

print("CVP slinger segments:")
for s, e in res_cvp["segments_time"]:
    print(f"  {s:.2f} s  ->  {e:.2f} s")

# -----------------------------
# 3. Plot everything
# -----------------------------
fig, ax = plt.subplots(4, 1, figsize=(13, 12), sharex=True)

# ABP signal
ax[0].plot(t, abp, label="ABP")
for i, (s, e) in enumerate(res_abp["segments_time"]):
    ax[0].axvspan(s, e, alpha=0.3, color="tab:red",
                  label="ABP slinger" if i == 0 else None)
ax[0].set_ylabel("mmHg")
ax[0].set_title("ABP with detected slinger artifacts")
ax[0].legend()

# CVP signal
ax[1].plot(t, cvp, label="CVP")
for i, (s, e) in enumerate(res_cvp["segments_time"]):
    ax[1].axvspan(s, e, alpha=0.3, color="tab:orange",
                  label="CVP slinger" if i == 0 else None)
ax[1].set_ylabel("mmHg")
ax[1].set_title("CVP with detected slinger artifacts")
ax[1].legend()

# ABP detection signal
ax[2].plot(t[0] + res_abp["ts"], res_abp["hf_log"], label="HF energy ABP", alpha=0.4)
ax[2].plot(t[0] + res_abp["ts"], res_abp["hf_smooth"], label="HF smooth ABP")
ax[2].axhline(res_abp["threshold"], linestyle="--", label="Threshold ABP")
for s, e in res_abp["segments_time"]:
    ax[2].axvspan(s, e, alpha=0.3, color="tab:red")
ax[2].set_title("Detection signal ABP")
ax[2].legend()

# CVP detection signal
ax[3].plot(t[0] + res_cvp["ts"], res_cvp["hf_log"], label="HF energy CVP", alpha=0.4)
ax[3].plot(t[0] + res_cvp["ts"], res_cvp["hf_smooth"], label="HF smooth CVP")
ax[3].axhline(res_cvp["threshold"], linestyle="--", label="Threshold CVP")
for s, e in res_cvp["segments_time"]:
    ax[3].axvspan(s, e, alpha=0.3, color="tab:orange")
ax[3].set_title("Detection signal CVP")
ax[3].set_xlabel("Time [s]")
ax[3].legend()

plt.tight_layout()
plt.show()