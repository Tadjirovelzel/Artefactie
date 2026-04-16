from matplotlib.ticker import MultipleLocator
from pathlib import Path
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import Tk, filedialog


def select_excel_file(initial_dir=None):
    root = Tk()
    root.withdraw()  # hide the small Tk window
    root.attributes("-topmost", True)

    file_path = filedialog.askopenfilename(
        title="Select an Excel file",
        initialdir=base_path / "data" / "KT3401_AFdata_2025",
        filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
    )

    root.destroy()

    if not file_path:
        raise FileNotFoundError("No file was selected.")

    return Path(file_path)


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


def merge_close_segments(segments, max_gap=0.5):
    if not segments:
        return []

    merged = [segments[0]]

    for s, e in segments[1:]:
        last_s, last_e = merged[-1]

        if s - last_e <= max_gap:
            merged[-1] = (last_s, e)
        else:
            merged.append((s, e))

    return merged


def detect_slinger_segments(
     t,
    signal,
    fs,
    hf_low=8,
    hf_high=20,
    baseline_percentile=40,
    k=2.0,
    smooth_window=3,
    fill_gap_bins=2,
    min_duration_sec=0.5,
    merge_gap_sec=1.5,
):
    # Korter venster = minder randproblemen
    nperseg = int(1.0 * fs)
    noverlap = int(0.5 * fs)
    nfft = int(3.0 * fs)

    # Reflect padding
    pad = nperseg // 2
    signal_padded = np.pad(signal, pad_width=pad, mode="reflect")

    # Spectrogram
    f, ts_pad, Sxx = spectrogram(
        signal_padded,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=False,
        scaling="density",
        mode="psd",
    )

    # Tijdas terugzetten naar origineel
    ts = ts_pad - pad / fs

    # Alleen tijdstippen houden waarvan het HELE venster
    # binnen de echte data valt
    half_window = nperseg / (2 * fs)
    duration = t[-1] - t[0]

    valid = (ts >= half_window) & (ts <= duration - half_window)
    ts = ts[valid]
    Sxx = Sxx[:, valid]

    # HF-energie
    hf_band = (f >= hf_low) & (f <= hf_high)
    hf_energy = np.sum(Sxx[hf_band, :], axis=0)
    hf_log = np.log1p(hf_energy)

    # Gladstrijken
    hf_smooth = (
        pd.Series(hf_log)
        .rolling(window=smooth_window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )

    # Baseline
    # baseline_limit = np.percentile(hf_log, baseline_percentile)
    # baseline = hf_smooth[hf_log <= baseline_limit]

    std = np.std(hf_smooth)
    med = np.median(hf_smooth)
    # mad = np.median(np.abs(baseline - med))
    mean = np.mean(hf_smooth)

    # Robust threshold
    threshold = mean + k * std
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
    segments_time = merge_close_segments(segments_time, max_gap=merge_gap_sec)

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
file_path = select_excel_file(initial_dir=base_path)

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
    k=1.5,
    min_duration_sec=1.5,
    fill_gap_bins=2,
)

res_cvp = detect_slinger_segments(
    t, cvp, fs,
    k=0,
    min_duration_sec=1.5,
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

# X-as beter leesbaar maken
for a in ax:
    a.xaxis.set_major_locator(MultipleLocator(1.0))
    a.xaxis.set_minor_locator(MultipleLocator(0.5))
    a.grid(True, which="major", axis="x", alpha=0.4)
    a.grid(True, which="minor", axis="x", alpha=0.2)

plt.tight_layout()
plt.show()