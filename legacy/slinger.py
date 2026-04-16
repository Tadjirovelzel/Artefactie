from pathlib import Path
from scipy.signal import spectrogram, find_peaks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

base_path = Path(__file__).resolve().parent
file_path = base_path / "data" / "KT3401_AFdata_2025"  / "Infuus_op_CVD/D05Inf-op-CVP_Slinger.xlsx"

print("Bestand:", file_path)
print("Bestaat:", file_path.exists())

df = pd.read_excel(file_path)
df = df.iloc[1:].reset_index(drop=True)

df["Time"] = pd.to_numeric(df["Time"])
df["ABP"] = pd.to_numeric(df["ABP"])
df["CVP"] = pd.to_numeric(df["CVP"])

t = df["Time"].to_numpy()
abp = df["ABP"].to_numpy()
cvp = df["CVP"].to_numpy()

fs = 1 / np.median(np.diff(t))

# -----------------------------
# 2. Spectrogram van ABP
# -----------------------------
f, ts, Sxx = spectrogram(
    abp,
    fs=fs,
    nperseg=int(2 * fs),
    noverlap=int(1 * fs),
    nfft=int(4 * fs)
)

# -----------------------------
# 3. Hoge-frequentie energie
# -----------------------------
hf_band = (f >= 8) & (f <= 20)
hf_energy = np.sum(Sxx[hf_band, :], axis=0)

# log-schaal maakt de toename duidelijker
hf_log = np.log1p(hf_energy)

# -----------------------------
# 4. Drempel bepalen
# -----------------------------
# gebruik alleen de laagste 40% van de HF-energie als baseline
baseline_limit = np.percentile(hf_log, 40)
baseline = hf_log[hf_log <= baseline_limit]

std = np.std(baseline)

med = np.median(baseline)
mad = np.median(np.abs(baseline - med))

peak = np.max(find_peaks(hf_log, height=baseline_limit)[1]["peak_heights"])

threshold = med + 5 * std
candidate = hf_log > threshold

# -----------------------------
# 5. Kleine gaten dichtmaken
# -----------------------------
# eenvoudige rolling max om korte onderbrekingen op te vullen
candidate_smooth = (
    pd.Series(candidate.astype(int))
    .rolling(window=1, center=True, min_periods=1)
    .max()
    .to_numpy()
    .astype(bool)
)

# -----------------------------
# 6. Begin- en eindsegmenten zoeken
# -----------------------------
idx = np.where(candidate_smooth)[0]

if len(idx) == 0:
    print("Geen slingerartefact gevonden")
    start_time = None
    end_time = None
else:
    segments = []
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

    # kies langste segment
    s_spec, e_spec = max(segments, key=lambda x: x[1] - x[0])

    start_time = ts[s_spec] + t[0]
    end_time   = ts[e_spec] + t[0]

    print(f"Begin slinger: {start_time:.2f} s")
    print(f"Einde slinger: {end_time:.2f} s")

# -----------------------------
# 7. Plotten en markeren
# -----------------------------
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, cvp, label="ABP")
if start_time is not None:
    plt.axvspan(start_time, end_time, alpha=0.3, label="Slinger")
plt.ylabel("mmHg")
plt.legend()
plt.title("ABP met gedetecteerd slingergebied")

plt.subplot(3, 1, 2)
plt.pcolormesh(t[0] + ts, f, 10 * np.log10(Sxx + 1e-12), shading="gouraud")
plt.ylim(0, 25)
if start_time is not None:
    plt.axvspan(start_time, end_time, alpha=0.3, color="white")
plt.ylabel("Frequentie [Hz]")
plt.title("Spectrogram")

plt.subplot(3, 1, 3)
plt.plot(t[0] + ts, hf_log, label="HF energie (8-20 Hz)")
plt.axhline(threshold, linestyle="--", label="Drempel")
if start_time is not None:
    plt.axvspan(start_time, end_time, alpha=0.3)
plt.xlabel("Tijd [s]")
plt.legend()
plt.title("Detectiesignaal")

plt.tight_layout()
plt.show()
