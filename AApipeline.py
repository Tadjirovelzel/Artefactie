from readArtefacts import read_artefacts
from pathlib import Path
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog

def _get_project_root() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd()

def select_artifact_file() -> Path:
    project_root = _get_project_root()
    data_dir = project_root / "data" / "KT3401_AFdata_2025"

    root = tk.Tk()
    root.withdraw()          # verberg hoofdvenster
    root.attributes("-topmost", True)

    file_path = filedialog.askopenfilename(
        title="Kies een artefactbestand",
        initialdir=str(data_dir),
        filetypes=[
            ("Excel files", "*.xlsx *.xls"),
            ("All files", "*.*"),
        ],
    )

    root.destroy()

    if not file_path:
        raise FileNotFoundError("Geen bestand geselecteerd.")

    return Path(file_path)

path = select_artifact_file()
t, abp, cvp = read_artefacts(path, fs=100)

print(f"Gekozen bestand: {path.name}")


def find_start_stop(mask):
    # create a mask of boolean values
    mask = np.asarray(mask, dtype=bool)
    # pad with False at both ends 
    x = np.r_[False, mask, False]
    # find where the value False changes to True 
    starts = np.where(~x[:-1] & x[1:])[0]
    # find where the value True changes to False
    stops  = np.where(x[:-1] & ~x[1:])[0] - 1
    return list(zip(starts, stops))


def detect_calibration(t, abp, cvp,
                       window_size=0.5, min_duration_size=3.0,
                       abp_level_thr=10, cvp_level_thr=3,
                       abp_std_thr=5, cvp_std_thr=1.5):

    # determine frequency of signal, window length and minimum duration between start and stop
    fs = 1 / np.median(np.diff(t))
    w = max(3, int(window_size * fs))
    min_len = int(min_duration_size * fs)

    abp_s = pd.Series(abp)
    cvp_s = pd.Series(cvp)

    abp_med = abp_s.rolling(w, center=True, min_periods=2).median()
    cvp_med = cvp_s.rolling(w, center=True, min_periods=2).median()

    abp_std = abp_s.rolling(w, center=True, min_periods=1).std().fillna(0)
    cvp_std = cvp_s.rolling(w, center=True, min_periods=1).std().fillna(0)

    # create a mask where both ABP and CVP are below their respective thresholds
    mask = (
        (abp_med.abs() < abp_level_thr) &
        (cvp_med.abs() < cvp_level_thr) &
        (abp_std < abp_std_thr) &
        (cvp_std < cvp_std_thr)
    )

    # collect times of start and stop
    times = [(s, e) for s, e in find_start_stop(mask) if (e - s + 1) >= min_len]

    # save times of start and stop in a list of dictionaries
    artifacts = []
    for s, e in times:
        artifacts.append({
            "start_time": t[s],
            "stop_time": t[e],
            "duration_s": t[e] - t[s]
        })

    return artifacts, mask

artifacts, mask = detect_calibration(t, abp, cvp)
print(artifacts)




# ____________________________________________
# Artefact classificatie en feature extractie
# ____________________________________________
from AAfeatures import compute_artifact_features
from AAdecision_tree import classify_artifact
from AAsegments import detect_candidate_segments, relabel_calibration_borders
from AAvisualise import visualize_detected_artifacts

df = pd.DataFrame({
    "Time": t,
    "ABP": abp,
    "CVP": cvp,
})

segments, debug = detect_candidate_segments(df)

results = []

for seg in segments:
    features = compute_artifact_features(df, seg["start_idx"], seg["stop_idx"])
    label = classify_artifact(features)

    results.append({
        "segment": seg,
        "label": label,
        "features": features,
    })


from AAfeatures import compute_border_features
from AAdecision_tree import is_border_of_calibration

def relabel_calibration_borders(results, df):
    if not results:
        return results

    results = sorted(results, key=lambda r: r["segment"]["start_time"])

    for i, r in enumerate(results):
        if r["label"] != "calibratie_core":
            continue

        core_seg = r["segment"]

        # linker buur
        if i - 1 >= 0:
            left = results[i - 1]
            bf_left = compute_border_features(
                df,
                left["segment"]["start_idx"],
                left["segment"]["stop_idx"],
                side="left"
            )
            if is_border_of_calibration(left["segment"], left["label"], bf_left, core_seg):
                left["label"] = "border_calibratie"

        # rechter buur
        if i + 1 < len(results):
            right = results[i + 1]
            bf_right = compute_border_features(
                df,
                right["segment"]["start_idx"],
                right["segment"]["stop_idx"],
                side="right"
            )
            if is_border_of_calibration(right["segment"], right["label"], bf_right, core_seg):
                right["label"] = "border_calibratie"

    return results


results = relabel_calibration_borders(results)

visualize_detected_artifacts(
    t,
    abp,
    cvp,
    [(r["segment"], r["label"], r["features"]) for r in results],
    title=f"Gedetecteerde artefacten - {path.name}"
)
# ____________________________________________
# Visualisatie van gedetecteerde kalibraties
# ____________________________________________

# def visualize_calibration(t, abp, cvp, mask):
#     import matplotlib.pyplot as plt

#     plt.figure(figsize=(15, 6))
#     plt.plot(t, abp, label="ABP", color="blue")
#     plt.plot(t, cvp, label="CVP", color="orange")
#     plt.fill_between(t, -50, 50, where=mask, color="red", alpha=0.3, label="Calibration Detected")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Signal Value")
#     plt.title("Calibration Detection in ABP and CVP Signals")
#     plt.legend()
#     plt.grid()
#     plt.show()


# visualize_calibration(t, abp, cvp, mask)