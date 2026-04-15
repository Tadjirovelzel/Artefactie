from pathlib import Path
import pandas as pd
import tkinter as tk
from tkinter import filedialog

from readArtefacts import read_artefacts
from AAsegments import detect_candidate_segments
from AAfeatures import compute_artifact_features
from AAdecision_tree import classify_artifact
from AAvisualise import visualize_detected_artifacts
from AAborders import merge_calibration_labels, relabel_calibration_borders


def _get_project_root() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd()


def select_artifact_file() -> Path:
    project_root = _get_project_root()
    data_dir = project_root / "data" / "KT3401_AFdata_2025"

    root = tk.Tk()
    root.withdraw()
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


def load_dataset(path, fs=100):
    t, abp, cvp = read_artefacts(path, fs=fs)

    df = pd.DataFrame({
        "Time": t,
        "ABP": abp,
        "CVP": cvp,
    })

    return t, abp, cvp, df


def run_pipeline(path, fs=100):
    t, abp, cvp, df = load_dataset(path, fs=fs)

    segments, debug = detect_candidate_segments(df)

    results = []
    for seg in segments:
        features = compute_artifact_features(
            df,
            seg["start_idx"],
            seg["stop_idx"]
        )
        label = classify_artifact(features)

        results.append({
            "segment": seg,
            "label": label,
            "features": features,
        })

    results = relabel_calibration_borders(results, df)
    results = merge_calibration_labels(results, df)

    return t, abp, cvp, df, segments, results, debug

def print_results(results): 
    print("\nGedetecteerde segmenten:")
    for i, res in enumerate(results, start=1):
        seg = res["segment"]
        label = res["label"]
        print(
            f"{i:02d}. "
            f"{seg['start_time']:.2f}s - {seg['stop_time']:.2f}s | "
            f"{seg['duration_s']:.2f}s | "
            f"{label}"
        )


def main():
    while True:
        try:
            path = select_artifact_file()
        except FileNotFoundError:
            print("Geen bestand geselecteerd. Programma gestopt.")
            break

        print(f"\nGekozen bestand: {path.name}")

        try:
            t, abp, cvp, df, segments, results, debug = run_pipeline(path, fs=100)
            print_results(results)

            visualize_detected_artifacts(
                t,
                abp,
                cvp,
                [(r["segment"], r["label"], r["features"]) for r in results],
                title=f"Gedetecteerde artefacten - {path.name}"
            )

            print("Figuur gesloten. Kies een nieuw bestand...")

        except Exception as e:
            print(f"Fout bij verwerken van {path.name}: {e}")



if __name__ == "__main__":
    main()