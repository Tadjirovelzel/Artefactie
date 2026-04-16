from __future__ import annotations

import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------
# User-tunable settings
# -------------------------------
CHANNEL_SETTINGS = {
    "ABP": {
        "baseline_window_s": 0.25,
        "feature_window_s": 0.50,
        "trend_window_s": 1.00,
        "z_threshold": 4.0,
        "score_threshold": 10.0,
        "min_region_s": 0.40,
        "merge_gap_s": 1.20,
    },
    "CVP": {
        "baseline_window_s": 0.25,
        "feature_window_s": 0.60,
        "trend_window_s": 1.20,
        "z_threshold": 3.5,
        "score_threshold": 8.0,
        "min_region_s": 0.30,
        "merge_gap_s": 0.80,
    },
}

DEFAULT_FOLDER_CANDIDATES = [
    Path(r"data\KT3401_AFdata_2025"),
    Path.cwd() / "data" / "KT3401_AFdata_2025",
    Path(__file__).resolve().parent / "data" / "KT3401_AFdata_2025",
]


# -------------------------------
# Helpers
# -------------------------------
def robust_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad < 1e-12:
        return np.zeros_like(x)
    return 0.6745 * (x - med) / mad


def rolling_mean(x: np.ndarray, win: int) -> np.ndarray:
    return (
        pd.Series(x)
        .rolling(win, center=True, min_periods=max(1, win // 3))
        .mean()
        .to_numpy()
    )


def rolling_median(x: np.ndarray, win: int) -> np.ndarray:
    return (
        pd.Series(x)
        .rolling(win, center=True, min_periods=max(1, win // 3))
        .median()
        .to_numpy()
    )


def mask_to_regions(mask: np.ndarray, time: np.ndarray, min_region_s: float, merge_gap_s: float):
    mask = np.asarray(mask, dtype=bool).copy()
    time = np.asarray(time, dtype=float)
    if len(mask) == 0:
        return mask, []

    dt = float(np.nanmedian(np.diff(time)))
    max_gap_samples = max(1, int(round(merge_gap_s / dt)))
    min_region_samples = max(1, int(round(min_region_s / dt)))

    # Fill small False gaps between True regions.
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            j = i
            while j < n and not mask[j]:
                j += 1
            gap_len = j - i
            if i > 0 and j < n and gap_len <= max_gap_samples:
                mask[i:j] = True
            i = j
        else:
            i += 1

    # Remove very short True regions.
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            region_len = j - i
            if region_len < min_region_samples:
                mask[i:j] = False
            i = j
        else:
            i += 1

    # Extract regions.
    regions = []
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            start_t = float(time[i])
            end_t = float(time[j - 1])
            regions.append((start_t, end_t))
            i = j
        else:
            i += 1

    return mask, regions


def detect_artifacts(time: np.ndarray, signal: np.ndarray, settings: dict):
    dt = float(np.nanmedian(np.diff(time)))
    fs = 1.0 / dt

    baseline_win = max(5, int(round(settings["baseline_window_s"] * fs)))
    feature_win = max(3, int(round(settings["feature_window_s"] * fs)))
    trend_win = max(3, int(round(settings["trend_window_s"] * fs)))

    if baseline_win % 2 == 0:
        baseline_win += 1
    if feature_win % 2 == 0:
        feature_win += 1
    if trend_win % 2 == 0:
        trend_win += 1

    baseline = rolling_median(signal, baseline_win)
    slow_trend = rolling_mean(signal, trend_win)
    residual = signal - baseline
    slope = np.abs(np.gradient(signal, time))

    residual_rms = np.sqrt(np.maximum(rolling_mean(residual ** 2, feature_win), 0))
    slope_mean = rolling_mean(slope, feature_win)
    trend_deviation = np.abs(signal - slow_trend)

    z_residual = np.clip(robust_zscore(residual_rms), 0, None)
    z_slope = np.clip(robust_zscore(slope_mean), 0, None)
    z_deviation = np.clip(robust_zscore(trend_deviation), 0, None)

    # Combined score. Residual RMS and slope are the most important.
    score = 1.0 * z_residual + 1.0 * z_slope + 0.4 * z_deviation

    raw_mask = (
        ((z_residual > settings["z_threshold"]) & (z_slope > settings["z_threshold"]))
        | (score > settings["score_threshold"])
    )

    clean_mask, regions = mask_to_regions(
        raw_mask,
        time,
        min_region_s=settings["min_region_s"],
        merge_gap_s=settings["merge_gap_s"],
    )

    return {
        "mask": clean_mask,
        "regions": regions,
        "score": score,
        "baseline": baseline,
        "z_residual": z_residual,
        "z_slope": z_slope,
        "z_deviation": z_deviation,
    }


def choose_initial_directory() -> Path:
    for candidate in DEFAULT_FOLDER_CANDIDATES:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return Path.cwd()


def select_file_gui() -> Path | None:
    initialdir = choose_initial_directory()

    root = tk.Tk()
    root.withdraw()
    root.update()

    file_path = filedialog.askopenfilename(
        title="Select a dataset",
        initialdir=str(initialdir),
        filetypes=[
            ("Supported files", "*.xlsx *.xls *.csv"),
            ("Excel files", "*.xlsx *.xls"),
            ("CSV files", "*.csv"),
            ("All files", "*.*"),
        ],
    )

    root.destroy()
    if not file_path:
        return None
    return Path(file_path)


def load_dataset(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    expected = ["Time", "ABP", "CVP"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Expected columns: {expected}")

    # Convert to numeric. This automatically drops the unit row like [s, mmHg, mmHg].
    for col in expected:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=expected).reset_index(drop=True)
    if len(df) < 10:
        raise ValueError("Not enough numeric rows found after cleaning the file.")

    if not np.all(np.diff(df["Time"].to_numpy()) > 0):
        raise ValueError("The Time column must be strictly increasing.")

    return df


def results_to_dataframe(results_by_channel: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for channel, result in results_by_channel.items():
        for start_t, end_t in result["regions"]:
            rows.append(
                {
                    "channel": channel,
                    "start_time_s": start_t,
                    "end_time_s": end_t,
                    "duration_s": end_t - start_t,
                }
            )
    return pd.DataFrame(rows)


def plot_results(df: pd.DataFrame, results_by_channel: dict[str, dict], output_png: Path):
    time = df["Time"].to_numpy()
    t0 = time[0]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Detected slinger / artifact regions")

    for ax, channel in zip(axes, ["ABP", "CVP"]):
        y = df[channel].to_numpy()
        res = results_by_channel[channel]

        ax.plot(time - t0, y, linewidth=1.0, label=channel)
        ax.plot(time - t0, res["baseline"], linewidth=1.0, alpha=0.8, label=f"{channel} baseline")

        ymin = np.nanmin(y)
        ymax = np.nanmax(y)
        for start_t, end_t in res["regions"]:
            ax.axvspan(start_t - t0, end_t - t0, alpha=0.25)

        ax.set_ylabel(channel)
        ax.set_ylim(ymin - 0.05 * (ymax - ymin + 1e-9), ymax + 0.05 * (ymax - ymin + 1e-9))
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.show()


def main():
    try:
        if len(sys.argv) > 1:
            file_path = Path(sys.argv[1])
        else:
            file_path = select_file_gui()
            if file_path is None:
                print("No file selected. Exiting.")
                return

        df = load_dataset(file_path)

        time = df["Time"].to_numpy(dtype=float)
        results_by_channel = {
            "ABP": detect_artifacts(time, df["ABP"].to_numpy(dtype=float), CHANNEL_SETTINGS["ABP"]),
            "CVP": detect_artifacts(time, df["CVP"].to_numpy(dtype=float), CHANNEL_SETTINGS["CVP"]),
        }

        regions_df = results_to_dataframe(results_by_channel)

        output_dir = file_path.parent
        stem = file_path.stem
        output_csv = output_dir / f"{stem}_detected_artifact_regions.csv"
        output_png = output_dir / f"{stem}_detected_artifact_regions.png"

        regions_df.to_csv(output_csv, index=False)
        plot_results(df, results_by_channel, output_png)

        print("\nDetected regions:")
        if regions_df.empty:
            print("No artifact regions detected.")
        else:
            print(regions_df.to_string(index=False))

        print(f"\nSaved region table to: {output_csv}")
        print(f"Saved plot to: {output_png}")

    except Exception as exc:
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Artifact detection error", str(exc))
            root.destroy()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
