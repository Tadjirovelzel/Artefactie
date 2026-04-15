import matplotlib.pyplot as plt
import numpy as np


def visualize_detected_artifacts(t, abp, cvp, results, title="Gedetecteerde artefacten"):
    t = np.asarray(t)
    abp = np.asarray(abp)
    cvp = np.asarray(cvp)

    fig, ax = plt.subplots(figsize=(16, 6))

    ax.plot(t, abp, label="ABP", linewidth=1.0)
    ax.plot(t, cvp, label="CVP", linewidth=1.0)

    y_min = min(np.min(abp), np.min(cvp))
    y_max = max(np.max(abp), np.max(cvp))
    y_text = y_max - 0.08 * (y_max - y_min)

    color_map = {
        "calibratie_core": "red",
        "border_calibratie": "gold",
        "flush": "purple",
        "slinger": "orange",
        "infuus_op_cvp": "green",
        "transducer_hoog": "brown",
        "gasbel": "cyan",
        "geen_artefact": "gray",
        "onbekend_artefact": "pink",
    }

    used_labels = set()

    for seg, label, features in results:
        start_time = seg["start_time"]
        stop_time = seg["stop_time"]

        color = color_map.get(label, "gray")
        legend_label = label if label not in used_labels else None

        ax.axvspan(start_time, stop_time, color=color, alpha=0.25, label=legend_label)
        used_labels.add(label)

        x_mid = 0.5 * (start_time + stop_time)
        text_label = f"{label}\n{start_time:.1f}-{stop_time:.1f}s"

        ax.text(
            x_mid,
            y_text,
            text_label,
            ha="center",
            va="top",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                alpha=0.8,
                edgecolor=color
            )
        )

    ax.set_xlabel("Tijd [s]")
    ax.set_ylabel("Druk [mmHg]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()