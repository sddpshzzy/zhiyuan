from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
LABELS = {
    "G0_current_geology": "G0 current",
    "G1_plus_CAD_solid": "G1 + solid",
    "G2_plus_TP_CVOCA_current": "G2 + CVOCA",
    "G3_plus_solid_TP_CVOCA": "G3 + both",
    "G4_solid_TP_CVOCA_top32": "G4 selected",
}


def main() -> None:
    frame = pd.read_csv(ROOT / "results" / "pooled_metrics.csv")
    order = list(LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=False)
    for axis, scheme in zip(axes, ("drillhole", "spatial_block")):
        subset = frame[frame.scheme == scheme]
        for model, color in (("catboost", "#28618f"), ("xgboost", "#d95f02")):
            values = subset[subset.model == model].set_index("feature_contract").reindex(order)
            axis.plot(range(len(order)), values.r2, marker="o", label=model, color=color)
        axis.axhline(0.0, color="#55606a", linewidth=0.8)
        axis.set_xticks(range(len(order)), [LABELS[item] for item in order], rotation=28, ha="right")
        axis.set_ylabel("Pooled outer-fold R²")
        axis.set_title("Held-out drillholes" if scheme == "drillhole" else "Contiguous spatial blocks")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Strict representation-increment comparison")
    fig.tight_layout()
    output = ROOT / "figures" / "pooled_r2_reproduced.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()

