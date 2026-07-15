from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


MODEL_ORDER = ["Mean", "OrdinaryKriging", "XGBoost", "OOFResidualKriging", "ConvexFusion"]
MODEL_LABELS = {
    "Mean": "Training mean",
    "OrdinaryKriging": "Ordinary kriging",
    "XGBoost": "XGBoost",
    "OOFResidualKriging": "OOF residual kriging",
    "ConvexFusion": "Convex fusion",
}
SCHEME_ORDER = ["random_interval", "drillhole", "spatial_block"]
SCHEME_LABELS = {
    "random_interval": "Random interval",
    "drillhole": "Held-out drillhole",
    "spatial_block": "Spatial block",
}
PALETTE = {
    "Mean": "#9CA3AF",
    "OrdinaryKriging": "#2563EB",
    "XGBoost": "#F59E0B",
    "OOFResidualKriging": "#8B5CF6",
    "ConvexFusion": "#059669",
}


def style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_all(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf", "tiff"]:
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=600 if suffix != "pdf" else None)
    plt.close(fig)


def figure_data_and_validation(data_dir: Path, results_dir: Path, output_dir: Path) -> None:
    clean = pd.read_csv(results_dir / "predictions" / "clean_assay_intervals_confidential.csv")
    legacy = pd.read_excel(data_dir / "Processed_Point_Data_审计用.xlsx")
    legacy_audit = pd.read_csv(results_dir / "tables" / "legacy_data_audit.csv").set_index("item")["value"]
    fold = pd.read_csv(results_dir / "tables" / "fold_metrics.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))

    counts = pd.Series(
        {
            "Assay\nintervals": int(float(legacy_audit["original_assay_intervals"])),
            "Discretized\nassay rows": int(float(legacy_audit["legacy_discretized_assayed_rows"])),
            "Legacy\nrows": int(float(legacy_audit["legacy_processed_rows"])),
            "Mean-filled\nrows": int(float(legacy_audit["global_mean_filled_rows"])),
        }
    )
    colors = ["#059669", "#60A5FA", "#D1D5DB", "#DC2626"]
    axes[0, 0].bar(np.arange(len(counts)), counts.values, color=colors)
    axes[0, 0].set_xticks(np.arange(len(counts)), counts.index, fontsize=7.5)
    axes[0, 0].set_ylabel("Number of rows")
    axes[0, 0].set_title("a  Data-support audit", loc="left", fontweight="bold")
    for x, value in enumerate(counts.values):
        axes[0, 0].text(x, value + 250, f"{value:,}", ha="center", va="bottom", fontsize=8)
    axes[0, 0].set_ylim(0, max(counts.values) * 1.17)

    legacy_grade = pd.to_numeric(legacy["TFe"], errors="coerce")
    bins = np.linspace(0, 50, 41)
    axes[0, 1].hist(clean["TFe"], bins=bins, density=True, alpha=0.72, color="#059669", label="Original intervals")
    axes[0, 1].hist(legacy_grade, bins=bins, density=True, histtype="step", linewidth=1.8, color="#DC2626", label="Legacy processed table")
    fill_value = float(legacy_audit["global_mean_fill_value"])
    axes[0, 1].axvline(fill_value, color="#DC2626", linestyle="--", linewidth=1.2, label=f"Global-mean fill ({fill_value:.2f})")
    axes[0, 1].set_xlabel("TFe (%)")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].set_title("b  Artificial grade spike in the legacy table", loc="left", fontweight="bold")
    axes[0, 1].legend(frameon=False, fontsize=7)

    collars = clean.groupby("hole", as_index=False).agg(x=("collar_x", "first"), y=("collar_y", "first"), block=("spatial_block", "first"), n=("interval_id", "size"))
    collars["east_rel"] = collars["x"] - collars["x"].min()
    collars["north_rel"] = collars["y"] - collars["y"].min()
    scatter = axes[1, 0].scatter(
        collars["east_rel"],
        collars["north_rel"],
        c=collars["block"],
        cmap="viridis",
        s=25 + collars["n"] * 0.7,
        edgecolor="white",
        linewidth=0.5,
    )
    axes[1, 0].set_xlabel("Relative easting (m)")
    axes[1, 0].set_ylabel("Relative northing (m)")
    axes[1, 0].set_title("c  Five contiguous spatial blocks", loc="left", fontweight="bold")
    legend = axes[1, 0].legend(*scatter.legend_elements(), title="Block", loc="best", frameon=False, fontsize=7)
    legend.get_title().set_fontsize(8)

    fold_sizes = fold[fold["model"] == "XGBoost"][["scheme", "outer_fold", "n_test"]]
    pivot = fold_sizes.pivot(index="outer_fold", columns="scheme", values="n_test")[SCHEME_ORDER]
    bottom = np.zeros(len(pivot))
    colors_schemes = ["#60A5FA", "#F59E0B", "#8B5CF6"]
    for scheme, color in zip(SCHEME_ORDER, colors_schemes):
        axes[1, 1].bar(pivot.index.astype(str), pivot[scheme], bottom=bottom, color=color, label=SCHEME_LABELS[scheme])
        bottom += pivot[scheme].to_numpy()
    axes[1, 1].set_xlabel("Outer fold")
    axes[1, 1].set_ylabel("Held-out intervals across schemes")
    axes[1, 1].set_title("d  Balanced five-fold evaluation", loc="left", fontweight="bold")
    axes[1, 1].legend(frameon=False, fontsize=7, ncol=1)
    fig.tight_layout()
    save_all(fig, output_dir, "Figure1_data_support_and_validation_design")


def figure_performance(results_dir: Path, output_dir: Path) -> None:
    pooled = pd.read_csv(results_dir / "tables" / "pooled_metrics.csv")
    fig, axes = plt.subplots(2, 3, figsize=(8.0, 5.2), sharex=True)
    for col, scheme in enumerate(SCHEME_ORDER):
        block = pooled[pooled["scheme"] == scheme].set_index("model").loc[MODEL_ORDER]
        x = np.arange(len(MODEL_ORDER))
        axes[0, col].bar(x, block["rmse"], color=[PALETTE[m] for m in MODEL_ORDER])
        axes[1, col].bar(x, block["r2"], color=[PALETTE[m] for m in MODEL_ORDER])
        axes[0, col].set_title(SCHEME_LABELS[scheme], fontweight="bold")
        axes[0, col].set_ylim(0, max(8.0, block["rmse"].max() * 1.12))
        axes[1, col].axhline(0, color="#374151", linewidth=0.8)
        axes[1, col].set_xticks(x, [MODEL_LABELS[m] for m in MODEL_ORDER], rotation=55, ha="right")
        for row, metric in enumerate(["rmse", "r2"]):
            for i, value in enumerate(block[metric]):
                axes[row, col].text(i, value + (0.12 if row == 0 else 0.015), f"{value:.2f}", ha="center", va="bottom", fontsize=6.5)
    axes[0, 0].set_ylabel("Pooled OOF RMSE (% TFe)")
    axes[1, 0].set_ylabel("Pooled OOF R²")
    fig.suptitle("Model ranking changes with validation separation", fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_all(fig, output_dir, "Figure2_validation_aware_model_performance")


def figure_observed_predicted(results_dir: Path, output_dir: Path) -> None:
    pred = pd.read_csv(results_dir / "predictions" / "oof_predictions.csv")
    models = ["XGBoost", "ConvexFusion"]
    schemes = ["drillhole", "spatial_block"]
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 6.2), sharex=True, sharey=True)
    for r, scheme in enumerate(schemes):
        for c, model in enumerate(models):
            block = pred[(pred["scheme"] == scheme) & (pred["model"] == model)]
            ax = axes[r, c]
            ax.scatter(block["TFe"], block["prediction"], s=8, alpha=0.32, color=PALETTE[model], linewidth=0)
            ax.plot([0, 50], [0, 50], linestyle="--", color="#111827", linewidth=0.8)
            rmse = np.sqrt(np.mean((block["prediction"] - block["TFe"]) ** 2))
            r2 = 1 - np.sum((block["prediction"] - block["TFe"]) ** 2) / np.sum((block["TFe"] - block["TFe"].mean()) ** 2)
            ax.text(0.04, 0.94, f"RMSE = {rmse:.2f}\nR² = {r2:.2f}", transform=ax.transAxes, va="top", fontsize=8)
            ax.set_title(f"{SCHEME_LABELS[scheme]} – {MODEL_LABELS[model]}", fontsize=9, fontweight="bold")
            ax.set_xlim(0, 50)
            ax.set_ylim(0, 50)
    for ax in axes[-1, :]:
        ax.set_xlabel("Observed TFe (%)")
    for ax in axes[:, 0]:
        ax.set_ylabel("OOF predicted TFe (%)")
    fig.tight_layout()
    save_all(fig, output_dir, "Figure3_strict_validation_observed_vs_predicted")


def figure_fusion_and_residual(results_dir: Path, output_dir: Path) -> None:
    weights = pd.read_csv(results_dir / "tables" / "fusion_weights.csv")
    fold = pd.read_csv(results_dir / "tables" / "fold_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    sns.boxplot(data=weights, x="scheme", y="xgboost_weight", order=SCHEME_ORDER, color="#FDE68A", width=0.55, ax=axes[0])
    sns.stripplot(data=weights, x="scheme", y="xgboost_weight", order=SCHEME_ORDER, color="#92400E", size=4, ax=axes[0])
    axes[0].set_xticks(range(len(SCHEME_ORDER)), [SCHEME_LABELS[s] for s in SCHEME_ORDER], rotation=20, ha="right")
    axes[0].set_ylabel("Training-OOF XGBoost weight")
    axes[0].set_xlabel("")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("a  Fusion relies mainly on XGBoost", loc="left", fontweight="bold")

    base = fold[fold["model"] == "XGBoost"][["scheme", "outer_fold", "rmse"]].rename(columns={"rmse": "xgb_rmse"})
    compare = fold[fold["model"].isin(["OOFResidualKriging", "ConvexFusion"])].merge(base, on=["scheme", "outer_fold"])
    compare["delta_rmse"] = compare["rmse"] - compare["xgb_rmse"]
    compare["model_label"] = compare["model"].map(MODEL_LABELS)
    sns.barplot(data=compare, x="scheme", y="delta_rmse", hue="model_label", order=SCHEME_ORDER, errorbar="sd", palette=[PALETTE["OOFResidualKriging"], PALETTE["ConvexFusion"]], ax=axes[1])
    axes[1].axhline(0, color="#111827", linewidth=0.8)
    axes[1].set_xticks(range(len(SCHEME_ORDER)), [SCHEME_LABELS[s] for s in SCHEME_ORDER], rotation=20, ha="right")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Fold RMSE difference from XGBoost")
    axes[1].set_title("b  Residual correction is not stable", loc="left", fontweight="bold")
    axes[1].legend(title="", frameon=False, fontsize=7)
    fig.tight_layout()
    save_all(fig, output_dir, "Figure4_fusion_weights_and_residual_stability")


def figure_feature_ablation(results_dir: Path, output_dir: Path) -> None:
    ablation = pd.read_csv(results_dir / "tables" / "feature_ablation_pooled_metrics.csv")
    order = ["coordinates_only", "trajectory_geometry", "full_geology"]
    labels = ["Coordinates only", "Trajectory geometry", "Full geology"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, metric, ylabel in zip(axes, ["rmse", "r2"], ["Pooled OOF RMSE (% TFe)", "Pooled OOF R²"]):
        sns.barplot(data=ablation, x="feature_set", y=metric, hue="scheme", order=order, hue_order=["drillhole", "spatial_block"], palette=["#F59E0B", "#8B5CF6"], ax=ax)
        ax.set_xticks(range(len(order)), labels, rotation=20, ha="right")
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.legend(title="", labels=["Held-out drillhole", "Spatial block"], frameon=False, fontsize=7)
        if metric == "r2":
            ax.axhline(0, color="#111827", linewidth=0.8)
    axes[0].set_title("a  Prediction error", loc="left", fontweight="bold")
    axes[1].set_title("b  Explained variance", loc="left", fontweight="bold")
    fig.suptitle("Lithological information is essential under strict validation", fontsize=10.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_all(fig, output_dir, "Figure5_feature_group_ablation")


def figure_distance_and_high_grade(results_dir: Path, output_dir: Path) -> None:
    distance = pd.read_csv(results_dir / "tables" / "distance_metrics.csv")
    pooled = pd.read_csv(results_dir / "tables" / "pooled_metrics.csv")
    distance = distance[
        distance["scheme"].isin(["drillhole", "spatial_block"])
        & distance["model"].isin(["XGBoost", "ConvexFusion"])
    ]
    quartile_order = ["Q1_near", "Q2", "Q3", "Q4_far"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    sns.lineplot(data=distance, x="distance_quartile", y="rmse", hue="model", style="scheme", hue_order=["XGBoost", "ConvexFusion"], style_order=["drillhole", "spatial_block"], markers=True, dashes=True, sort=False, ax=axes[0], palette=PALETTE)
    axes[0].set_xticks(range(4), ["Q1 near", "Q2", "Q3", "Q4 far"])
    axes[0].set_xlabel("Nearest-training-distance quartile")
    axes[0].set_ylabel("RMSE (% TFe)")
    axes[0].set_title("a  Error increases with separation", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=6.5)

    high = pooled[
        pooled["scheme"].isin(["drillhole", "spatial_block"])
        & pooled["model"].isin(["OrdinaryKriging", "XGBoost", "OOFResidualKriging", "ConvexFusion"])
    ]
    sns.barplot(data=high, x="model", y="high_grade_bias", hue="scheme", order=["OrdinaryKriging", "XGBoost", "OOFResidualKriging", "ConvexFusion"], hue_order=["drillhole", "spatial_block"], palette=["#F59E0B", "#8B5CF6"], ax=axes[1])
    axes[1].axhline(0, color="#111827", linewidth=0.8)
    axes[1].set_xticks(range(4), ["OK", "XGB", "Residual OK", "Fusion"], rotation=20, ha="right")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("High-grade bias (% TFe)")
    axes[1].set_title("b  Systematic high-grade underprediction", loc="left", fontweight="bold")
    axes[1].legend(title="", labels=["Held-out drillhole", "Spatial block"], frameon=False, fontsize=7)
    fig.tight_layout()
    save_all(fig, output_dir, "Figure6_distance_and_high_grade_diagnostics")


def figure_fold_variability(results_dir: Path, output_dir: Path) -> None:
    fold = pd.read_csv(results_dir / "tables" / "fold_metrics.csv")
    strict = fold[fold["scheme"].isin(["drillhole", "spatial_block"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, scheme in zip(axes, ["drillhole", "spatial_block"]):
        block = strict[strict["scheme"] == scheme]
        for model in MODEL_ORDER:
            m = block[block["model"] == model].sort_values("outer_fold")
            ax.plot(m["outer_fold"], m["rmse"], marker="o", linewidth=1.2, markersize=3.5, color=PALETTE[model], label=MODEL_LABELS[model])
        ax.set_xticks(range(1, 6))
        ax.set_xlabel("Outer fold")
        ax.set_title(SCHEME_LABELS[scheme], fontweight="bold")
    axes[0].set_ylabel("RMSE (% TFe)")
    axes[1].legend(frameon=False, fontsize=6.7, loc="best")
    fig.suptitle("Fold-to-fold variability exceeds the small fusion gain", fontsize=10.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_all(fig, output_dir, "Figure7_foldwise_performance_variability")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    style()
    figure_data_and_validation(args.data_dir, args.results_dir, args.output_dir)
    figure_performance(args.results_dir, args.output_dir)
    figure_observed_predicted(args.results_dir, args.output_dir)
    figure_fusion_and_residual(args.results_dir, args.output_dir)
    figure_feature_ablation(args.results_dir, args.output_dir)
    figure_distance_and_high_grade(args.results_dir, args.output_dir)
    figure_fold_variability(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
