from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.mmex_validation import (
    audit_legacy_processed_data,
    load_clean_intervals,
    load_config,
    run_feature_ablation,
    run_nested_validation,
    write_source_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leakage-safe ore-grade validation experiments.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing the six whitelisted XLSX files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for local, potentially sensitive outputs.")
    parser.add_argument("--config", type=Path, default=Path("config/analysis.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = args.output_dir / "tables"
    predictions_dir = args.output_dir / "predictions"
    tables_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    write_source_manifest(args.data_dir, tables_dir / "source_manifest.csv")
    intervals, audit = load_clean_intervals(
        args.data_dir,
        n_spatial_blocks=int(config["spatial_blocks"]),
        random_seed=int(config["random_seed"]),
    )
    intervals.to_csv(predictions_dir / "clean_assay_intervals_confidential.csv", index=False)
    audit.to_csv(tables_dir / "data_audit.csv", index=False)
    legacy_audit = audit_legacy_processed_data(args.data_dir, intervals)
    legacy_audit.to_csv(tables_dir / "legacy_data_audit.csv", index=False)
    results = run_nested_validation(intervals, config, args.output_dir)
    results.update(run_feature_ablation(intervals, config, results["oof_predictions"]))
    for name, table in results.items():
        target_dir = predictions_dir if name.endswith("predictions") else tables_dir
        table.to_csv(target_dir / f"{name}.csv", index=False)
    with (tables_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": config,
                "data_rows": len(intervals),
                "drillholes": int(intervals["hole"].nunique()),
                "sensitive_outputs_publication_status": "local_only",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
