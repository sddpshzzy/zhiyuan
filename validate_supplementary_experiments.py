from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from src.qidashan_strict import sha256_file


ROOT = Path(__file__).resolve().parent
SIJIAYING = ROOT / "results" / "sijiaying_blind_external"
SIMULATION = ROOT / "results" / "failure_mechanism_simulation"
OUTPUT = ROOT / "results" / "supplementary_validation"


def _check(
    rows: list[dict[str, object]],
    name: str,
    passed: bool,
    observed: object,
    expected: object,
) -> None:
    rows.append(
        {
            "check": name,
            "passed": bool(passed),
            "observed": observed,
            "expected": expected,
        }
    )


def main() -> None:
    rows: list[dict[str, object]] = []
    development = pd.read_csv(SIJIAYING / "sijiaying_development_predictions.csv")
    development_metrics = pd.read_csv(
        SIJIAYING / "sijiaying_development_metrics.csv"
    ).set_index("model")
    blind = pd.read_csv(
        SIJIAYING / "sijiaying_blind_predictions_ONE_SHOT.csv"
    )
    blind_metrics = pd.read_csv(
        SIJIAYING / "sijiaying_blind_metrics_ONE_SHOT.csv"
    ).set_index("model")
    sentinel = json.loads(
        (SIJIAYING / "BLIND_TARGET_OPENED_ONCE.json").read_text(
            encoding="utf-8"
        )
    )
    freeze = json.loads(
        (SIJIAYING / "partition_and_code_freeze_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    partition = pd.read_csv(
        SIJIAYING / "sijiaying_frozen_hole_partition.csv"
    )
    development_holes = set(
        partition.loc[partition["partition_role"].eq("DEVELOPMENT"), "hole"]
    )
    blind_holes = set(
        partition.loc[partition["partition_role"].eq("BLIND"), "hole"]
    )
    _check(
        rows,
        "sijiaying_development_blind_hole_overlap",
        not bool(set(development["hole"]) & blind_holes),
        len(set(development["hole"]) & blind_holes),
        0,
    )
    _check(
        rows,
        "sijiaying_blind_development_hole_overlap",
        not bool(set(blind["hole"]) & development_holes),
        len(set(blind["hole"]) & development_holes),
        0,
    )
    _check(
        rows,
        "sijiaying_blind_block_is_frozen_zero",
        set(blind["spatial_block"]) == {0},
        sorted(blind["spatial_block"].unique().tolist()),
        [0],
    )
    for label, predictions, metrics in [
        ("development", development, development_metrics),
        ("blind", blind, blind_metrics),
    ]:
        for model, frame in predictions.groupby("model"):
            r2 = float(r2_score(frame["observed"], frame["predicted"]))
            rmse = float(
                mean_squared_error(frame["observed"], frame["predicted"]) ** 0.5
            )
            _check(
                rows,
                f"sijiaying_{label}_{model}_r2_recomputed",
                abs(r2 - float(metrics.loc[model, "r2"])) <= 1e-12,
                r2,
                float(metrics.loc[model, "r2"]),
            )
            _check(
                rows,
                f"sijiaying_{label}_{model}_rmse_recomputed",
                abs(rmse - float(metrics.loc[model, "rmse"])) <= 1e-12,
                rmse,
                float(metrics.loc[model, "rmse"]),
            )
    _check(
        rows,
        "sijiaying_blind_metrics_hash",
        sha256_file(SIJIAYING / "sijiaying_blind_metrics_ONE_SHOT.csv")
        == sentinel["metrics_sha256"],
        sha256_file(SIJIAYING / "sijiaying_blind_metrics_ONE_SHOT.csv"),
        sentinel["metrics_sha256"],
    )
    _check(
        rows,
        "sijiaying_partition_hash",
        sha256_file(SIJIAYING / "sijiaying_frozen_hole_partition.csv")
        == freeze["partition_sha256"],
        sha256_file(SIJIAYING / "sijiaying_frozen_hole_partition.csv"),
        freeze["partition_sha256"],
    )

    simulation = pd.read_csv(
        SIMULATION / "failure_simulation_predictions_deidentified.csv"
    )
    replicate_metrics = pd.read_csv(
        SIMULATION / "failure_simulation_replicate_metrics.csv"
    ).set_index(["scenario", "replicate", "method"])
    _check(
        rows,
        "simulation_no_mine_coordinates_exported",
        not bool({"x", "y", "z", "collar_x", "collar_y"} & set(simulation.columns)),
        sorted({"x", "y", "z", "collar_x", "collar_y"} & set(simulation.columns)),
        [],
    )
    _check(
        rows,
        "simulation_all_numeric_outputs_finite",
        bool(
            np.isfinite(
                simulation[
                    ["observed", "predicted", "nearest_train_distance_m"]
                ].to_numpy(float)
            ).all()
        ),
        int(
            np.isfinite(
                simulation[
                    ["observed", "predicted", "nearest_train_distance_m"]
                ].to_numpy(float)
            ).sum()
        ),
        int(len(simulation) * 3),
    )
    scenario_replicates = simulation.groupby("scenario")["replicate"].nunique()
    _check(
        rows,
        "simulation_thirty_replicates_each_scenario",
        bool(scenario_replicates.eq(30).all()),
        scenario_replicates.to_dict(),
        "30 each",
    )
    leakage_scenarios = set(
        simulation.loc[
            simulation["method"].eq("M5_leakage_positive_control"), "scenario"
        ]
    )
    _check(
        rows,
        "leakage_control_only_in_S6",
        leakage_scenarios == {"S6_target_leakage"},
        sorted(leakage_scenarios),
        ["S6_target_leakage"],
    )
    maximum_metric_error = 0.0
    maximum_rmse_error = 0.0
    for keys, frame in simulation.groupby(
        ["scenario", "replicate", "method"], sort=False
    ):
        r2 = float(r2_score(frame["observed"], frame["predicted"]))
        rmse = float(
            mean_squared_error(frame["observed"], frame["predicted"]) ** 0.5
        )
        stored = replicate_metrics.loc[keys]
        maximum_metric_error = max(maximum_metric_error, abs(r2 - stored["r2"]))
        maximum_rmse_error = max(maximum_rmse_error, abs(rmse - stored["rmse"]))
    _check(
        rows,
        "simulation_replicate_r2_recomputed",
        maximum_metric_error <= 1e-12,
        maximum_metric_error,
        "<=1e-12",
    )
    _check(
        rows,
        "simulation_replicate_rmse_recomputed",
        maximum_rmse_error <= 1e-12,
        maximum_rmse_error,
        "<=1e-12",
    )

    validation = pd.DataFrame(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    validation.to_csv(OUTPUT / "independent_validation.csv", index=False)
    receipt = {
        "check_count": len(validation),
        "passed_count": int(validation["passed"].sum()),
        "failed_count": int((~validation["passed"]).sum()),
        "overall_passed": bool(validation["passed"].all()),
        "blind_one_shot_status": sentinel["status"],
        "blind_selected_model": sentinel["selected_model"],
        "simulation_predictions_sha256": sha256_file(
            SIMULATION / "failure_simulation_predictions_deidentified.csv"
        ),
    }
    (OUTPUT / "validation_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    if not receipt["overall_passed"]:
        print(validation.loc[~validation["passed"]].to_string(index=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
