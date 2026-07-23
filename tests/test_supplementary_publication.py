from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "supplementary"


def test_one_shot_blind_metrics_match_frozen_report():
    metrics = pd.read_csv(RESULTS / "sijiaying_external_metrics.csv")
    row = metrics.loc[
        metrics["stage"].eq("Blind")
        & metrics["model"].eq("Geometry + lithology")
    ].iloc[0]
    assert round(float(row["r2"]), 4) == 0.1777
    assert round(float(row["rmse"]), 3) == 5.830
    assert round(float(row["far_quartile_r2"]), 4) == 0.4087


def test_failure_simulation_keeps_all_scenarios_and_replicates():
    summary = pd.read_csv(RESULTS / "failure_simulation_summary.csv")
    assert summary["scenario"].nunique() == 7
    assert set(summary["replicates"]) == {30}
    leakage = summary.loc[
        summary["method"].eq("M5_leakage_positive_control")
    ]
    assert set(leakage["scenario"]) == {"S6_target_leakage"}
    assert round(float(leakage.iloc[0]["r2_mean"]), 4) == 0.5288


def test_public_supplement_contains_no_coordinate_columns():
    prohibited = {
        "x",
        "y",
        "z",
        "collar_x",
        "collar_y",
        "collar_z",
        "easting",
        "northing",
    }
    for path in RESULTS.glob("*.csv"):
        columns = {str(column).strip().lower() for column in pd.read_csv(path, nrows=1)}
        assert not columns.intersection(prohibited), path.name


def test_independent_recomputation_checks_all_passed():
    checks = pd.read_csv(RESULTS / "independent_validation.csv")
    assert len(checks) == 27
    assert checks["passed"].astype(str).str.lower().eq("true").all()
