from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


FEATURE_CONTRACTS = {
    "G0_current_geology": "Current target-free fold-safe geology",
    "G1_plus_CAD_solid": "G0 plus reconstructed CAD-solid features",
    "G2_plus_TP_CVOCA_current": "G0 plus current-channel complex depth features",
    "G3_plus_solid_TP_CVOCA": "G0 plus solid and all complex depth features",
    "G4_solid_TP_CVOCA_top32": "G0 plus solid and outer-training-selected complex features",
}

FORBIDDEN_TOKENS = (
    "tfe",
    "feo",
    "sfe",
    "magnetic_susceptibility",
    "target",
    "assay_neighbour",
    "grade_domain",
)


def assert_disjoint_groups(
    train: pd.DataFrame,
    test: pd.DataFrame,
    group_column: str = "physical_group",
) -> None:
    overlap = set(train[group_column].dropna()) & set(test[group_column].dropna())
    if overlap:
        preview = sorted(map(str, overlap))[:5]
        raise ValueError(f"outer train/test group overlap: {preview}")


def assert_no_forbidden_features(
    feature_names: Iterable[str],
    forbidden_tokens: tuple[str, ...] = FORBIDDEN_TOKENS,
) -> None:
    violations = [
        name
        for name in feature_names
        if any(token in name.lower() for token in forbidden_tokens)
    ]
    if violations:
        raise ValueError(f"forbidden target-derived features: {violations}")


def assert_prediction_completeness(
    predictions: pd.DataFrame,
    interval_column: str = "interval_id",
    combination_columns: tuple[str, ...] = ("scheme", "model", "feature_contract"),
) -> None:
    expected = predictions[interval_column].nunique()
    counts = predictions.groupby(list(combination_columns))[interval_column].nunique()
    bad = counts[counts != expected]
    if not bad.empty:
        raise ValueError(f"incomplete outer predictions: {bad.to_dict()}")

