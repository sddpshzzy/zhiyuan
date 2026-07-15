import numpy as np
import pandas as pd

from src.mmex_validation import CATEGORICAL_FEATURES, NUMERIC_FEATURES, convex_weight, make_outer_splits


def _toy_data():
    rows = []
    for hole in range(10):
        for interval in range(4):
            rows.append(
                {
                    "hole": f"h{hole:02d}",
                    "spatial_block": hole // 2,
                    "TFe": 20.0 + hole + interval,
                }
            )
    return pd.DataFrame(rows)


def test_grouped_splits_hold_out_complete_holes():
    data = _toy_data()
    for scheme in ["drillhole", "spatial_block"]:
        seen = []
        for train, test in make_outer_splits(data, scheme, 5, 42):
            assert set(data.iloc[train]["hole"]).isdisjoint(set(data.iloc[test]["hole"]))
            seen.extend(test.tolist())
        assert sorted(seen) == list(range(len(data)))


def test_convex_weight_is_bounded():
    y = np.array([1.0, 2.0, 3.0])
    xgb = np.array([1.1, 1.9, 2.8])
    ok = np.array([0.0, 4.0, 1.0])
    assert 0.0 <= convex_weight(y, xgb, ok) <= 1.0


def test_predictor_contract_excludes_grade_derived_fields():
    features = set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)
    forbidden = {"TFe", "prediction", "residual", "high_grade_threshold", "is_high_grade"}
    assert features.isdisjoint(forbidden)
