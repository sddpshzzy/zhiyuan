from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from article2xg import (
    assert_disjoint_groups,
    assert_no_forbidden_features,
    fit_cvoca_transform,
    transform_cvoca,
)


def test_complex_features_are_finite_and_target_independent() -> None:
    frame = pd.DataFrame(
        {
            "hole": ["A"] * 6 + ["B"] * 6,
            "mid_depth": [0.0, 3.0, 8.0, 15.0, 24.0, 36.0] * 2,
            "z": np.linspace(100.0, 10.0, 12),
            "solid_probability": np.linspace(0.1, 0.9, 12),
            "TFe": np.linspace(15.0, 65.0, 12),
        }
    )
    fitted = fit_cvoca_transform(frame.iloc[:8], ["z", "solid_probability"])
    first = transform_cvoca(frame, fitted)
    changed = frame.copy()
    changed["TFe"] += 1000.0
    second = transform_cvoca(changed, fitted)
    assert first.shape == (12, 21)
    assert np.isfinite(first.to_numpy()).all()
    np.testing.assert_allclose(first.to_numpy(), second.to_numpy())


def test_validation_guards_reject_overlap_and_target_names() -> None:
    train = pd.DataFrame({"physical_group": ["A", "B"]})
    test = pd.DataFrame({"physical_group": ["B", "C"]})
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_groups(train, test)
    with pytest.raises(ValueError, match="forbidden"):
        assert_no_forbidden_features(["x", "neighbour_TFe_mean"])
