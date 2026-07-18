from __future__ import annotations

import numpy as np
import pandas as pd

from article2xg import fit_cvoca_transform, transform_cvoca


def make_synthetic_holes() -> pd.DataFrame:
    rows = []
    for hole_index, hole in enumerate(("A", "B", "C")):
        depth = np.asarray([0.0, 4.5, 10.0, 17.5, 26.0, 36.0, 49.0, 64.0])
        z = 120.0 - depth + hole_index * 2.0
        ore_probability = 0.5 + 0.35 * np.sin((depth + 5 * hole_index) / 18.0)
        rows.extend(
            {
                "hole": hole,
                "mid_depth": float(d),
                "z": float(elevation),
                "solid_ore_probability": float(probability),
                "TFe": float(25.0 + 20.0 * probability + hole_index),
            }
            for d, elevation, probability in zip(depth, z, ore_probability)
        )
    return pd.DataFrame(rows)


def main() -> None:
    frame = make_synthetic_holes()
    channels = ["z", "solid_ore_probability"]
    transform = fit_cvoca_transform(frame[frame.hole != "C"], channels)
    first = transform_cvoca(frame, transform)
    altered = frame.copy()
    altered["TFe"] = altered["TFe"] * -100.0
    second = transform_cvoca(altered, transform)
    np.testing.assert_allclose(first.to_numpy(), second.to_numpy())
    print(f"rows={len(first)} features={len(first.columns)} finite={np.isfinite(first.to_numpy()).all()}")
    print("target-change invariance: passed")


if __name__ == "__main__":
    main()

