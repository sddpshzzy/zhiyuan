#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quick-test (smoke test) for the XGBoost–Kriging fusion workflow.

Purpose:
- Run an end-to-end pipeline on small synthetic spatial data
- Avoid any proprietary data
- Finish quickly
- Write simple outputs to results/
- Print PASS/FAIL

Outputs:
- results/quick_test_metrics.json
- results/quick_test_scatter_ml.png
- results/quick_test_scatter_fusion.png
"""

from __future__ import annotations

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import xgboost as xgb
from pykrige.ok import OrdinaryKriging


def make_synthetic_data(n: int = 250, seed: int = 42):
    """
    Create a small 2D spatial dataset with:
    - nonlinear attribute effects (learned by XGBoost)
    - spatially structured residual component (corrected by kriging)
    """
    rng = np.random.default_rng(seed)

    # Coordinates
    x = rng.uniform(0.0, 1000.0, size=n)
    y = rng.uniform(0.0, 1000.0, size=n)

    # Auxiliary attributes (nonlinear controls)
    a1 = np.sin(x / 150.0) + 0.10 * rng.normal(size=n)
    a2 = np.cos(y / 180.0) + 0.10 * rng.normal(size=n)
    a3 = (x * y) / (1000.0 * 1000.0) + 0.05 * rng.normal(size=n)

    # Nonlinear mean term
    mean_term = 2.0 * a1 + 1.5 * (a2 ** 2) - 1.0 * np.tanh(2.0 * a3)

    # Spatially structured residual (fast radial mixture)
    centres = np.array([[200.0, 300.0], [700.0, 200.0], [600.0, 800.0]])
    weights = np.array([0.8, -0.6, 0.5])
    scales = np.array([220.0, 180.0, 260.0])

    res = np.zeros(n, dtype=float)
    for (cx, cy), w, s in zip(centres, weights, scales):
        d2 = (x - cx) ** 2 + (y - cy) ** 2
        res += w * np.exp(-d2 / (2.0 * s ** 2))

    noise = 0.10 * rng.normal(size=n)

    z = mean_term + res + noise
    X = np.column_stack([x, y, a1, a2, a3])
    return X, z


def main():
    os.makedirs("results", exist_ok=True)

    # 1) Data
    X, z = make_synthetic_data(n=250, seed=42)
    xcoord, ycoord = X[:, 0], X[:, 1]

    X_tr, X_te, z_tr, z_te, x_tr, x_te, y_tr, y_te = train_test_split(
        X, z, xcoord, ycoord, test_size=0.25, random_state=42
    )

    # 2) XGBoost: attribute learning
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42
    )
    model.fit(X_tr, z_tr)

    pred_ml_tr = model.predict(X_tr)
    pred_ml_te = model.predict(X_te)

    # 3) Residuals and kriging correction
    res_tr = z_tr - pred_ml_tr

    ok = OrdinaryKriging(
        x_tr, y_tr, res_tr,
        variogram_model="spherical",
        verbose=False,
        enable_plotting=False
    )
    res_corr_te, _ = ok.execute("points", x_te, y_te)

    pred_fusion_te = pred_ml_te + np.asarray(res_corr_te)

    # 4) Metrics
    rmse_ml = float(np.sqrt(mean_squared_error(z_te, pred_ml_te)))
    r2_ml = float(r2_score(z_te, pred_ml_te))

    rmse_f = float(np.sqrt(mean_squared_error(z_te, pred_fusion_te)))
    r2_f = float(r2_score(z_te, pred_fusion_te))

    summary = {
        "rmse_ml": rmse_ml,
        "r2_ml": r2_ml,
        "rmse_fusion": rmse_f,
        "r2_fusion": r2_f,
        "n_train": int(len(z_tr)),
        "n_test": int(len(z_te)),
        "seed": 42
    }

    with open("results/quick_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 5) Plots (small, fast)
    plt.figure()
    plt.scatter(z_te, pred_ml_te, s=18, alpha=0.8)
    plt.xlabel("Observed")
    plt.ylabel("Predicted (XGBoost)")
    plt.tight_layout()
    plt.savefig("results/quick_test_scatter_ml.png", dpi=200)
    plt.close()

    plt.figure()
    plt.scatter(z_te, pred_fusion_te, s=18, alpha=0.8)
    plt.xlabel("Observed")
    plt.ylabel("Predicted (Fusion)")
    plt.tight_layout()
    plt.savefig("results/quick_test_scatter_fusion.png", dpi=200)
    plt.close()

    # 6) PASS/FAIL (simple criterion)
    passed = (rmse_f <= rmse_ml + 1e-9) and (r2_f >= r2_ml - 1e-9)

    print("Quick-test summary:", summary)
    print("PASS" if passed else "FAIL")

    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
