from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.spatial import cKDTree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.qidashan_strict import SPATIAL_FEATURES, sha256_file
from src.sijiaying_external import (
    LITHOLOGY_CATEGORICAL_FEATURE,
    LITHOLOGY_NUMERIC_FEATURES,
    load_authorised_intervals,
)


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config"
OUTPUT = ROOT / "results" / "sijiaying_blind_external"
SOURCE_CONTRACT = CONFIG / "sijiaying_source_selection_freeze_20260723.json"
SOURCE_PATH = Path(
    json.loads(SOURCE_CONTRACT.read_text(encoding="utf-8"))["selected_source"][
        "path"
    ]
)
HELPER_PATH = ROOT / "src" / "mdb_assay_subset.py"
PARTITION_PATH = OUTPUT / "sijiaying_frozen_hole_partition.csv"
DEVELOPMENT_HOLES = OUTPUT / "development_holes.json"
BLIND_HOLES = OUTPUT / "blind_holes_DO_NOT_OPEN_BEFORE_RELEASE.json"
FREEZE_MANIFEST = OUTPUT / "partition_and_code_freeze_manifest.json"
RELEASE_MANIFEST = OUTPUT / "blind_release_manifest.json"
BLIND_SENTINEL = OUTPUT / "BLIND_TARGET_OPENED_ONCE.json"
BASE_FEATURES = list(SPATIAL_FEATURES)
LITHOLOGY_FEATURES = BASE_FEATURES + LITHOLOGY_NUMERIC_FEATURES
PARAMS = {
    "depth": 8,
    "learning_rate": 0.05,
    "iterations": 350,
    "l2_leaf_reg": 8.0,
}
SEED = 20260723


def _verify_frozen_code() -> dict:
    manifest = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
    if sha256_file(SOURCE_PATH) != manifest["source_sha256"]:
        raise RuntimeError("selected source hash changed after freeze")
    if sha256_file(PARTITION_PATH) != manifest["partition_sha256"]:
        raise RuntimeError("frozen partition changed")
    for relative, expected in manifest["code_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"frozen code changed: {relative}")
    return manifest


def _matrix(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    include_lithology: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    train_x = train[features].copy()
    test_x = test[features].copy()
    medians = train_x.median(numeric_only=True)
    train_x = train_x.fillna(medians).fillna(0.0)
    test_x = test_x.fillna(medians).fillna(0.0)
    categorical: list[int] = []
    if include_lithology:
        train_x[LITHOLOGY_CATEGORICAL_FEATURE] = (
            train[LITHOLOGY_CATEGORICAL_FEATURE]
            .astype("string")
            .fillna("UNKNOWN")
            .astype(str)
        )
        test_x[LITHOLOGY_CATEGORICAL_FEATURE] = (
            test[LITHOLOGY_CATEGORICAL_FEATURE]
            .astype("string")
            .fillna("UNKNOWN")
            .astype(str)
        )
        categorical = [train_x.columns.get_loc(LITHOLOGY_CATEGORICAL_FEATURE)]
    return train_x, test_x, categorical


def _catboost_predict(
    train: pd.DataFrame, test: pd.DataFrame, include_lithology: bool, seed: int
) -> np.ndarray:
    features = LITHOLOGY_FEATURES if include_lithology else BASE_FEATURES
    train_x, test_x, categorical = _matrix(
        train, test, features, include_lithology
    )
    model = CatBoostRegressor(
        loss_function="RMSE",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
        **PARAMS,
    )
    model.fit(
        train_x,
        train["TFe"].to_numpy(float),
        cat_features=categorical or None,
    )
    return model.predict(test_x)


def _idw_predict(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    train_coords = train[["x", "y", "z"]].to_numpy(float)
    test_coords = test[["x", "y", "z"]].to_numpy(float)
    train_coords[:, 2] *= 0.25
    test_coords[:, 2] *= 0.25
    distances, neighbours = cKDTree(train_coords).query(
        test_coords, k=min(16, len(train))
    )
    if distances.ndim == 1:
        distances = distances[:, None]
        neighbours = neighbours[:, None]
    weights = 1.0 / np.maximum(distances, 1e-9) ** 2.0
    predictions = np.sum(
        weights * train["TFe"].to_numpy(float)[neighbours], axis=1
    ) / np.sum(weights, axis=1)
    return predictions, distances[:, 0]


def _prediction_frame(
    test: pd.DataFrame,
    predicted: np.ndarray,
    nearest_distance: np.ndarray,
    model: str,
    fold: int,
) -> pd.DataFrame:
    frame = test[
        [
            "interval_id",
            "hole",
            "physical_hole_group",
            "spatial_block",
            "TFe",
        ]
    ].copy()
    frame = frame.rename(columns={"TFe": "observed"})
    frame["predicted"] = predicted
    frame["nearest_train_distance_m"] = nearest_distance
    frame["model"] = model
    frame["fold"] = int(fold)
    return frame


def _metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, frame in predictions.groupby("model", sort=False):
        y = frame["observed"].to_numpy(float)
        p = frame["predicted"].to_numpy(float)
        far_cut = float(frame["nearest_train_distance_m"].quantile(0.75))
        far = frame["nearest_train_distance_m"].ge(far_cut)
        rows.append(
            {
                "model": model,
                "n": len(frame),
                "r2": float(r2_score(y, p)),
                "rmse": float(mean_squared_error(y, p) ** 0.5),
                "mae": float(mean_absolute_error(y, p)),
                "bias": float(np.mean(p - y)),
                "far_quartile_r2": (
                    float(
                        r2_score(
                            frame.loc[far, "observed"],
                            frame.loc[far, "predicted"],
                        )
                    )
                    if far.sum() >= 2
                    else np.nan
                ),
                "far_quartile_n": int(far.sum()),
            }
        )
    return pd.DataFrame(rows)


def run_development() -> None:
    freeze = _verify_frozen_code()
    data, audit = load_authorised_intervals(
        SOURCE_PATH,
        HELPER_PATH,
        DEVELOPMENT_HOLES,
        PARTITION_PATH,
    )
    if data["partition_role"].ne("DEVELOPMENT").any():
        raise RuntimeError("blind row crossed the development firewall")
    prediction_frames: list[pd.DataFrame] = []
    development_blocks = sorted(data["spatial_block"].unique())
    for fold, block in enumerate(development_blocks):
        train = data.loc[data["spatial_block"].ne(block)].reset_index(drop=True)
        test = data.loc[data["spatial_block"].eq(block)].reset_index(drop=True)
        idw, nearest = _idw_predict(train, test)
        prediction_frames.append(
            _prediction_frame(test, np.repeat(train["TFe"].mean(), len(test)), nearest, "mean", fold)
        )
        prediction_frames.append(
            _prediction_frame(test, idw, nearest, "IDW_fixed", fold)
        )
        prediction_frames.append(
            _prediction_frame(
                test,
                _catboost_predict(train, test, False, SEED + fold),
                nearest,
                "E0_geometry_CatBoost",
                fold,
            )
        )
        prediction_frames.append(
            _prediction_frame(
                test,
                _catboost_predict(train, test, True, SEED + 100 + fold),
                nearest,
                "E2_geometry_lithology_CatBoost",
                fold,
            )
        )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = _metrics(predictions)
    base_r2 = float(
        metrics.loc[metrics["model"].eq("E0_geometry_CatBoost"), "r2"].iloc[0]
    )
    lith_r2 = float(
        metrics.loc[
            metrics["model"].eq("E2_geometry_lithology_CatBoost"), "r2"
        ].iloc[0]
    )
    delta = lith_r2 - base_r2
    gate_passed = bool(delta >= 0.10)
    selection = (
        "E2_geometry_lithology_CatBoost"
        if gate_passed
        else "E0_geometry_CatBoost"
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT / "sijiaying_development_predictions.csv", index=False)
    metrics.to_csv(OUTPUT / "sijiaying_development_metrics.csv", index=False)
    audit.to_csv(OUTPUT / "sijiaying_development_data_audit.csv", index=False)
    release = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "DEVELOPMENT_COMPLETE_BLIND_RELEASE_FROZEN",
        "source_sha256": freeze["source_sha256"],
        "partition_sha256": freeze["partition_sha256"],
        "code_sha256": freeze["code_sha256"],
        "development_blocks": [int(value) for value in development_blocks],
        "blind_block": int(freeze["blind_block"]),
        "baseline_r2": base_r2,
        "lithology_candidate_r2": lith_r2,
        "lithology_delta_r2": delta,
        "lithology_gate_threshold": 0.10,
        "lithology_gate_passed": gate_passed,
        "blind_selected_contract": selection,
        "entity_contract_available": False,
        "entity_reason": (
            "Only a grade-threshold staged/block model was found; it is "
            "ineligible as an independent geological predictor."
        ),
        "hyperparameters": PARAMS,
        "random_seed": SEED,
    }
    RELEASE_MANIFEST.write_text(
        json.dumps(release, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(release, ensure_ascii=False, indent=2))


def _group_bootstrap_delta(
    predictions: pd.DataFrame,
    left_model: str,
    right_model: str,
    n_bootstrap: int = 2000,
) -> dict[str, float]:
    pivot = predictions.pivot_table(
        index=[
            "interval_id",
            "physical_hole_group",
            "observed",
        ],
        columns="model",
        values="predicted",
        aggfunc="first",
    ).reset_index()
    groups = pivot["physical_hole_group"].unique()
    rng = np.random.default_rng(SEED + 9000)
    deltas: list[float] = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        pieces = [
            pivot.loc[pivot["physical_hole_group"].eq(group)] for group in sampled
        ]
        boot = pd.concat(pieces, ignore_index=True)
        if boot["observed"].nunique() < 2:
            continue
        left = r2_score(boot["observed"], boot[left_model])
        right = r2_score(boot["observed"], boot[right_model])
        deltas.append(float(left - right))
    values = np.asarray(deltas, dtype=float)
    return {
        "point_delta_r2": float(
            r2_score(pivot["observed"], pivot[left_model])
            - r2_score(pivot["observed"], pivot[right_model])
        ),
        "ci_lower_2_5": float(np.quantile(values, 0.025)),
        "ci_upper_97_5": float(np.quantile(values, 0.975)),
        "bootstrap_replicates": int(len(values)),
    }


def run_blind() -> None:
    freeze = _verify_frozen_code()
    if BLIND_SENTINEL.exists():
        raise RuntimeError("blind target has already been opened; rerun prohibited")
    release = json.loads(RELEASE_MANIFEST.read_text(encoding="utf-8"))
    if release["code_sha256"] != freeze["code_sha256"]:
        raise RuntimeError("release code does not match the frozen code")
    development, _ = load_authorised_intervals(
        SOURCE_PATH,
        HELPER_PATH,
        DEVELOPMENT_HOLES,
        PARTITION_PATH,
    )
    blind, blind_audit = load_authorised_intervals(
        SOURCE_PATH,
        HELPER_PATH,
        BLIND_HOLES,
        PARTITION_PATH,
    )
    if blind["spatial_block"].nunique() != 1 or int(
        blind["spatial_block"].iloc[0]
    ) != int(freeze["blind_block"]):
        raise RuntimeError("blind firewall emitted the wrong spatial block")
    idw, nearest = _idw_predict(development, blind)
    frames = [
        _prediction_frame(
            blind,
            np.repeat(development["TFe"].mean(), len(blind)),
            nearest,
            "mean",
            int(freeze["blind_block"]),
        ),
        _prediction_frame(
            blind, idw, nearest, "IDW_fixed", int(freeze["blind_block"])
        ),
        _prediction_frame(
            blind,
            _catboost_predict(development, blind, False, SEED + 500),
            nearest,
            "E0_geometry_CatBoost",
            int(freeze["blind_block"]),
        ),
    ]
    selected = str(release["blind_selected_contract"])
    if selected == "E2_geometry_lithology_CatBoost":
        frames.append(
            _prediction_frame(
                blind,
                _catboost_predict(development, blind, True, SEED + 600),
                nearest,
                selected,
                int(freeze["blind_block"]),
            )
        )
    predictions = pd.concat(frames, ignore_index=True)
    metrics = _metrics(predictions)
    comparisons = []
    for reference in ["mean", "IDW_fixed"]:
        comparisons.append(
            {
                "selected_model": selected,
                "reference_model": reference,
                **_group_bootstrap_delta(predictions, selected, reference),
            }
        )
    comparison_frame = pd.DataFrame(comparisons)
    predictions.to_csv(OUTPUT / "sijiaying_blind_predictions_ONE_SHOT.csv", index=False)
    metrics.to_csv(OUTPUT / "sijiaying_blind_metrics_ONE_SHOT.csv", index=False)
    comparison_frame.to_csv(
        OUTPUT / "sijiaying_blind_group_bootstrap_ONE_SHOT.csv", index=False
    )
    blind_audit.to_csv(OUTPUT / "sijiaying_blind_data_audit_ONE_SHOT.csv", index=False)
    sentinel = {
        "opened_at": datetime.now().astimezone().isoformat(),
        "status": "BLIND_TARGET_OPENED_AND_REPORTED_ONCE",
        "source_sha256": freeze["source_sha256"],
        "partition_sha256": freeze["partition_sha256"],
        "selected_model": selected,
        "metrics_sha256": sha256_file(
            OUTPUT / "sijiaying_blind_metrics_ONE_SHOT.csv"
        ),
        "predictions_sha256": sha256_file(
            OUTPUT / "sijiaying_blind_predictions_ONE_SHOT.csv"
        ),
        "no_refit_after_opening": True,
    }
    BLIND_SENTINEL.write_text(
        json.dumps(sentinel, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(sentinel, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["development", "blind"])
    args = parser.parse_args()
    if args.mode == "development":
        run_development()
    else:
        run_blind()


if __name__ == "__main__":
    main()
