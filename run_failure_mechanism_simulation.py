from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.spatial import cKDTree
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from src.local_residual_kriging import fit_variogram, local_ordinary_kriging
from src.qidashan_strict import SPATIAL_FEATURES, load_qidashan_intervals, sha256_file


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "failure_mechanism_simulation_contract_20260723.json"
IMPLEMENTATION_PATH = (
    ROOT / "config" / "failure_simulation_implementation_freeze_20260723.json"
)
OUTPUT = ROOT / "results" / "failure_mechanism_simulation"
CATBOOST_PARAMS = {
    "depth": 5,
    "learning_rate": 0.08,
    "iterations": 60,
    "l2_leaf_reg": 8.0,
}
SEED = 20260723
SCENARIOS = [
    "S0_stationary",
    "S1_anisotropy_misspecified",
    "S2_domain_mean_shift",
    "S3_fault_offset",
    "S4_support_mismatch",
    "S5_sparse_far_extrapolation",
    "S6_target_leakage",
]


def _representative_geometry(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for _, group in data.groupby("physical_hole_group", sort=True):
        middle = float(group["mid_depth"].median())
        index = (group["mid_depth"] - middle).abs().idxmin()
        rows.append(group.loc[index])
    geometry = pd.DataFrame(rows).reset_index(drop=True)
    geometry["simulation_point_id"] = np.arange(len(geometry))
    return geometry


def _rff_process(
    rng: np.random.Generator,
    length_scales: tuple[float, float, float],
    rotation_deg: float = 0.0,
    components: int = 128,
):
    angle = math.radians(rotation_deg)
    rotation = np.asarray(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    frequency = rng.normal(size=(3, components))
    frequency = rotation @ (
        frequency / np.asarray(length_scales, dtype=float)[:, None]
    )
    phase = rng.uniform(0.0, 2.0 * np.pi, size=components)
    weight = rng.normal(size=components)

    def evaluate(coordinates: np.ndarray) -> np.ndarray:
        values = (
            np.sqrt(2.0 / components)
            * np.cos(coordinates @ frequency + phase)
            @ weight
        )
        return np.asarray(values, dtype=float)

    return evaluate


def _standardise(values: np.ndarray, scale: float = 7.0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - values.mean()) / max(values.std(ddof=1), 1e-9) * scale


def _synthetic_target(
    geometry: pd.DataFrame, scenario: str, replicate: int
) -> np.ndarray:
    rng = np.random.default_rng(SEED + replicate * 100 + SCENARIOS.index(scenario))
    coordinates = geometry[["x", "y", "z"]].to_numpy(float)
    if scenario == "S1_anisotropy_misspecified":
        process = _rff_process(rng, (700.0, 45.0, 900.0), rotation_deg=37.0)
        latent = process(coordinates)
    elif scenario == "S4_support_mismatch":
        process = _rff_process(rng, (180.0, 180.0, 12.0), rotation_deg=10.0)
        offsets = np.linspace(-0.5, 0.5, 11)
        azimuth_sin = geometry["azimuth_sin"].to_numpy(float)
        azimuth_cos = geometry["azimuth_cos"].to_numpy(float)
        dip_sin = geometry["dip_sin"].to_numpy(float)
        dip_cos = geometry["dip_cos"].to_numpy(float)
        direction = np.column_stack(
            [
                dip_cos * azimuth_sin,
                dip_cos * azimuth_cos,
                dip_sin,
            ]
        )
        support = np.maximum(geometry["length"].to_numpy(float), 0.5)
        samples = []
        for fraction in offsets:
            shifted = coordinates + direction * (fraction * support)[:, None]
            samples.append(process(shifted))
        latent = np.mean(samples, axis=0)
    elif scenario == "S5_sparse_far_extrapolation":
        process = _rff_process(rng, (55.0, 55.0, 120.0), rotation_deg=0.0)
        latent = process(coordinates)
    else:
        process = _rff_process(rng, (300.0, 300.0, 1200.0), rotation_deg=0.0)
        latent = process(coordinates)
    target = 32.0 + _standardise(latent, 7.0)
    principal = geometry.groupby("spatial_block")["x"].transform("mean")
    if scenario == "S2_domain_mean_shift":
        threshold = float(np.median(geometry["x"]))
        target += np.where(geometry["x"].to_numpy(float) >= threshold, 7.0, -7.0)
    elif scenario == "S3_fault_offset":
        x0 = geometry["x"].to_numpy(float) - float(geometry["x"].median())
        y0 = geometry["y"].to_numpy(float) - float(geometry["y"].median())
        target += np.where(x0 + 0.45 * y0 >= 0.0, 8.0, -5.0)
    elif scenario == "S5_sparse_far_extrapolation":
        target += 0.002 * (
            principal.to_numpy(float) - float(principal.mean())
        )
    target += rng.normal(0.0, 1.2, size=len(target))
    return target


def _scaled_coordinates(frame: pd.DataFrame) -> np.ndarray:
    values = frame[["x", "y", "z"]].to_numpy(float).copy()
    values[:, 2] *= 0.25
    return values


def _idw(
    train: pd.DataFrame, test: pd.DataFrame, k: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    distance, neighbours = cKDTree(_scaled_coordinates(train)).query(
        _scaled_coordinates(test), k=min(k, len(train))
    )
    if distance.ndim == 1:
        distance = distance[:, None]
        neighbours = neighbours[:, None]
    weights = 1.0 / np.maximum(distance, 1e-9) ** 2.0
    prediction = np.sum(
        weights * train["TFe"].to_numpy(float)[neighbours], axis=1
    ) / weights.sum(axis=1)
    return prediction, distance[:, 0]


def _fit_catboost(
    train: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
    extra_feature: str | None = None,
) -> np.ndarray:
    features = list(SPATIAL_FEATURES)
    if extra_feature is not None:
        features.append(extra_feature)
    model = CatBoostRegressor(
        loss_function="RMSE",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=1,
        **CATBOOST_PARAMS,
    )
    model.fit(train[features], train["TFe"])
    return model.predict(test[features])


def _crossfitted_residuals(
    train: pd.DataFrame, replicate: int, outer_fold: int
) -> np.ndarray:
    output = np.full(len(train), np.nan)
    splitter = GroupKFold(n_splits=3)
    for inner_fold, (fit_index, valid_index) in enumerate(
        splitter.split(train, groups=train["spatial_block"])
    ):
        output[valid_index] = _fit_catboost(
            train.iloc[fit_index],
            train.iloc[valid_index],
            SEED + replicate * 1000 + outer_fold * 20 + inner_fold,
        )
    if np.isnan(output).any():
        raise RuntimeError("incomplete residual cross-fitting")
    return train["TFe"].to_numpy(float) - output


def _kriging_prediction(
    train: pd.DataFrame,
    values: np.ndarray,
    test: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, int]:
    try:
        variogram, _ = fit_variogram(
            train,
            values,
            z_scale=0.25,
            model="exponential",
            maximum_pairs=5000,
            bins=10,
            seed=seed,
        )
        prediction, audit = local_ordinary_kriging(
            train, values, test, variogram, neighbours=24
        )
        return prediction, int(audit["idw_fallback"].sum())
    except Exception:
        return np.repeat(float(np.mean(values)), len(test)), len(test)


def _leakage_feature(frame: pd.DataFrame) -> np.ndarray:
    coordinates = _scaled_coordinates(frame)
    distance, neighbours = cKDTree(coordinates).query(
        coordinates, k=min(9, len(frame))
    )
    if distance.ndim == 1:
        neighbours = neighbours[:, None]
    neighbours = neighbours[:, 1:] if neighbours.shape[1] > 1 else neighbours
    return frame["TFe"].to_numpy(float)[neighbours].mean(axis=1)


def _append_predictions(
    records: list[pd.DataFrame],
    test: pd.DataFrame,
    prediction: np.ndarray,
    distance: np.ndarray,
    scenario: str,
    replicate: int,
    fold: int,
    method: str,
    kriging_fallback_count: int = 0,
) -> None:
    frame = test[
        [
            "simulation_point_id",
            "physical_hole_group",
            "spatial_block",
            "TFe",
        ]
    ].copy()
    frame = frame.rename(columns={"TFe": "observed"})
    frame["predicted"] = prediction
    frame["nearest_train_distance_m"] = distance
    frame["scenario"] = scenario
    frame["replicate"] = replicate
    frame["fold"] = fold
    frame["method"] = method
    frame["kriging_fallback_count_fold"] = kriging_fallback_count
    records.append(frame)


def run_replicate(
    geometry: pd.DataFrame, scenario: str, replicate: int
) -> pd.DataFrame:
    data = geometry.copy()
    data["TFe"] = _synthetic_target(data, scenario, replicate)
    if scenario == "S6_target_leakage":
        data["leakage_neighbour_target_mean"] = _leakage_feature(data)
    frames: list[pd.DataFrame] = []
    for fold, block in enumerate(sorted(data["spatial_block"].unique())):
        train = data.loc[data["spatial_block"].ne(block)].reset_index(drop=True)
        test = data.loc[data["spatial_block"].eq(block)].reset_index(drop=True)
        idw_prediction, nearest_distance = _idw(train, test)
        _append_predictions(
            frames,
            test,
            np.repeat(train["TFe"].mean(), len(test)),
            nearest_distance,
            scenario,
            replicate,
            fold,
            "M0_mean",
        )
        _append_predictions(
            frames,
            test,
            idw_prediction,
            nearest_distance,
            scenario,
            replicate,
            fold,
            "M1_idw",
        )
        ordinary, ordinary_fallback = _kriging_prediction(
            train,
            train["TFe"].to_numpy(float),
            test,
            SEED + replicate * 100 + fold,
        )
        _append_predictions(
            frames,
            test,
            ordinary,
            nearest_distance,
            scenario,
            replicate,
            fold,
            "M2_ordinary_kriging",
            ordinary_fallback,
        )
        trend = _fit_catboost(
            train, test, SEED + replicate * 1000 + fold
        )
        _append_predictions(
            frames,
            test,
            trend,
            nearest_distance,
            scenario,
            replicate,
            fold,
            "M3_catboost_coordinates",
        )
        residuals = _crossfitted_residuals(train, replicate, fold)
        residual_prediction, residual_fallback = _kriging_prediction(
            train,
            residuals,
            test,
            SEED + replicate * 200 + fold + 50,
        )
        _append_predictions(
            frames,
            test,
            trend + residual_prediction,
            nearest_distance,
            scenario,
            replicate,
            fold,
            "M4_crossfitted_residual_kriging",
            residual_fallback,
        )
        if scenario == "S6_target_leakage":
            leakage = _fit_catboost(
                train,
                test,
                SEED + replicate * 1000 + fold + 500,
                extra_feature="leakage_neighbour_target_mean",
            )
            _append_predictions(
                frames,
                test,
                leakage,
                nearest_distance,
                scenario,
                replicate,
                fold,
                "M5_leakage_positive_control",
            )
    return pd.concat(frames, ignore_index=True)


def summarise(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    replicate_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for (scenario, replicate, method), frame in predictions.groupby(
        ["scenario", "replicate", "method"], sort=False
    ):
        far_cut = float(frame["nearest_train_distance_m"].quantile(0.75))
        far = frame["nearest_train_distance_m"].ge(far_cut)
        fold_r2 = []
        for fold, fold_frame in frame.groupby("fold"):
            value = float(
                r2_score(fold_frame["observed"], fold_frame["predicted"])
            )
            fold_r2.append(value)
            fold_rows.append(
                {
                    "scenario": scenario,
                    "replicate": replicate,
                    "method": method,
                    "fold": fold,
                    "r2": value,
                    "n": len(fold_frame),
                }
            )
        replicate_rows.append(
            {
                "scenario": scenario,
                "replicate": replicate,
                "method": method,
                "n": len(frame),
                "r2": float(r2_score(frame["observed"], frame["predicted"])),
                "rmse": float(
                    mean_squared_error(
                        frame["observed"], frame["predicted"]
                    )
                    ** 0.5
                ),
                "bias": float(
                    np.mean(frame["predicted"] - frame["observed"])
                ),
                "far_quartile_r2": float(
                    r2_score(
                        frame.loc[far, "observed"],
                        frame.loc[far, "predicted"],
                    )
                ),
                "fold_r2_std": float(np.std(fold_r2, ddof=1)),
                "kriging_fallback_count": int(
                    frame.groupby("fold")["kriging_fallback_count_fold"]
                    .first()
                    .sum()
                ),
            }
        )
    replicate_metrics = pd.DataFrame(replicate_rows)
    summary_rows: list[dict[str, object]] = []
    for (scenario, method), frame in replicate_metrics.groupby(
        ["scenario", "method"], sort=False
    ):
        summary_rows.append(
            {
                "scenario": scenario,
                "method": method,
                "replicates": len(frame),
                "r2_mean": float(frame["r2"].mean()),
                "r2_q025": float(frame["r2"].quantile(0.025)),
                "r2_q975": float(frame["r2"].quantile(0.975)),
                "probability_r2_positive": float(frame["r2"].gt(0).mean()),
                "rmse_mean": float(frame["rmse"].mean()),
                "abs_bias_mean": float(frame["bias"].abs().mean()),
                "far_quartile_r2_mean": float(
                    frame["far_quartile_r2"].mean()
                ),
                "fold_r2_std_mean": float(frame["fold_r2_std"].mean()),
                "kriging_fallback_count_total": int(
                    frame["kriging_fallback_count"].sum()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    comparison = replicate_metrics.pivot_table(
        index=["scenario", "replicate"],
        columns="method",
        values="r2",
    ).reset_index()
    delta_rows: list[dict[str, object]] = []
    for scenario, frame in comparison.groupby("scenario"):
        if (
            "M4_crossfitted_residual_kriging" in frame
            and "M3_catboost_coordinates" in frame
        ):
            delta = (
                frame["M4_crossfitted_residual_kriging"]
                - frame["M3_catboost_coordinates"]
            )
            delta_rows.append(
                {
                    "scenario": scenario,
                    "comparison": "M4_minus_M3",
                    "delta_r2_mean": float(delta.mean()),
                    "delta_r2_q025": float(delta.quantile(0.025)),
                    "delta_r2_q975": float(delta.quantile(0.975)),
                    "probability_delta_r2_positive": float(delta.gt(0).mean()),
                }
            )
        if (
            "M2_ordinary_kriging" in frame
            and "M1_idw" in frame
        ):
            delta = frame["M2_ordinary_kriging"] - frame["M1_idw"]
            delta_rows.append(
                {
                    "scenario": scenario,
                    "comparison": "M2_minus_M1",
                    "delta_r2_mean": float(delta.mean()),
                    "delta_r2_q025": float(delta.quantile(0.025)),
                    "delta_r2_q975": float(delta.quantile(0.975)),
                    "probability_delta_r2_positive": float(delta.gt(0).mean()),
                }
            )
    return replicate_metrics, pd.DataFrame(fold_rows), pd.DataFrame(delta_rows), summary


def main() -> None:
    contract = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    implementation = json.loads(IMPLEMENTATION_PATH.read_text(encoding="utf-8"))
    if sha256_file(Path(__file__)) != implementation["code_sha256"]:
        raise RuntimeError("simulation code changed after implementation freeze")
    source = Path(json.loads((ROOT / "config" / "experiment.json").read_text())["source_mdb"])
    data, _ = load_qidashan_intervals(source, n_spatial_blocks=5)
    geometry = _representative_geometry(data)
    if len(geometry) != geometry["physical_hole_group"].nunique():
        raise RuntimeError("simulation geometry is not one row per physical group")
    all_predictions: list[pd.DataFrame] = []
    replicates = int(contract["geometry"]["replicates_per_scenario"])
    for scenario in SCENARIOS:
        for replicate in range(replicates):
            all_predictions.append(run_replicate(geometry, scenario, replicate))
            print(
                f"COMPLETE scenario={scenario} replicate={replicate + 1}/{replicates}",
                flush=True,
            )
    predictions = pd.concat(all_predictions, ignore_index=True)
    replicate_metrics, fold_metrics, deltas, summary = summarise(predictions)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        OUTPUT / "failure_simulation_predictions_deidentified.csv", index=False
    )
    replicate_metrics.to_csv(
        OUTPUT / "failure_simulation_replicate_metrics.csv", index=False
    )
    fold_metrics.to_csv(
        OUTPUT / "failure_simulation_fold_metrics.csv", index=False
    )
    deltas.to_csv(
        OUTPUT / "failure_simulation_delta_metrics.csv", index=False
    )
    summary.to_csv(
        OUTPUT / "failure_simulation_summary.csv", index=False
    )
    geometry_audit = {
        "source_sha256": sha256_file(source),
        "source_coordinates_exported": False,
        "simulation_point_count": len(geometry),
        "physical_hole_group_count": int(
            geometry["physical_hole_group"].nunique()
        ),
        "spatial_block_counts": {
            str(key): int(value)
            for key, value in geometry["spatial_block"].value_counts().sort_index().items()
        },
        "representative_rule": implementation["representative_rule"],
        "replicates_per_scenario": replicates,
        "scenario_count": len(SCENARIOS),
        "code_sha256": implementation["code_sha256"],
    }
    (OUTPUT / "failure_simulation_run_manifest.json").write_text(
        json.dumps(geometry_audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
