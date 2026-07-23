from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.spatial import cKDTree
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


SPATIAL_FEATURES = [
    "x", "y", "z", "collar_x", "collar_y", "collar_z", "max_depth",
    "from", "to", "length", "mid_depth", "depth_ratio",
    "azimuth_sin", "azimuth_cos", "dip_sin", "dip_cos",
]
COASSAY_FEATURES = [
    "FeO", "SFe", "FeO_missing", "SFe_missing"
]
FORBIDDEN_OUTCOME_FEATURES = {
    "TFe", "prediction", "residual", "high_grade", "high_grade_threshold",
    "spatial_block", "hole", "interval_id", "magnetic_rate", "magnetic_rate_missing",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_mdb_table(mdb_path: Path, table: str) -> pd.DataFrame:
    completed = subprocess.run(
        ["mdb-export", str(mdb_path), table], check=True, capture_output=True
    )
    text = completed.stdout.decode("utf-8-sig")
    return pd.read_csv(io.StringIO(text), quoting=csv.QUOTE_MINIMAL)


def _normalise_hole(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def _collar_connected_components(coordinates: np.ndarray, radius_m: float = 5.0) -> np.ndarray:
    """Conservatively group collocated collars to prevent cross-name leakage."""
    parent = np.arange(len(coordinates))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    tree = cKDTree(coordinates)
    for left, right in sorted(tree.query_pairs(radius_m)):
        union(int(left), int(right))
    roots = [find(index) for index in range(len(coordinates))]
    mapping = {root: group for group, root in enumerate(sorted(set(roots)))}
    return np.asarray([mapping[root] for root in roots], dtype=int)


def _minimum_curvature_delta(
    measured_length: float,
    azimuth_1_deg: float,
    dip_1_deg: float,
    azimuth_2_deg: float,
    dip_2_deg: float,
) -> tuple[float, float, float]:
    if measured_length <= 0:
        return 0.0, 0.0, 0.0
    a1, a2 = np.deg2rad([azimuth_1_deg, azimuth_2_deg])
    d1, d2 = np.deg2rad([dip_1_deg, dip_2_deg])
    cos_dogleg = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(a2 - a1)
    dogleg = float(np.arccos(np.clip(cos_dogleg, -1.0, 1.0)))
    ratio = 1.0 if abs(dogleg) < 1e-12 else 2.0 / dogleg * math.tan(dogleg / 2.0)
    north = measured_length / 2.0 * (np.cos(d1) * np.cos(a1) + np.cos(d2) * np.cos(a2)) * ratio
    east = measured_length / 2.0 * (np.cos(d1) * np.sin(a1) + np.cos(d2) * np.sin(a2)) * ratio
    vertical = measured_length / 2.0 * (np.sin(d1) + np.sin(d2)) * ratio
    return float(east), float(north), float(vertical)


@dataclass
class HoleTrajectory:
    depths: np.ndarray
    azimuths: np.ndarray
    dips: np.ndarray
    east: np.ndarray
    north: np.ndarray
    vertical: np.ndarray

    @classmethod
    def from_survey(cls, survey: pd.DataFrame) -> "HoleTrajectory":
        stations = survey[["depth", "azimuth", "dip"]].dropna().sort_values("depth")
        stations = stations.drop_duplicates("depth", keep="last")
        if stations.empty:
            raise ValueError("drillhole has no valid survey stations")
        if float(stations.iloc[0]["depth"]) > 0:
            first = stations.iloc[[0]].copy()
            first.loc[:, "depth"] = 0.0
            stations = pd.concat([first, stations], ignore_index=True)
        depths = stations["depth"].to_numpy(float)
        azimuths = stations["azimuth"].to_numpy(float) % 360.0
        dips = stations["dip"].to_numpy(float)
        east = np.zeros(len(stations), dtype=float)
        north = np.zeros(len(stations), dtype=float)
        vertical = np.zeros(len(stations), dtype=float)
        for idx in range(1, len(stations)):
            de, dn, dz = _minimum_curvature_delta(
                depths[idx] - depths[idx - 1], azimuths[idx - 1], dips[idx - 1],
                azimuths[idx], dips[idx]
            )
            east[idx] = east[idx - 1] + de
            north[idx] = north[idx - 1] + dn
            vertical[idx] = vertical[idx - 1] + dz
        return cls(depths, azimuths, dips, east, north, vertical)

    def position(self, measured_depth: float) -> tuple[float, float, float, float, float]:
        md = float(measured_depth)
        idx = int(np.searchsorted(self.depths, md, side="right") - 1)
        idx = min(max(idx, 0), len(self.depths) - 1)
        if idx == len(self.depths) - 1 or md >= self.depths[-1]:
            target_azimuth, target_dip = self.azimuths[idx], self.dips[idx]
        else:
            span = self.depths[idx + 1] - self.depths[idx]
            fraction = 0.0 if span <= 0 else (md - self.depths[idx]) / span
            delta_azimuth = ((self.azimuths[idx + 1] - self.azimuths[idx] + 180.0) % 360.0) - 180.0
            target_azimuth = (self.azimuths[idx] + fraction * delta_azimuth) % 360.0
            target_dip = self.dips[idx] + fraction * (self.dips[idx + 1] - self.dips[idx])
        de, dn, dz = _minimum_curvature_delta(
            md - self.depths[idx], self.azimuths[idx], self.dips[idx],
            target_azimuth, target_dip
        )
        return (
            self.east[idx] + de, self.north[idx] + dn, self.vertical[idx] + dz,
            float(target_azimuth), float(target_dip),
        )


def load_qidashan_intervals(mdb_path: Path, n_spatial_blocks: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    collar = export_mdb_table(mdb_path, "定位表")
    survey = export_mdb_table(mdb_path, "测斜表")
    assay = export_mdb_table(mdb_path, "化验表")
    collar.columns = ["hole", "collar_x", "collar_y", "collar_z", "max_depth", "trajectory_type"]
    survey.columns = ["hole", "depth", "azimuth", "dip"]
    assay.columns = ["hole", "from", "to", "TFe", "FeO", "SFe", "magnetic_rate"]
    for frame in (collar, survey, assay):
        frame["hole"] = _normalise_hole(frame["hole"])
    for column in ["collar_x", "collar_y", "collar_z", "max_depth"]:
        collar[column] = pd.to_numeric(collar[column], errors="coerce")
    for column in ["depth", "azimuth", "dip"]:
        survey[column] = pd.to_numeric(survey[column], errors="coerce")
    for column in ["from", "to", "TFe", "FeO", "SFe", "magnetic_rate"]:
        assay[column] = pd.to_numeric(assay[column], errors="coerce")
    assay = assay.dropna(subset=["hole", "from", "to", "TFe"]).copy()
    assay["length"] = assay["to"] - assay["from"]
    if assay.duplicated(["hole", "from", "to"]).any():
        raise ValueError("duplicate assay intervals detected")
    if (assay["length"] <= 0).any():
        raise ValueError("non-positive assay interval length detected")
    assay_holes = set(assay["hole"])
    missing_collar = assay_holes - set(collar["hole"])
    missing_survey = assay_holes - set(survey["hole"])
    if missing_collar or missing_survey:
        raise ValueError(f"assay holes missing collar={missing_collar}, survey={missing_survey}")
    collar_one = collar.drop_duplicates("hole").set_index("hole")
    trajectories = {
        hole: HoleTrajectory.from_survey(survey.loc[survey["hole"] == hole])
        for hole in sorted(assay_holes)
    }
    rows: list[dict] = []
    for record in assay.to_dict(orient="records"):
        midpoint = (float(record["from"]) + float(record["to"])) / 2.0
        de, dn, dz, azimuth, dip = trajectories[record["hole"]].position(midpoint)
        collar_row = collar_one.loc[record["hole"]]
        row = dict(record)
        row.update({
            "interval_id": f"{record['hole']}_{float(record['from']):.2f}_{float(record['to']):.2f}",
            "mid_depth": midpoint,
            "x": float(collar_row.collar_x) + de,
            "y": float(collar_row.collar_y) + dn,
            "z": float(collar_row.collar_z) + dz,
            "collar_x": float(collar_row.collar_x),
            "collar_y": float(collar_row.collar_y),
            "collar_z": float(collar_row.collar_z),
            "max_depth": float(collar_row.max_depth),
            "depth_ratio": midpoint / max(float(collar_row.max_depth), 1e-6),
            "azimuth_sin": math.sin(math.radians(azimuth)),
            "azimuth_cos": math.cos(math.radians(azimuth)),
            "dip_sin": math.sin(math.radians(dip)),
            "dip_cos": math.cos(math.radians(dip)),
        })
        rows.append(row)
    intervals = pd.DataFrame(rows).sort_values(["hole", "from", "to"]).reset_index(drop=True)
    for variable in ["FeO", "SFe", "magnetic_rate"]:
        intervals[f"{variable}_missing"] = intervals[variable].isna().astype(int)

    hole_coordinates = intervals.groupby("hole", as_index=False).agg(
        collar_x=("collar_x", "first"), collar_y=("collar_y", "first"),
        interval_count=("interval_id", "size"),
    )
    xy = hole_coordinates[["collar_x", "collar_y"]].to_numpy(float)
    hole_coordinates["physical_hole_group"] = _collar_connected_components(xy, radius_m=5.0)
    centred = xy - xy.mean(axis=0)
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    hole_coordinates["principal_axis_score"] = centred @ vh[0]
    hole_coordinates = hole_coordinates.sort_values("principal_axis_score").reset_index(drop=True)
    midpoint_count = hole_coordinates["interval_count"].cumsum() - hole_coordinates["interval_count"] / 2.0
    hole_coordinates["spatial_block"] = np.minimum(
        (midpoint_count / hole_coordinates["interval_count"].sum() * n_spatial_blocks).astype(int),
        n_spatial_blocks - 1,
    )
    intervals = intervals.merge(
        hole_coordinates[["hole", "physical_hole_group", "spatial_block"]],
        on="hole", how="left",
    )
    audit = pd.DataFrame([
        {"check": "interval_count", "value": len(intervals)},
        {"check": "hole_count", "value": intervals["hole"].nunique()},
        {"check": "physical_hole_group_count_5m", "value": intervals["physical_hole_group"].nunique()},
        {"check": "max_names_per_physical_hole_group", "value": intervals.groupby("physical_hole_group")["hole"].nunique().max()},
        {"check": "duplicate_interval_count", "value": intervals.duplicated(["hole", "from", "to"]).sum()},
        {"check": "invalid_length_count", "value": (intervals["length"] <= 0).sum()},
        {"check": "FeO_missing_count", "value": intervals["FeO"].isna().sum()},
        {"check": "SFe_missing_count", "value": intervals["SFe"].isna().sum()},
        {"check": "magnetic_missing_count", "value": intervals["magnetic_rate"].isna().sum()},
        {"check": "TFe_mean", "value": intervals["TFe"].mean()},
        {"check": "TFe_std", "value": intervals["TFe"].std(ddof=1)},
        {"check": "TFe_min", "value": intervals["TFe"].min()},
        {"check": "TFe_max", "value": intervals["TFe"].max()},
        {"check": "spatial_block_count", "value": intervals["spatial_block"].nunique()},
    ])
    return intervals, audit


def audit_target_derived_ratio(data: pd.DataFrame) -> pd.DataFrame:
    complete = data[["TFe", "FeO", "magnetic_rate"]].notna().all(axis=1) & data["FeO"].ne(0)
    ratio = data.loc[complete, "TFe"] / data.loc[complete, "FeO"]
    reconstructed = data.loc[complete, "FeO"] * data.loc[complete, "magnetic_rate"]
    residual = data.loc[complete, "TFe"] - reconstructed
    return pd.DataFrame([
        {"check": "complete_TFe_FeO_ratio_rows", "value": int(complete.sum()), "interpretation": "rows used in formula audit"},
        {"check": "corr_source_ratio_vs_TFe_div_FeO", "value": float(data.loc[complete, "magnetic_rate"].corr(ratio)), "interpretation": "near one indicates target-derived ratio"},
        {"check": "share_abs_TFe_minus_FeO_times_ratio_le_0_15", "value": float((residual.abs() <= 0.15).mean()), "interpretation": "direct algebraic reconstruction rate"},
        {"check": "share_abs_TFe_minus_FeO_times_ratio_le_0_50", "value": float((residual.abs() <= 0.50).mean()), "interpretation": "direct algebraic reconstruction rate"},
        {"check": "median_abs_reconstruction_error", "value": float(residual.abs().median()), "interpretation": "TFe percentage points"},
    ])


def make_outer_splits(data: pd.DataFrame, scheme: str, n_splits: int = 5):
    indices = np.arange(len(data))
    if scheme == "drillhole":
        groups = data["physical_hole_group"] if "physical_hole_group" in data else data["hole"]
        return list(GroupKFold(n_splits=n_splits).split(indices, groups=groups))
    if scheme == "spatial_block":
        return list(LeaveOneGroupOut().split(indices, groups=data["spatial_block"]))
    raise ValueError(f"unsupported validation scheme: {scheme}")


def make_inner_splits(data: pd.DataFrame, scheme: str, n_splits: int = 3):
    indices = np.arange(len(data))
    if scheme == "drillhole":
        groups = data["physical_hole_group"] if "physical_hole_group" in data else data["hole"]
        folds = min(n_splits, pd.Series(groups).nunique())
        return list(GroupKFold(n_splits=folds).split(indices, groups=groups))
    if scheme == "spatial_block":
        blocks = data["spatial_block"].nunique()
        if blocks >= 3:
            folds = min(n_splits, blocks)
            return list(GroupKFold(n_splits=folds).split(indices, groups=data["spatial_block"]))
        groups = data["physical_hole_group"] if "physical_hole_group" in data else data["hole"]
        folds = min(n_splits, pd.Series(groups).nunique())
        return list(GroupKFold(n_splits=folds).split(indices, groups=groups))
    raise ValueError(f"unsupported validation scheme: {scheme}")


def _scaled_coordinates(data: pd.DataFrame, z_scale: float) -> np.ndarray:
    coords = data[["x", "y", "z"]].to_numpy(float).copy()
    coords[:, 2] *= z_scale
    return coords


def idw_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    k: int,
    power: float,
    z_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    train_coords = _scaled_coordinates(train, z_scale)
    test_coords = _scaled_coordinates(test, z_scale)
    tree = cKDTree(train_coords)
    distances, neighbours = tree.query(test_coords, k=min(k, len(train)))
    if distances.ndim == 1:
        distances = distances[:, None]
        neighbours = neighbours[:, None]
    weights = 1.0 / np.maximum(distances, 1e-9) ** power
    prediction = np.sum(weights * train["TFe"].to_numpy(float)[neighbours], axis=1) / np.sum(weights, axis=1)
    return prediction, distances[:, 0]


IDW_CANDIDATES = [
    {"k": 4, "power": 2.0, "z_scale": 0.25},
    {"k": 8, "power": 2.0, "z_scale": 0.25},
    {"k": 16, "power": 2.0, "z_scale": 0.25},
    {"k": 8, "power": 1.0, "z_scale": 0.5},
    {"k": 16, "power": 1.5, "z_scale": 0.5},
    {"k": 32, "power": 2.0, "z_scale": 1.0},
    {"k": 16, "power": 2.5, "z_scale": 2.0},
]


MODEL_CANDIDATES = [
    {"depth": 6, "learning_rate": 0.08, "iterations": 250, "l2_leaf_reg": 5.0},
    {"depth": 8, "learning_rate": 0.05, "iterations": 350, "l2_leaf_reg": 8.0},
    {"depth": 10, "learning_rate": 0.035, "iterations": 450, "l2_leaf_reg": 12.0},
]


def _fill_training_medians(train: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    train_x = train[features].copy()
    test_x = test[features].copy()
    medians = train_x.median(numeric_only=True)
    return train_x.fillna(medians).fillna(0.0), test_x.fillna(medians).fillna(0.0)


def fit_predict_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    params: dict,
    family: str,
    seed: int,
) -> np.ndarray:
    if set(features) & FORBIDDEN_OUTCOME_FEATURES:
        raise ValueError("outcome-derived or split fields present in feature contract")
    train_x, test_x = _fill_training_medians(train, test, features)
    if family == "catboost":
        model = CatBoostRegressor(
            loss_function="RMSE", random_seed=seed, verbose=False,
            allow_writing_files=False, thread_count=-1, **params,
        )
    elif family == "xgboost":
        model = XGBRegressor(
            objective="reg:squarederror", random_state=seed, n_jobs=-1,
            tree_method="hist", max_depth=params["depth"],
            learning_rate=params["learning_rate"], n_estimators=params["iterations"],
            reg_lambda=params["l2_leaf_reg"], subsample=0.85, colsample_bytree=0.9,
        )
    else:
        raise ValueError(f"unknown model family: {family}")
    model.fit(train_x, train["TFe"].to_numpy(float))
    return model.predict(test_x)


def _pooled_r2(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    return float(r2_score(np.asarray(y_true), np.asarray(y_pred)))


def tune_idw(train: pd.DataFrame, scheme: str, n_inner: int) -> tuple[dict, list[dict]]:
    records: list[dict] = []
    splits = make_inner_splits(train, scheme, n_inner)
    for candidate in IDW_CANDIDATES:
        observed, predicted = [], []
        for inner_train, inner_valid in splits:
            pred, _ = idw_predict(train.iloc[inner_train], train.iloc[inner_valid], **candidate)
            observed.extend(train.iloc[inner_valid]["TFe"].to_numpy(float))
            predicted.extend(pred)
        score = _pooled_r2(observed, predicted)
        records.append({**candidate, "inner_r2": score})
    best = max(records, key=lambda row: row["inner_r2"])
    return {key: best[key] for key in ["k", "power", "z_scale"]}, records


def oof_idw_predictions(
    train: pd.DataFrame,
    scheme: str,
    params: dict,
    n_inner: int,
) -> np.ndarray:
    oof = np.full(len(train), np.nan)
    for inner_train, inner_valid in make_inner_splits(train, scheme, n_inner):
        oof[inner_valid], _ = idw_predict(
            train.iloc[inner_train], train.iloc[inner_valid], **params
        )
    if np.isnan(oof).any():
        raise RuntimeError("incomplete training-only IDW OOF predictions")
    return oof


def convex_fusion_weight(
    observed: np.ndarray,
    tree_prediction: np.ndarray,
    idw_prediction: np.ndarray,
) -> float:
    difference = tree_prediction - idw_prediction
    denominator = float(np.dot(difference, difference))
    if denominator <= 1e-12:
        return 0.5
    weight = float(np.dot(observed - idw_prediction, difference) / denominator)
    return float(np.clip(weight, 0.0, 1.0))


def tune_tree_model(
    train: pd.DataFrame,
    scheme: str,
    features: list[str],
    family: str,
    n_inner: int,
    seed: int,
) -> tuple[dict, list[dict]]:
    records: list[dict] = []
    splits = make_inner_splits(train, scheme, n_inner)
    for candidate_idx, candidate in enumerate(MODEL_CANDIDATES):
        observed, predicted = [], []
        for fold_idx, (inner_train, inner_valid) in enumerate(splits):
            pred = fit_predict_model(
                train.iloc[inner_train], train.iloc[inner_valid], features,
                candidate, family, seed + 100 * candidate_idx + fold_idx,
            )
            observed.extend(train.iloc[inner_valid]["TFe"].to_numpy(float))
            predicted.extend(pred)
        score = _pooled_r2(observed, predicted)
        records.append({**candidate, "inner_r2": score})
    best = max(records, key=lambda row: row["inner_r2"])
    return {key: best[key] for key in ["depth", "learning_rate", "iterations", "l2_leaf_reg"]}, records


def training_only_tail_calibration(
    observed: np.ndarray,
    prediction: np.ndarray,
    high_grade_threshold: float,
) -> tuple[str, object | None, list[dict]]:
    candidates: list[tuple[str, object | None, np.ndarray]] = [("none", None, prediction)]
    isotonic = IsotonicRegression(out_of_bounds="clip")
    candidates.append(("isotonic", isotonic.fit(prediction, observed), isotonic.predict(prediction)))
    threshold_pred = float(np.quantile(prediction, 0.80))
    scale = max(float(np.std(prediction)) * 0.15, 0.5)
    for uplift in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        corrected = prediction + uplift / (1.0 + np.exp(-(prediction - threshold_pred) / scale))
        candidates.append((f"tail_uplift_{uplift:g}", (uplift, threshold_pred, scale), corrected))
    rows = []
    high_mask = observed >= high_grade_threshold
    for name, payload, corrected in candidates:
        r2 = _pooled_r2(observed, corrected)
        high_bias = float(np.mean(corrected[high_mask] - observed[high_mask])) if high_mask.any() else np.nan
        objective = r2 - 0.0025 * abs(high_bias)
        rows.append({"calibration": name, "r2": r2, "high_grade_bias": high_bias, "objective": objective})
    best_index = int(np.argmax([row["objective"] for row in rows]))
    return candidates[best_index][0], candidates[best_index][1], rows


def apply_calibration(name: str, payload: object | None, prediction: np.ndarray) -> np.ndarray:
    if name == "none":
        return prediction
    if name == "isotonic":
        return payload.predict(prediction)  # type: ignore[union-attr]
    if name.startswith("tail_uplift_"):
        uplift, threshold, scale = payload  # type: ignore[misc]
        return prediction + uplift / (1.0 + np.exp(-(prediction - threshold) / scale))
    raise ValueError(name)


def oof_tree_predictions(
    train: pd.DataFrame,
    scheme: str,
    features: list[str],
    family: str,
    params: dict,
    n_inner: int,
    seed: int,
) -> np.ndarray:
    oof = np.full(len(train), np.nan)
    for fold_idx, (inner_train, inner_valid) in enumerate(make_inner_splits(train, scheme, n_inner)):
        oof[inner_valid] = fit_predict_model(
            train.iloc[inner_train], train.iloc[inner_valid], features, params,
            family, seed + fold_idx,
        )
    if np.isnan(oof).any():
        raise RuntimeError("incomplete training-only OOF predictions")
    return oof


def evaluate_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled_rows, fold_rows = [], []
    group_columns = ["scheme", "feature_contract", "model"]
    for keys, frame in predictions.groupby(group_columns, sort=False):
        y = frame["observed"].to_numpy(float)
        p = frame["predicted"].to_numpy(float)
        far_cut = float(frame["nearest_train_distance"].quantile(0.75))
        far = frame["nearest_train_distance"] >= far_cut
        high = frame["observed"] >= frame["train_high_grade_threshold"]
        pooled_rows.append({
            **dict(zip(group_columns, keys)), "n": len(frame),
            "r2": r2_score(y, p), "rmse": mean_squared_error(y, p) ** 0.5,
            "mae": mean_absolute_error(y, p), "bias": float(np.mean(p - y)),
            "far_quartile_r2": r2_score(frame.loc[far, "observed"], frame.loc[far, "predicted"]),
            "far_quartile_n": int(far.sum()),
            "high_grade_bias": float(np.mean(frame.loc[high, "predicted"] - frame.loc[high, "observed"])),
            "high_grade_n": int(high.sum()),
        })
        for fold, fold_frame in frame.groupby("fold"):
            fold_rows.append({
                **dict(zip(group_columns, keys)), "fold": int(fold), "n": len(fold_frame),
                "r2": r2_score(fold_frame["observed"], fold_frame["predicted"]),
                "rmse": mean_squared_error(fold_frame["observed"], fold_frame["predicted"]) ** 0.5,
                "bias": float(np.mean(fold_frame["predicted"] - fold_frame["observed"])),
            })
    return pd.DataFrame(pooled_rows), pd.DataFrame(fold_rows)


def run_nested_experiment(
    data: pd.DataFrame,
    schemes: list[str],
    n_outer: int,
    n_inner: int,
    high_grade_quantile: float,
    seed: int,
    checkpoint_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    tuning_rows: list[dict] = []
    calibration_rows: list[dict] = []
    contracts = {
        "spatial_deployable": SPATIAL_FEATURES,
        "coassay_no_target_ratio": COASSAY_FEATURES,
    }
    for scheme in schemes:
        for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(make_outer_splits(data, scheme, n_outer)):
            print(f"START scheme={scheme} outer_fold={fold_idx + 1}/{n_outer}", flush=True)
            outer_train = data.iloc[outer_train_idx].reset_index(drop=True)
            outer_test = data.iloc[outer_test_idx].reset_index(drop=True)
            high_threshold = float(outer_train["TFe"].quantile(high_grade_quantile))

            idw_params, idw_records = tune_idw(outer_train, scheme, n_inner)
            idw_prediction, nearest_distance = idw_predict(outer_train, outer_test, **idw_params)
            idw_oof_prediction = oof_idw_predictions(
                outer_train, scheme, idw_params, n_inner
            )
            for row in idw_records:
                tuning_rows.append({"scheme": scheme, "fold": fold_idx, "feature_contract": "spatial_deployable", "model": "IDW", **row})
            base = outer_test[["interval_id", "hole"]].copy()
            base["scheme"] = scheme
            base["fold"] = fold_idx
            base["feature_contract"] = "spatial_deployable"
            base["model"] = "IDW"
            base["observed"] = outer_test["TFe"].to_numpy(float)
            base["predicted"] = idw_prediction
            base["nearest_train_distance"] = nearest_distance
            base["train_high_grade_threshold"] = high_threshold
            prediction_frames.append(base)

            for contract_name, features in contracts.items():
                for family in ["catboost", "xgboost"]:
                    print(
                        f"  FIT contract={contract_name} family={family}",
                        flush=True,
                    )
                    params, records = tune_tree_model(
                        outer_train, scheme, features, family, n_inner, seed + fold_idx * 1000
                    )
                    for row in records:
                        tuning_rows.append({
                            "scheme": scheme, "fold": fold_idx, "feature_contract": contract_name,
                            "model": family, **row,
                        })
                    raw_prediction = fit_predict_model(
                        outer_train, outer_test, features, params, family,
                        seed + fold_idx * 1000 + 99,
                    )
                    oof_prediction = oof_tree_predictions(
                        outer_train, scheme, features, family, params, n_inner,
                        seed + fold_idx * 1000 + 200,
                    )
                    calibration_name, calibration_payload, records = training_only_tail_calibration(
                        outer_train["TFe"].to_numpy(float), oof_prediction, high_threshold
                    )
                    for row in records:
                        calibration_rows.append({
                            "scheme": scheme, "fold": fold_idx, "feature_contract": contract_name,
                            "model": family, "selected": row["calibration"] == calibration_name, **row,
                        })
                    calibrated_prediction = apply_calibration(calibration_name, calibration_payload, raw_prediction)
                    for label, prediction in [(family, raw_prediction), (f"{family}_calibrated", calibrated_prediction)]:
                        frame = outer_test[["interval_id", "hole"]].copy()
                        frame["scheme"] = scheme
                        frame["fold"] = fold_idx
                        frame["feature_contract"] = contract_name
                        frame["model"] = label
                        frame["observed"] = outer_test["TFe"].to_numpy(float)
                        frame["predicted"] = prediction
                        frame["nearest_train_distance"] = nearest_distance
                        frame["train_high_grade_threshold"] = high_threshold
                        prediction_frames.append(frame)
                    if contract_name == "spatial_deployable":
                        fusion_weight = convex_fusion_weight(
                            outer_train["TFe"].to_numpy(float), oof_prediction,
                            idw_oof_prediction,
                        )
                        fusion_oof = fusion_weight * oof_prediction + (1.0 - fusion_weight) * idw_oof_prediction
                        fusion_prediction = fusion_weight * raw_prediction + (1.0 - fusion_weight) * idw_prediction
                        fusion_calibration_name, fusion_calibration_payload, fusion_calibration_records = training_only_tail_calibration(
                            outer_train["TFe"].to_numpy(float), fusion_oof, high_threshold
                        )
                        for row in fusion_calibration_records:
                            calibration_rows.append({
                                "scheme": scheme, "fold": fold_idx,
                                "feature_contract": contract_name,
                                "model": f"{family}_idw_fusion",
                                "fusion_weight_tree": fusion_weight,
                                "selected": row["calibration"] == fusion_calibration_name,
                                **row,
                            })
                        calibrated_fusion = apply_calibration(
                            fusion_calibration_name, fusion_calibration_payload,
                            fusion_prediction,
                        )
                        for label, prediction in [
                            (f"{family}_idw_fusion", fusion_prediction),
                            (f"{family}_idw_fusion_calibrated", calibrated_fusion),
                        ]:
                            frame = outer_test[["interval_id", "hole"]].copy()
                            frame["scheme"] = scheme
                            frame["fold"] = fold_idx
                            frame["feature_contract"] = contract_name
                            frame["model"] = label
                            frame["observed"] = outer_test["TFe"].to_numpy(float)
                            frame["predicted"] = prediction
                            frame["nearest_train_distance"] = nearest_distance
                            frame["train_high_grade_threshold"] = high_threshold
                            prediction_frames.append(frame)
            if checkpoint_dir is not None:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                pd.concat(prediction_frames, ignore_index=True).to_csv(
                    checkpoint_dir / "strict_oof_predictions_checkpoint.csv", index=False
                )
                pd.DataFrame(tuning_rows).to_csv(
                    checkpoint_dir / "nested_tuning_checkpoint.csv", index=False
                )
                pd.DataFrame(calibration_rows).to_csv(
                    checkpoint_dir / "training_only_calibration_checkpoint.csv", index=False
                )
            print(f"DONE scheme={scheme} outer_fold={fold_idx + 1}/{n_outer}", flush=True)
    return pd.concat(prediction_frames, ignore_index=True), pd.DataFrame(tuning_rows), pd.DataFrame(calibration_rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
