from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


NUMERIC_FEATURES = [
    "x",
    "y",
    "z",
    "mid_depth",
    "collar_z",
    "azimuth_sin",
    "azimuth_cos",
    "dip_sin",
    "dip_cos",
]
CATEGORICAL_FEATURES = ["lithology_group", "lithology"]
MODELS = ["Mean", "OrdinaryKriging", "XGBoost", "OOFResidualKriging", "ConvexFusion"]
SCHEMES = ["random_interval", "drillhole", "spatial_block"]


def _normalise_hole(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _minimum_curvature_delta(
    measured_length: float,
    azimuth_1_deg: float,
    dip_1_deg: float,
    azimuth_2_deg: float,
    dip_2_deg: float,
) -> tuple[float, float, float]:
    """Return east, north and vertical increments for a survey segment.

    Dip follows the source convention: 0 degrees is horizontal and negative
    values point downward. Azimuth is clockwise from north.
    """
    if measured_length <= 0:
        return 0.0, 0.0, 0.0
    a1, a2 = np.deg2rad([azimuth_1_deg, azimuth_2_deg])
    d1, d2 = np.deg2rad([dip_1_deg, dip_2_deg])
    cos_dl = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(a2 - a1)
    dogleg = float(np.arccos(np.clip(cos_dl, -1.0, 1.0)))
    ratio = 1.0 if abs(dogleg) < 1e-12 else 2.0 / dogleg * math.tan(dogleg / 2.0)
    north = measured_length / 2.0 * (
        np.cos(d1) * np.cos(a1) + np.cos(d2) * np.cos(a2)
    ) * ratio
    east = measured_length / 2.0 * (
        np.cos(d1) * np.sin(a1) + np.cos(d2) * np.sin(a2)
    ) * ratio
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
        s = survey[["depth", "azimuth", "dip"]].dropna().sort_values("depth")
        s = s.drop_duplicates("depth", keep="last")
        if s.empty:
            raise ValueError("A drillhole has no valid survey stations")
        if float(s.iloc[0]["depth"]) > 0:
            first = s.iloc[[0]].copy()
            first.loc[:, "depth"] = 0.0
            s = pd.concat([first, s], ignore_index=True)
        depths = s["depth"].to_numpy(float)
        azimuths = s["azimuth"].to_numpy(float) % 360.0
        dips = s["dip"].to_numpy(float)
        east = np.zeros(len(s), dtype=float)
        north = np.zeros(len(s), dtype=float)
        vertical = np.zeros(len(s), dtype=float)
        for i in range(1, len(s)):
            de, dn, dz = _minimum_curvature_delta(
                depths[i] - depths[i - 1],
                azimuths[i - 1],
                dips[i - 1],
                azimuths[i],
                dips[i],
            )
            east[i] = east[i - 1] + de
            north[i] = north[i - 1] + dn
            vertical[i] = vertical[i - 1] + dz
        return cls(depths, azimuths, dips, east, north, vertical)

    def position(self, measured_depth: float) -> tuple[float, float, float, float, float]:
        md = float(measured_depth)
        if md <= self.depths[0]:
            idx = 0
        else:
            idx = int(np.searchsorted(self.depths, md, side="right") - 1)
            idx = min(max(idx, 0), len(self.depths) - 1)
        if idx == len(self.depths) - 1 or md >= self.depths[-1]:
            target_az, target_dip = self.azimuths[idx], self.dips[idx]
            de, dn, dz = _minimum_curvature_delta(
                md - self.depths[idx], target_az, target_dip, target_az, target_dip
            )
        else:
            span = self.depths[idx + 1] - self.depths[idx]
            fraction = 0.0 if span <= 0 else (md - self.depths[idx]) / span
            # Interpolate azimuth along the shortest angular path.
            delta_az = ((self.azimuths[idx + 1] - self.azimuths[idx] + 180.0) % 360.0) - 180.0
            target_az = (self.azimuths[idx] + fraction * delta_az) % 360.0
            target_dip = self.dips[idx] + fraction * (self.dips[idx + 1] - self.dips[idx])
            de, dn, dz = _minimum_curvature_delta(
                md - self.depths[idx],
                self.azimuths[idx],
                self.dips[idx],
                target_az,
                target_dip,
            )
        return (
            self.east[idx] + de,
            self.north[idx] + dn,
            self.vertical[idx] + dz,
            float(target_az),
            float(target_dip),
        )


def _lithology_group(text: str) -> str:
    text = str(text)
    if "赤铁" in text:
        return "hematite_quartzite"
    if "磁铁" in text or "磁铁矿" in text:
        return "magnetite_quartzite"
    if any(token in text for token in ["变粒岩", "混合岩", "片麻", "角闪岩"]):
        return "metamorphic_wall_rock"
    if any(token in text for token in ["正长", "霓辉", "辉石", "闪长"]):
        return "alkaline_mafic_rock"
    return "other"


def load_clean_intervals(data_dir: Path, n_spatial_blocks: int, random_seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = data_dir / "实验数据.xlsx"
    assay = pd.read_excel(source, sheet_name="化验表")
    collar = pd.read_excel(data_dir / "定位.xlsx")
    survey = pd.read_excel(data_dir / "侧斜.xlsx").iloc[:, :4].copy()
    lithology = pd.read_excel(data_dir / "岩性.xlsx")

    assay.columns = ["hole", "from", "to", "length", "TFe", "FeO", "SFe", "TFe_FeO"]
    collar.columns = ["hole", "collar_x", "collar_y", "collar_z", "max_depth", "trajectory_type"]
    survey.columns = ["hole", "depth", "azimuth", "dip"]
    lithology.columns = ["hole", "from", "to", "lithology"]
    for frame in [assay, collar, survey, lithology]:
        frame["hole"] = _normalise_hole(frame["hole"])
    for col in ["from", "to", "length", "TFe"]:
        assay[col] = pd.to_numeric(assay[col], errors="coerce")
    for col in ["collar_x", "collar_y", "collar_z", "max_depth"]:
        collar[col] = pd.to_numeric(collar[col], errors="coerce")
    for col in ["depth", "azimuth", "dip"]:
        survey[col] = pd.to_numeric(survey[col], errors="coerce")
    for col in ["from", "to"]:
        lithology[col] = pd.to_numeric(lithology[col], errors="coerce")

    assay = assay.dropna(subset=["hole", "from", "to", "length", "TFe"]).copy()
    if assay.duplicated(["hole", "from", "to"]).any():
        raise ValueError("Duplicate assay intervals detected")
    if ((assay["to"] <= assay["from"]) | ((assay["to"] - assay["from"] - assay["length"]).abs() > 0.02)).any():
        raise ValueError("Invalid assay interval lengths detected")
    assay_holes = set(assay["hole"])
    if assay_holes - set(collar["hole"]):
        raise ValueError("Some assayed holes do not have collar data")
    if assay_holes - set(survey["hole"]):
        raise ValueError("Some assayed holes do not have survey data")

    collar_one = collar.drop_duplicates("hole").set_index("hole")
    trajectories = {
        hole: HoleTrajectory.from_survey(
            survey.loc[survey["hole"] == hole, ["depth", "azimuth", "dip"]]
        )
        for hole in sorted(assay_holes)
    }
    rows: list[dict] = []
    for row in assay.to_dict(orient="records"):
        mid = (float(row["from"]) + float(row["to"])) / 2.0
        lit_match = lithology.loc[
            (lithology["hole"] == row["hole"])
            & (lithology["from"] <= mid)
            & (lithology["to"] >= mid),
            "lithology",
        ]
        lit = "Unknown" if lit_match.empty else str(lit_match.iloc[0]).strip()
        de, dn, dz, azimuth, dip = trajectories[row["hole"]].position(mid)
        c = collar_one.loc[row["hole"]]
        rows.append(
            {
                "interval_id": f"{row['hole']}_{float(row['from']):.2f}_{float(row['to']):.2f}",
                "hole": row["hole"],
                "from": float(row["from"]),
                "to": float(row["to"]),
                "length": float(row["length"]),
                "mid_depth": mid,
                "TFe": float(row["TFe"]),
                "x": float(c.collar_x) + de,
                "y": float(c.collar_y) + dn,
                "z": float(c.collar_z) + dz,
                "collar_x": float(c.collar_x),
                "collar_y": float(c.collar_y),
                "collar_z": float(c.collar_z),
                "azimuth": azimuth,
                "dip": dip,
                "azimuth_sin": math.sin(math.radians(azimuth)),
                "azimuth_cos": math.cos(math.radians(azimuth)),
                "dip_sin": math.sin(math.radians(dip)),
                "dip_cos": math.cos(math.radians(dip)),
                "lithology": lit,
                "lithology_group": _lithology_group(lit),
            }
        )
    intervals = pd.DataFrame(rows).sort_values(["hole", "from", "to"]).reset_index(drop=True)

    # Build contiguous, approximately sample-balanced spatial slabs along the
    # dominant collar-coordinate axis. The block definition is deterministic
    # and does not use grade values.
    hole_xy = intervals.groupby("hole", as_index=False).agg(
        collar_x=("collar_x", "first"),
        collar_y=("collar_y", "first"),
        interval_count=("interval_id", "size"),
    )
    xy = hole_xy[["collar_x", "collar_y"]].to_numpy(float)
    centred = xy - xy.mean(axis=0)
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    hole_xy["principal_axis_score"] = centred @ vh[0]
    hole_xy = hole_xy.sort_values("principal_axis_score").reset_index(drop=True)
    midpoint_count = hole_xy["interval_count"].cumsum() - hole_xy["interval_count"] / 2.0
    hole_xy["spatial_block"] = np.minimum(
        (midpoint_count / hole_xy["interval_count"].sum() * n_spatial_blocks).astype(int),
        n_spatial_blocks - 1,
    )
    intervals = intervals.merge(hole_xy[["hole", "spatial_block"]], on="hole", how="left")

    audit_rows = [
        ("assay_interval_count", len(intervals)),
        ("assayed_drillhole_count", intervals["hole"].nunique()),
        ("assayed_length_m", intervals["length"].sum()),
        ("missing_TFe_count", intervals["TFe"].isna().sum()),
        ("duplicate_interval_count", intervals.duplicated(["hole", "from", "to"]).sum()),
        ("lithology_midpoint_missing_count", (intervals["lithology"] == "Unknown").sum()),
        ("TFe_mean", intervals["TFe"].mean()),
        ("TFe_std", intervals["TFe"].std(ddof=1)),
        ("TFe_min", intervals["TFe"].min()),
        ("TFe_max", intervals["TFe"].max()),
        ("spatial_block_count", intervals["spatial_block"].nunique()),
        ("max_horizontal_deviation_m", np.hypot(intervals["x"] - intervals["collar_x"], intervals["y"] - intervals["collar_y"]).max()),
    ]
    audit = pd.DataFrame(audit_rows, columns=["item", "value"])
    return intervals, audit


def make_outer_splits(data: pd.DataFrame, scheme: str, n_splits: int, random_seed: int):
    indices = np.arange(len(data))
    if scheme == "random_interval":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        return list(splitter.split(indices))
    if scheme == "drillhole":
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(indices, groups=data["hole"].to_numpy()))
    if scheme == "spatial_block":
        splitter = LeaveOneGroupOut()
        return list(splitter.split(indices, groups=data["spatial_block"].to_numpy()))
    raise ValueError(f"Unknown validation scheme: {scheme}")


def make_inner_splits(data: pd.DataFrame, scheme: str, n_splits: int, random_seed: int):
    indices = np.arange(len(data))
    if scheme == "random_interval":
        folds = min(n_splits, len(data))
        return list(KFold(folds, shuffle=True, random_state=random_seed).split(indices))
    group_column = "hole" if scheme == "drillhole" else "spatial_block"
    groups = data[group_column].to_numpy()
    unique_groups = np.unique(groups)
    folds = min(n_splits, len(unique_groups))
    if folds < 2:
        raise ValueError(f"Not enough {group_column} groups for inner validation")
    if scheme == "spatial_block" and folds == len(unique_groups):
        return list(LeaveOneGroupOut().split(indices, groups=groups))
    return list(GroupKFold(folds).split(indices, groups=groups))


def build_xgb_pipeline(
    params: dict,
    random_seed: int,
    numeric_features: Sequence[str] = NUMERIC_FEATURES,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
) -> Pipeline:
    transformers = []
    if numeric_features:
        transformers.append(("numeric", "passthrough", list(numeric_features)))
    if categorical_features:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=5,
                    sparse_output=True,
                ),
                list(categorical_features),
            )
        )
    preprocessing = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )
    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=random_seed,
        n_jobs=2,
        reg_alpha=0.05,
        reg_lambda=2.0,
        gamma=0.0,
        **params,
    )
    return Pipeline([("preprocess", preprocessing), ("model", model)])


def xgb_oof_predictions(
    data: pd.DataFrame,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    params: dict,
    random_seed: int,
    numeric_features: Sequence[str] = NUMERIC_FEATURES,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
) -> np.ndarray:
    pred = np.full(len(data), np.nan, dtype=float)
    features = list(numeric_features) + list(categorical_features)
    for fold, (train_idx, valid_idx) in enumerate(splits):
        pipe = build_xgb_pipeline(
            params,
            random_seed + fold,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )
        pipe.fit(data.iloc[train_idx][features], data.iloc[train_idx]["TFe"])
        pred[valid_idx] = pipe.predict(data.iloc[valid_idx][features])
    if np.isnan(pred).any():
        raise RuntimeError("Incomplete XGBoost OOF predictions")
    return pred


def tune_xgb(
    data: pd.DataFrame,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    grid: Sequence[dict],
    random_seed: int,
    numeric_features: Sequence[str] = NUMERIC_FEATURES,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
) -> tuple[dict, np.ndarray, pd.DataFrame]:
    records = []
    best_params = None
    best_pred = None
    best_rmse = float("inf")
    for candidate_id, params in enumerate(grid):
        pred = xgb_oof_predictions(
            data,
            splits,
            params,
            random_seed + candidate_id * 101,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )
        rmse = float(mean_squared_error(data["TFe"], pred) ** 0.5)
        records.append({"candidate_id": candidate_id, "rmse": rmse, **params})
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = dict(params)
            best_pred = pred.copy()
    assert best_params is not None and best_pred is not None
    return best_params, best_pred, pd.DataFrame(records)


def anisotropic_coordinates(data: pd.DataFrame, vertical_scale: float) -> np.ndarray:
    xyz = data[["x", "y", "z"]].to_numpy(float).copy()
    xyz[:, 2] *= float(vertical_scale)
    return xyz


def _fit_exponential_variogram(
    coords: np.ndarray,
    values: np.ndarray,
    random_seed: int,
    pair_sample: int,
    n_bins: int,
) -> dict:
    rng = np.random.default_rng(random_seed)
    n = len(values)
    pair_count = min(pair_sample, max(n * (n - 1) // 2, 1))
    i = rng.integers(0, n, pair_count * 2)
    j = rng.integers(0, n, pair_count * 2)
    keep = i != j
    i, j = i[keep][:pair_count], j[keep][:pair_count]
    h = np.linalg.norm(coords[i] - coords[j], axis=1)
    gamma = 0.5 * (values[i] - values[j]) ** 2
    finite = np.isfinite(h) & np.isfinite(gamma) & (h > 0)
    h, gamma = h[finite], gamma[finite]
    variance = max(float(np.var(values, ddof=1)), 1e-6)
    if len(h) < 100:
        return {"nugget": 0.05 * variance, "partial_sill": 0.95 * variance, "range": 1.0}
    max_h = float(np.quantile(h, 0.9))
    edges = np.linspace(0.0, max_h, n_bins + 1)
    centers, empirical, counts = [], [], []
    for b in range(n_bins):
        mask = (h >= edges[b]) & (h < edges[b + 1])
        if mask.sum() >= 20:
            centers.append(float(np.median(h[mask])))
            empirical.append(float(np.median(gamma[mask])))
            counts.append(int(mask.sum()))
    centers = np.asarray(centers)
    empirical = np.asarray(empirical)
    counts = np.asarray(counts)

    def model(distance, nugget, partial_sill, range_):
        return nugget + partial_sill * (1.0 - np.exp(-3.0 * distance / np.maximum(range_, 1e-9)))

    initial = [0.05 * variance, 0.95 * variance, max(np.median(h), 1.0)]
    try:
        fitted, _ = curve_fit(
            model,
            centers,
            empirical,
            p0=initial,
            sigma=1.0 / np.sqrt(np.maximum(counts, 1)),
            absolute_sigma=False,
            bounds=([0.0, 1e-8, 1e-6], [2.0 * variance, 4.0 * variance, 5.0 * max_h]),
            maxfev=20000,
        )
        nugget, partial_sill, range_ = map(float, fitted)
    except Exception:
        nugget, partial_sill, range_ = initial
    return {"nugget": nugget, "partial_sill": partial_sill, "range": range_}


def local_ordinary_kriging(
    train_coords: np.ndarray,
    train_values: np.ndarray,
    predict_coords: np.ndarray,
    variogram: dict,
    n_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    k = min(int(n_neighbors), len(train_values))
    tree = cKDTree(train_coords)
    distances, indices = tree.query(predict_coords, k=k)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    predictions = np.empty(len(predict_coords), dtype=float)
    variances = np.empty(len(predict_coords), dtype=float)
    nugget = float(variogram["nugget"])
    partial_sill = float(variogram["partial_sill"])
    range_ = max(float(variogram["range"]), 1e-9)
    total_sill = nugget + partial_sill
    for row_id, (neighbor_idx, target_distance) in enumerate(zip(indices, distances)):
        local_coords = train_coords[neighbor_idx]
        local_values = train_values[neighbor_idx]
        delta = local_coords[:, None, :] - local_coords[None, :, :]
        h = np.linalg.norm(delta, axis=2)
        covariance = partial_sill * np.exp(-3.0 * h / range_)
        np.fill_diagonal(covariance, total_sill + 1e-8 * max(total_sill, 1.0))
        cross_cov = partial_sill * np.exp(-3.0 * np.asarray(target_distance) / range_)
        system = np.empty((k + 1, k + 1), dtype=float)
        system[:k, :k] = covariance
        system[:k, k] = 1.0
        system[k, :k] = 1.0
        system[k, k] = 0.0
        rhs = np.r_[cross_cov, 1.0]
        try:
            solution = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            solution = np.linalg.lstsq(system, rhs, rcond=None)[0]
        weights, multiplier = solution[:k], solution[k]
        predictions[row_id] = float(np.dot(weights, local_values))
        variances[row_id] = max(float(total_sill - np.dot(weights, cross_cov) + multiplier), 0.0)
    return predictions, variances


def ok_oof_predictions(
    data: pd.DataFrame,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    vertical_scale: float,
    n_neighbors: int,
    random_seed: int,
    pair_sample: int,
    n_bins: int,
) -> tuple[np.ndarray, list[dict]]:
    pred = np.full(len(data), np.nan, dtype=float)
    variograms: list[dict] = []
    for fold, (train_idx, valid_idx) in enumerate(splits):
        train = data.iloc[train_idx]
        valid = data.iloc[valid_idx]
        train_coords = anisotropic_coordinates(train, vertical_scale)
        valid_coords = anisotropic_coordinates(valid, vertical_scale)
        variogram = _fit_exponential_variogram(
            train_coords,
            train["TFe"].to_numpy(float),
            random_seed + fold,
            pair_sample,
            n_bins,
        )
        fold_pred, _ = local_ordinary_kriging(
            train_coords,
            train["TFe"].to_numpy(float),
            valid_coords,
            variogram,
            n_neighbors,
        )
        pred[valid_idx] = fold_pred
        variograms.append(variogram)
    if np.isnan(pred).any():
        raise RuntimeError("Incomplete ordinary-kriging OOF predictions")
    return pred, variograms


def tune_ordinary_kriging(
    data: pd.DataFrame,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    vertical_scales: Iterable[float],
    neighbor_candidates: Iterable[int],
    random_seed: int,
    pair_sample: int,
    n_bins: int,
) -> tuple[dict, np.ndarray, pd.DataFrame]:
    records = []
    best = None
    best_pred = None
    best_rmse = float("inf")
    candidate_id = 0
    for vertical_scale in vertical_scales:
        for n_neighbors in neighbor_candidates:
            pred, _ = ok_oof_predictions(
                data,
                splits,
                float(vertical_scale),
                int(n_neighbors),
                random_seed + candidate_id * 101,
                pair_sample,
                n_bins,
            )
            rmse = float(mean_squared_error(data["TFe"], pred) ** 0.5)
            records.append(
                {
                    "candidate_id": candidate_id,
                    "vertical_scale": float(vertical_scale),
                    "n_neighbors": int(n_neighbors),
                    "rmse": rmse,
                }
            )
            if rmse < best_rmse:
                best_rmse = rmse
                best = {"vertical_scale": float(vertical_scale), "n_neighbors": int(n_neighbors)}
                best_pred = pred.copy()
            candidate_id += 1
    assert best is not None and best_pred is not None
    return best, best_pred, pd.DataFrame(records)


def convex_weight(y: np.ndarray, xgb_pred: np.ndarray, ok_pred: np.ndarray) -> float:
    direction = xgb_pred - ok_pred
    denominator = float(np.dot(direction, direction))
    if denominator <= 1e-12:
        return 0.5
    weight = float(np.dot(direction, y - ok_pred) / denominator)
    return float(np.clip(weight, 0.0, 1.0))


def prediction_metrics(y: np.ndarray, pred: np.ndarray, weights: np.ndarray | None = None) -> dict:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    residual = pred - y
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    if weights is None:
        weighted_rmse = rmse
        weighted_mae = mae
    else:
        w = np.asarray(weights, dtype=float)
        weighted_rmse = float(np.sqrt(np.average(residual ** 2, weights=w)))
        weighted_mae = float(np.average(np.abs(residual), weights=w))
    r2 = float(r2_score(y, pred)) if np.var(y) > 0 else float("nan")
    correlation = float(np.corrcoef(y, pred)[0, 1]) if np.std(pred) > 0 and np.std(y) > 0 else float("nan")
    if np.std(pred) > 1e-12:
        slope, intercept = np.polyfit(pred, y, 1)
    else:
        slope, intercept = float("nan"), float("nan")
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "bias": float(np.mean(residual)),
        "correlation": correlation,
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "length_weighted_rmse": weighted_rmse,
        "length_weighted_mae": weighted_mae,
    }


def run_nested_validation(data: pd.DataFrame, config: dict, output_dir: Path) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict] = []
    tuning_records: list[pd.DataFrame] = []
    variogram_records: list[dict] = []
    fusion_records: list[dict] = []
    timing_records: list[dict] = []
    random_seed = int(config["random_seed"])
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    for scheme_id, scheme in enumerate(SCHEMES):
        outer_splits = make_outer_splits(data, scheme, int(config["outer_folds"]), random_seed)
        for fold, (train_idx, test_idx) in enumerate(outer_splits, start=1):
            start_time = time.perf_counter()
            train = data.iloc[train_idx].reset_index(drop=True)
            test = data.iloc[test_idx].reset_index(drop=True)
            if scheme in {"drillhole", "spatial_block"}:
                overlap = set(train["hole"]) & set(test["hole"])
                if overlap:
                    raise RuntimeError(f"Held-out-hole leakage in {scheme} fold {fold}: {sorted(overlap)}")
            inner_splits = make_inner_splits(
                train, scheme, int(config["inner_folds"]), random_seed + scheme_id * 1000 + fold
            )

            best_xgb, xgb_inner_oof, xgb_tuning = tune_xgb(
                train,
                inner_splits,
                config["xgboost_grid"],
                random_seed + scheme_id * 10000 + fold * 101,
            )
            xgb_tuning.insert(0, "scheme", scheme)
            xgb_tuning.insert(1, "outer_fold", fold)
            xgb_tuning.insert(2, "component", "XGBoost")
            tuning_records.append(xgb_tuning)

            kcfg = config["kriging"]
            best_ok, ok_inner_oof, ok_tuning = tune_ordinary_kriging(
                train,
                inner_splits,
                kcfg["vertical_scale_candidates"],
                kcfg["neighbor_candidates"],
                random_seed + scheme_id * 20000 + fold * 211,
                int(kcfg["variogram_pair_sample"]),
                int(kcfg["variogram_bins"]),
            )
            ok_tuning.insert(0, "scheme", scheme)
            ok_tuning.insert(1, "outer_fold", fold)
            ok_tuning.insert(2, "component", "OrdinaryKriging")
            tuning_records.append(ok_tuning)

            xgb_model = build_xgb_pipeline(best_xgb, random_seed + scheme_id * 30000 + fold)
            xgb_model.fit(train[features], train["TFe"])
            xgb_test = xgb_model.predict(test[features])

            vertical_scale = best_ok["vertical_scale"]
            n_neighbors = best_ok["n_neighbors"]
            train_coords = anisotropic_coordinates(train, vertical_scale)
            test_coords = anisotropic_coordinates(test, vertical_scale)
            grade_variogram = _fit_exponential_variogram(
                train_coords,
                train["TFe"].to_numpy(float),
                random_seed + scheme_id * 40000 + fold,
                int(kcfg["variogram_pair_sample"]),
                int(kcfg["variogram_bins"]),
            )
            ok_test, ok_variance = local_ordinary_kriging(
                train_coords,
                train["TFe"].to_numpy(float),
                test_coords,
                grade_variogram,
                n_neighbors,
            )
            residual_oof = train["TFe"].to_numpy(float) - xgb_inner_oof
            residual_variogram = _fit_exponential_variogram(
                train_coords,
                residual_oof,
                random_seed + scheme_id * 50000 + fold,
                int(kcfg["variogram_pair_sample"]),
                int(kcfg["variogram_bins"]),
            )
            residual_test, residual_variance = local_ordinary_kriging(
                train_coords,
                residual_oof,
                test_coords,
                residual_variogram,
                n_neighbors,
            )
            hybrid_test = xgb_test + residual_test
            xgb_weight = convex_weight(train["TFe"].to_numpy(float), xgb_inner_oof, ok_inner_oof)
            fusion_test = xgb_weight * xgb_test + (1.0 - xgb_weight) * ok_test
            mean_test = np.repeat(float(train["TFe"].mean()), len(test))

            high_threshold = float(train["TFe"].quantile(float(config["high_grade_quantile"])))
            nearest_distance = cKDTree(train_coords).query(test_coords, k=1)[0]
            base = test[
                [
                    "interval_id",
                    "hole",
                    "from",
                    "to",
                    "length",
                    "mid_depth",
                    "TFe",
                    "lithology_group",
                    "spatial_block",
                ]
            ].copy()
            base.insert(0, "scheme", scheme)
            base.insert(1, "outer_fold", fold)
            base["high_grade_threshold"] = high_threshold
            base["is_high_grade"] = base["TFe"] >= high_threshold
            base["nearest_training_distance"] = nearest_distance
            model_predictions = {
                "Mean": mean_test,
                "OrdinaryKriging": ok_test,
                "XGBoost": xgb_test,
                "OOFResidualKriging": hybrid_test,
                "ConvexFusion": fusion_test,
            }
            for model_name, model_pred in model_predictions.items():
                block = base.copy()
                block["model"] = model_name
                block["prediction"] = model_pred
                block["residual"] = model_pred - block["TFe"]
                block["kriging_variance"] = ok_variance if model_name in {"OrdinaryKriging", "ConvexFusion"} else np.nan
                block["residual_kriging_variance"] = residual_variance if model_name == "OOFResidualKriging" else np.nan
                predictions.append(block)
                metric = prediction_metrics(block["TFe"], block["prediction"], block["length"])
                high = block[block["is_high_grade"]]
                high_metric = prediction_metrics(high["TFe"], high["prediction"], high["length"]) if len(high) else {}
                fold_metrics.append(
                    {
                        "scheme": scheme,
                        "outer_fold": fold,
                        "model": model_name,
                        "n_train": len(train),
                        "n_test": len(test),
                        "n_train_holes": train["hole"].nunique(),
                        "n_test_holes": test["hole"].nunique(),
                        "test_TFe_mean": float(test["TFe"].mean()),
                        "test_TFe_sd": float(test["TFe"].std(ddof=1)),
                        "high_grade_threshold": high_threshold,
                        "n_high_grade_test": len(high),
                        **metric,
                        "high_grade_rmse": high_metric.get("rmse", np.nan),
                        "high_grade_mae": high_metric.get("mae", np.nan),
                        "high_grade_bias": high_metric.get("bias", np.nan),
                    }
                )
            variogram_records.extend(
                [
                    {
                        "scheme": scheme,
                        "outer_fold": fold,
                        "component": "grade",
                        "vertical_scale": vertical_scale,
                        "n_neighbors": n_neighbors,
                        **grade_variogram,
                    },
                    {
                        "scheme": scheme,
                        "outer_fold": fold,
                        "component": "OOF_residual",
                        "vertical_scale": vertical_scale,
                        "n_neighbors": n_neighbors,
                        **residual_variogram,
                    },
                ]
            )
            fusion_records.append(
                {
                    "scheme": scheme,
                    "outer_fold": fold,
                    "xgboost_weight": xgb_weight,
                    "ordinary_kriging_weight": 1.0 - xgb_weight,
                    "inner_xgboost_rmse": float(mean_squared_error(train["TFe"], xgb_inner_oof) ** 0.5),
                    "inner_ordinary_kriging_rmse": float(mean_squared_error(train["TFe"], ok_inner_oof) ** 0.5),
                }
            )
            timing_records.append(
                {
                    "scheme": scheme,
                    "outer_fold": fold,
                    "elapsed_seconds": time.perf_counter() - start_time,
                }
            )
            print(
                f"completed scheme={scheme} fold={fold}/{len(outer_splits)} "
                f"test_n={len(test)} xgb_weight={xgb_weight:.3f}",
                flush=True,
            )

    prediction_table = pd.concat(predictions, ignore_index=True)
    fold_metric_table = pd.DataFrame(fold_metrics)
    pooled_rows = []
    for (scheme, model), group in prediction_table.groupby(["scheme", "model"], sort=False):
        metric = prediction_metrics(group["TFe"], group["prediction"], group["length"])
        high = group[group["is_high_grade"]]
        high_metric = prediction_metrics(high["TFe"], high["prediction"], high["length"]) if len(high) else {}
        pooled_rows.append(
            {
                "scheme": scheme,
                "model": model,
                "n": len(group),
                "n_high_grade": len(high),
                **metric,
                "high_grade_rmse": high_metric.get("rmse", np.nan),
                "high_grade_mae": high_metric.get("mae", np.nan),
                "high_grade_bias": high_metric.get("bias", np.nan),
            }
        )
    pooled_metric_table = pd.DataFrame(pooled_rows)
    hole_metric_rows = []
    for (scheme, model, hole), group in prediction_table.groupby(["scheme", "model", "hole"]):
        hole_metric_rows.append(
            {"scheme": scheme, "model": model, "hole": hole, "n": len(group), **prediction_metrics(group["TFe"], group["prediction"], group["length"])}
        )
    hole_metric_table = pd.DataFrame(hole_metric_rows)

    distance_table = prediction_table.copy()
    distance_table["distance_quartile"] = distance_table.groupby(["scheme", "model"])["nearest_training_distance"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 4, labels=["Q1_near", "Q2", "Q3", "Q4_far"])
    )
    distance_rows = []
    for (scheme, model, quartile), group in distance_table.groupby(["scheme", "model", "distance_quartile"], observed=True):
        distance_rows.append(
            {
                "scheme": scheme,
                "model": model,
                "distance_quartile": quartile,
                "n": len(group),
                "distance_min": group["nearest_training_distance"].min(),
                "distance_median": group["nearest_training_distance"].median(),
                "distance_max": group["nearest_training_distance"].max(),
                **prediction_metrics(group["TFe"], group["prediction"], group["length"]),
            }
        )

    return {
        "oof_predictions": prediction_table,
        "fold_metrics": fold_metric_table,
        "pooled_metrics": pooled_metric_table,
        "hole_metrics": hole_metric_table,
        "distance_metrics": pd.DataFrame(distance_rows),
        "tuning_records": pd.concat(tuning_records, ignore_index=True),
        "variogram_parameters": pd.DataFrame(variogram_records),
        "fusion_weights": pd.DataFrame(fusion_records),
        "timings": pd.DataFrame(timing_records),
    }


def write_source_manifest(data_dir: Path, output_path: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(data_dir.glob("*")):
        if path.is_file():
            rows.append(
                {
                    "file": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_path, index=False)
    return manifest


def audit_legacy_processed_data(data_dir: Path, clean_intervals: pd.DataFrame) -> pd.DataFrame:
    integrated_path = data_dir / "整合钻孔品位模型表_审计用.xlsx"
    processed_path = data_dir / "Processed_Point_Data_审计用.xlsx"
    if not integrated_path.exists() or not processed_path.exists():
        return pd.DataFrame(columns=["item", "value"])
    integrated = pd.read_excel(integrated_path)
    processed = pd.read_excel(processed_path)
    integrated["工程号"] = _normalise_hole(integrated["工程号"])
    processed["工程号"] = _normalise_hole(processed["工程号"])
    integrated_grade = pd.to_numeric(integrated["TFe"], errors="coerce")
    processed_grade = pd.to_numeric(processed["TFe"], errors="coerce")
    fill_value = float(integrated_grade.mean())
    filled = np.isclose(processed_grade.to_numpy(float), fill_value, rtol=0.0, atol=1e-10)
    extra_holes = sorted(set(processed["工程号"]) - set(clean_intervals["hole"]))
    rows = [
        ("original_assay_intervals", len(clean_intervals)),
        ("original_assayed_holes", clean_intervals["hole"].nunique()),
        ("legacy_discretized_assayed_rows", len(integrated)),
        ("legacy_processed_rows", len(processed)),
        ("legacy_processed_holes", processed["工程号"].nunique()),
        ("global_mean_fill_value", fill_value),
        ("global_mean_filled_rows", int(filled.sum())),
        ("global_mean_filled_percent", float(filled.mean() * 100.0)),
        ("rows_corresponding_to_discretized_assays", int((~filled).sum())),
        ("unassayed_extra_hole_count", len(extra_holes)),
        ("unassayed_extra_holes", ",".join(extra_holes)),
        ("rows_in_unassayed_extra_holes", int(processed["工程号"].isin(extra_holes).sum())),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


def run_feature_ablation(
    data: pd.DataFrame,
    config: dict,
    reference_predictions: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    feature_sets = {
        "coordinates_only": (["x", "y", "z"], []),
        "trajectory_geometry": (NUMERIC_FEATURES, []),
        "full_geology": (NUMERIC_FEATURES, CATEGORICAL_FEATURES),
    }
    prediction_rows = []
    fold_rows = []
    seed = int(config["random_seed"])
    for scheme_id, scheme in enumerate(["drillhole", "spatial_block"]):
        outer = make_outer_splits(data, scheme, int(config["outer_folds"]), seed)
        for fold, (train_idx, test_idx) in enumerate(outer, start=1):
            train = data.iloc[train_idx].reset_index(drop=True)
            test = data.iloc[test_idx].reset_index(drop=True)
            inner = make_inner_splits(train, scheme, int(config["inner_folds"]), seed + 70000 + fold)
            for set_id, (set_name, (numeric_features, categorical_features)) in enumerate(feature_sets.items()):
                if set_name == "full_geology" and reference_predictions is not None:
                    reference = reference_predictions[
                        (reference_predictions["scheme"] == scheme)
                        & (reference_predictions["outer_fold"] == fold)
                        & (reference_predictions["model"] == "XGBoost")
                    ].set_index("interval_id")
                    pred = reference.loc[test["interval_id"], "prediction"].to_numpy(float)
                else:
                    best, _, _ = tune_xgb(
                        train,
                        inner,
                        config["xgboost_grid"],
                        seed + scheme_id * 10000 + fold * 101 + set_id * 1000,
                        numeric_features=numeric_features,
                        categorical_features=categorical_features,
                    )
                    features = list(numeric_features) + list(categorical_features)
                    model = build_xgb_pipeline(
                        best,
                        seed + scheme_id * 20000 + fold * 211 + set_id,
                        numeric_features=numeric_features,
                        categorical_features=categorical_features,
                    )
                    model.fit(train[features], train["TFe"])
                    pred = model.predict(test[features])
                metric = prediction_metrics(test["TFe"], pred, test["length"])
                fold_rows.append(
                    {
                        "scheme": scheme,
                        "outer_fold": fold,
                        "feature_set": set_name,
                        "n_train": len(train),
                        "n_test": len(test),
                        **metric,
                    }
                )
                block = test[["interval_id", "hole", "TFe", "length"]].copy()
                block.insert(0, "scheme", scheme)
                block.insert(1, "outer_fold", fold)
                block["feature_set"] = set_name
                block["prediction"] = pred
                prediction_rows.append(block)
    prediction_table = pd.concat(prediction_rows, ignore_index=True)
    pooled_rows = []
    for (scheme, feature_set), group in prediction_table.groupby(["scheme", "feature_set"]):
        pooled_rows.append(
            {
                "scheme": scheme,
                "feature_set": feature_set,
                "n": len(group),
                **prediction_metrics(group["TFe"], group["prediction"], group["length"]),
            }
        )
    return {
        "feature_ablation_predictions": prediction_table,
        "feature_ablation_fold_metrics": pd.DataFrame(fold_rows),
        "feature_ablation_pooled_metrics": pd.DataFrame(pooled_rows),
    }


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
