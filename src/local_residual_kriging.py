from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Variogram:
    model: str
    nugget: float
    partial_sill: float
    range_m: float
    z_scale: float
    pair_count: int
    bin_count: int

    def semivariance(self, distance: np.ndarray) -> np.ndarray:
        distance = np.asarray(distance, dtype=float)
        ratio = distance / max(self.range_m, 1e-9)
        if self.model == "exponential":
            structured = 1.0 - np.exp(-3.0 * ratio)
        elif self.model == "spherical":
            structured = np.where(ratio < 1.0, 1.5 * ratio - 0.5 * ratio**3, 1.0)
        else:
            raise ValueError(self.model)
        result = self.nugget + self.partial_sill * structured
        return np.where(distance <= 1e-12, 0.0, result)

    def covariance(self, distance: np.ndarray) -> np.ndarray:
        distance = np.asarray(distance, dtype=float)
        if self.model == "exponential":
            structured = self.partial_sill * np.exp(-3.0 * distance / max(self.range_m, 1e-9))
        elif self.model == "spherical":
            ratio = distance / max(self.range_m, 1e-9)
            structured = self.partial_sill * np.where(
                ratio < 1.0, 1.0 - 1.5 * ratio + 0.5 * ratio**3, 0.0
            )
        else:
            raise ValueError(self.model)
        return structured + np.where(distance <= 1e-12, self.nugget, 0.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "nugget": self.nugget,
            "partial_sill": self.partial_sill,
            "total_sill": self.nugget + self.partial_sill,
            "range_m_scaled": self.range_m,
            "z_scale": self.z_scale,
            "pair_count": self.pair_count,
            "bin_count": self.bin_count,
        }


def scaled_coordinates(frame: pd.DataFrame, z_scale: float) -> np.ndarray:
    coordinates = frame[["x", "y", "z"]].to_numpy(float).copy()
    coordinates[:, 2] *= z_scale
    return coordinates


def fit_variogram(
    frame: pd.DataFrame,
    residuals: np.ndarray,
    *,
    z_scale: float = 0.25,
    model: str = "exponential",
    maximum_pairs: int = 100_000,
    bins: int = 14,
    seed: int = 20260723,
) -> tuple[Variogram, pd.DataFrame]:
    coordinates = scaled_coordinates(frame, z_scale)
    residuals = np.asarray(residuals, dtype=float)
    if len(coordinates) != len(residuals) or len(residuals) < 30:
        raise ValueError("variogram requires at least 30 aligned residual observations")
    rng = np.random.default_rng(seed)
    pair_count = min(maximum_pairs, len(residuals) * 30)
    left = rng.integers(0, len(residuals), size=pair_count)
    right = rng.integers(0, len(residuals), size=pair_count)
    different = left != right
    left, right = left[different], right[different]
    distance = np.linalg.norm(coordinates[left] - coordinates[right], axis=1)
    semivariance = 0.5 * (residuals[left] - residuals[right]) ** 2
    finite = np.isfinite(distance) & np.isfinite(semivariance) & (distance > 0)
    distance, semivariance = distance[finite], semivariance[finite]
    cutoff = float(np.quantile(distance, 0.90))
    keep = distance <= cutoff
    distance, semivariance = distance[keep], semivariance[keep]
    edges = np.linspace(0.0, cutoff, bins + 1)
    bin_index = np.clip(np.digitize(distance, edges) - 1, 0, bins - 1)
    rows: list[dict[str, float | int]] = []
    for index in range(bins):
        mask = bin_index == index
        if mask.sum() < 30:
            continue
        rows.append({
            "bin": index,
            "distance_median": float(np.median(distance[mask])),
            "semivariance_median": float(np.median(semivariance[mask])),
            "semivariance_mean": float(np.mean(semivariance[mask])),
            "pair_count": int(mask.sum()),
        })
    empirical = pd.DataFrame(rows)
    x = empirical["distance_median"].to_numpy(float)
    y = empirical["semivariance_median"].to_numpy(float)
    weights = np.sqrt(empirical["pair_count"].to_numpy(float))
    residual_variance = max(float(np.var(residuals, ddof=1)), 1e-6)

    def model_values(parameters: np.ndarray) -> np.ndarray:
        nugget, partial_sill, range_m = parameters
        ratio = x / max(range_m, 1e-9)
        if model == "exponential":
            structured = 1.0 - np.exp(-3.0 * ratio)
        elif model == "spherical":
            structured = np.where(ratio < 1.0, 1.5 * ratio - 0.5 * ratio**3, 1.0)
        else:
            raise ValueError(model)
        return nugget + partial_sill * structured

    initial = np.asarray([
        residual_variance * 0.15,
        residual_variance * 0.85,
        max(float(np.median(x)), 1.0),
    ])
    lower = np.asarray([0.0, residual_variance * 1e-4, max(float(np.min(x)) * 0.25, 0.5)])
    upper = np.asarray([
        residual_variance * 3.0,
        residual_variance * 5.0,
        max(float(np.max(x)) * 3.0, 2.0),
    ])
    result = least_squares(
        lambda parameters: (model_values(parameters) - y) * weights / max(np.mean(weights), 1.0),
        x0=np.clip(initial, lower, upper),
        bounds=(lower, upper),
    )
    variogram = Variogram(
        model=model,
        nugget=float(result.x[0]),
        partial_sill=float(result.x[1]),
        range_m=float(result.x[2]),
        z_scale=z_scale,
        pair_count=len(distance),
        bin_count=len(empirical),
    )
    empirical["fitted_semivariance"] = variogram.semivariance(x)
    return variogram, empirical


def local_ordinary_kriging(
    source: pd.DataFrame,
    residuals: np.ndarray,
    query: pd.DataFrame,
    variogram: Variogram,
    *,
    neighbours: int = 24,
) -> tuple[np.ndarray, pd.DataFrame]:
    source_coordinates = scaled_coordinates(source, variogram.z_scale)
    query_coordinates = scaled_coordinates(query, variogram.z_scale)
    residuals = np.asarray(residuals, dtype=float)
    tree = cKDTree(source_coordinates)
    distances, indices = tree.query(query_coordinates, k=min(neighbours, len(source)))
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    predictions = np.empty(len(query), dtype=float)
    fallback = np.zeros(len(query), dtype=int)
    condition_numbers = np.empty(len(query), dtype=float)
    lower_clip, upper_clip = np.quantile(residuals, [0.01, 0.99])
    for row_index, (neighbour_index, query_distance) in enumerate(zip(indices, distances)):
        local_coordinates = source_coordinates[neighbour_index]
        pair_distance = np.linalg.norm(
            local_coordinates[:, None, :] - local_coordinates[None, :, :], axis=2
        )
        covariance = variogram.covariance(pair_distance)
        covariance.flat[:: len(covariance) + 1] += max(
            1e-8, 1e-8 * (variogram.nugget + variogram.partial_sill)
        )
        system = np.empty((len(neighbour_index) + 1, len(neighbour_index) + 1), dtype=float)
        system[:-1, :-1] = covariance
        system[:-1, -1] = 1.0
        system[-1, :-1] = 1.0
        system[-1, -1] = 0.0
        rhs = np.append(variogram.covariance(query_distance), 1.0)
        condition_numbers[row_index] = float(np.linalg.cond(system))
        try:
            weights = np.linalg.solve(system, rhs)[:-1]
            value = float(np.dot(weights, residuals[neighbour_index]))
            if not np.isfinite(value):
                raise np.linalg.LinAlgError("non-finite kriging prediction")
        except np.linalg.LinAlgError:
            fallback[row_index] = 1
            inverse = 1.0 / np.maximum(query_distance, 1e-6) ** 2
            value = float(np.dot(inverse, residuals[neighbour_index]) / inverse.sum())
        predictions[row_index] = float(np.clip(value, lower_clip, upper_clip))
    audit = pd.DataFrame({
        "nearest_residual_distance_m": distances[:, 0],
        "farthest_local_neighbour_distance_m": distances[:, -1],
        "kriging_system_condition_number": condition_numbers,
        "idw_fallback": fallback,
    })
    return predictions, audit
