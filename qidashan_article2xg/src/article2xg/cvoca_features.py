from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


CVOCA_PREFIX = "cvoca_"


@dataclass(frozen=True)
class CvocaTransform:
    channels: tuple[str, ...]
    centre: np.ndarray
    scale: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    wavelengths_m: tuple[float, ...]


def fit_cvoca_transform(
    train: pd.DataFrame,
    channels: list[str],
    wavelengths_m: tuple[float, ...] = (15.0, 35.0, 75.0),
) -> CvocaTransform:
    matrix = train[channels].to_numpy(float)
    finite_matrix = np.where(np.isfinite(matrix), matrix, np.nan)
    lower = np.nanquantile(finite_matrix, 0.001, axis=0)
    upper = np.nanquantile(finite_matrix, 0.999, axis=0)
    raw_min = np.nanmin(finite_matrix, axis=0)
    raw_max = np.nanmax(finite_matrix, axis=0)
    lower = np.where(np.isfinite(lower), lower, raw_min)
    upper = np.where(np.isfinite(upper), upper, raw_max)
    invalid_bounds = ~np.isfinite(lower) | ~np.isfinite(upper) | (upper < lower)
    lower = np.where(invalid_bounds, 0.0, lower)
    upper = np.where(invalid_bounds, 0.0, upper)
    clipped = np.clip(finite_matrix, lower, upper)
    centre = np.nanmedian(clipped, axis=0)
    filled = np.where(np.isfinite(clipped), clipped, centre)
    q25 = np.nanquantile(filled, 0.25, axis=0)
    q75 = np.nanquantile(filled, 0.75, axis=0)
    robust_scale = (q75 - q25) / 1.349
    standard_scale = np.nanstd(filled, axis=0, ddof=1)
    scale = np.where(
        np.isfinite(robust_scale) & (robust_scale > 1e-9),
        robust_scale,
        standard_scale,
    )
    scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)
    return CvocaTransform(
        tuple(channels), centre, scale, lower, upper, tuple(wavelengths_m),
    )


def _complex_features_for_hole(
    depth: np.ndarray,
    values: np.ndarray,
    wavelengths_m: tuple[float, ...],
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    lag = depth[None, :] - depth[:, None]
    for wavelength in wavelengths_m:
        sigma = wavelength / 2.0
        window = np.abs(lag) <= 1.5 * wavelength
        gaussian = np.exp(-0.5 * (lag / max(sigma, 1e-9)) ** 2) * window
        phase = 2.0 * np.pi * lag / wavelength
        kernel = gaussian * np.exp(-1j * phase)
        denominator = np.maximum(np.sum(gaussian, axis=1), 1e-12)
        # Explicit weighted sums avoid spurious floating-point exceptions from
        # Apple's complex BLAS matmul while keeping the same convolution.
        response = np.sum(kernel[:, :, None] * values[None, :, :], axis=1)
        response = response / denominator[:, None]
        local_mean = np.sum(gaussian[:, :, None] * values[None, :, :], axis=1)
        local_mean = local_mean / denominator[:, None]
        response = response - local_mean * (np.sum(kernel, axis=1) / denominator)[:, None]
        tag = int(round(wavelength))
        for channel_index in range(values.shape[1]):
            complex_value = response[:, channel_index]
            amplitude = np.abs(complex_value)
            angle = np.angle(complex_value)
            prefix = f"{CVOCA_PREFIX}{channel_index:02d}_w{tag:03d}"
            result[f"{prefix}_amplitude"] = amplitude
            result[f"{prefix}_phase_sin"] = np.sin(angle)
            result[f"{prefix}_phase_cos"] = np.cos(angle)
        result[f"{CVOCA_PREFIX}w{tag:03d}_effective_support"] = np.sum(gaussian, axis=1)
    return result


def transform_cvoca(frame: pd.DataFrame, transform: CvocaTransform) -> pd.DataFrame:
    matrix = frame[list(transform.channels)].to_numpy(float)
    finite = np.where(np.isfinite(matrix), matrix, transform.centre)
    filled = np.clip(finite, transform.lower, transform.upper)
    standardised = (filled - transform.centre) / transform.scale
    standardised = np.clip(standardised, -25.0, 25.0)
    frames: list[pd.DataFrame] = []
    channel_tokens = [
        "".join(character if character.isalnum() else "_" for character in name).strip("_")
        for name in transform.channels
    ]
    for _, indices in frame.groupby("hole", sort=False).groups.items():
        index_array = np.asarray(list(indices), dtype=int)
        order = np.argsort(frame.loc[index_array, "mid_depth"].to_numpy(float))
        sorted_index = index_array[order]
        depth = frame.loc[sorted_index, "mid_depth"].to_numpy(float)
        positions = frame.index.get_indexer(sorted_index)
        values = standardised[positions]
        features = _complex_features_for_hole(depth, values, transform.wavelengths_m)
        local = pd.DataFrame(features, index=sorted_index)
        rename = {}
        for name in local.columns:
            if name.startswith(f"{CVOCA_PREFIX}w"):
                continue
            channel_index = int(name[len(CVOCA_PREFIX) : len(CVOCA_PREFIX) + 2])
            rename[name] = name.replace(
                f"{CVOCA_PREFIX}{channel_index:02d}",
                f"{CVOCA_PREFIX}{channel_tokens[channel_index]}",
                1,
            )
        frames.append(local.rename(columns=rename))
    output = pd.concat(frames, axis=0).reindex(frame.index)
    if output.columns.duplicated().any():
        output = output.loc[:, ~output.columns.duplicated()]
    if not np.isfinite(output.to_numpy(float)).all():
        raise ValueError("non-finite TP-CVOCA-style feature")
    return output
