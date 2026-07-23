from __future__ import annotations

import csv
import io
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .qidashan_strict import (
    HoleTrajectory,
    SPATIAL_FEATURES,
    _collar_connected_components,
    export_mdb_table,
)


LITHOLOGY_NUMERIC_FEATURES = [
    "lithology_unit_thickness_m",
    "lithology_relative_position",
    "lithology_nearest_boundary_distance_m",
    "lithology_known_indicator",
]
LITHOLOGY_CATEGORICAL_FEATURE = "lithology"


def normalise_sijiaying_hole(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .str.translate(str.maketrans({"（": "(", "）": ")", " ": ""}))
    )


def _read_csv_bytes(payload: bytes) -> pd.DataFrame:
    return pd.read_csv(
        io.StringIO(payload.decode("utf-8-sig")),
        quoting=csv.QUOTE_MINIMAL,
    )


def load_partition_inputs(
    mdb_path: Path, helper_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    collar = export_mdb_table(mdb_path, "定位表")
    survey = export_mdb_table(mdb_path, "测斜表")
    completed = subprocess.run(
        [
            sys.executable,
            str(helper_path),
            "--mdb",
            str(mdb_path),
            "--support-only",
        ],
        check=True,
        capture_output=True,
    )
    support = _read_csv_bytes(completed.stdout)
    collar.columns = [
        "hole",
        "collar_x",
        "collar_y",
        "collar_z",
        "max_depth",
        "trajectory_type",
    ]
    survey.columns = ["hole", "depth", "azimuth", "dip"]
    support.columns = ["hole", "from", "to", "length_reported"]
    for frame in (collar, survey, support):
        frame["hole"] = normalise_sijiaying_hole(frame["hole"])
    for column in ["collar_x", "collar_y", "collar_z", "max_depth"]:
        collar[column] = pd.to_numeric(collar[column], errors="coerce")
    for column in ["depth", "azimuth", "dip"]:
        survey[column] = pd.to_numeric(survey[column], errors="coerce")
    for column in ["from", "to", "length_reported"]:
        support[column] = pd.to_numeric(support[column], errors="coerce")
    return collar, survey, support


def build_blind_partition(
    collar: pd.DataFrame,
    survey: pd.DataFrame,
    support: pd.DataFrame,
    blind_block: int,
    n_blocks: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    support = support.dropna(subset=["hole", "from", "to"]).copy()
    support["length_computed"] = support["to"] - support["from"]
    assay_holes = sorted(set(support["hole"]))
    collar_valid = (
        collar.loc[collar["hole"].isin(assay_holes)]
        .dropna(subset=["collar_x", "collar_y", "collar_z", "max_depth"])
        .drop_duplicates("hole")
        .copy()
    )
    missing_collar = set(assay_holes) - set(collar_valid["hole"])
    valid_survey_holes = set(
        survey.dropna(subset=["depth", "azimuth", "dip"])["hole"]
    )
    missing_survey = set(assay_holes) - valid_survey_holes
    excluded_holes = missing_collar | missing_survey
    excluded_support_rows = int(support["hole"].isin(excluded_holes).sum())
    support = support.loc[~support["hole"].isin(excluded_holes)].copy()
    counts = support.groupby("hole").size().rename("assay_support_count")
    holes = collar_valid.merge(counts, on="hole", how="inner")
    xy = holes[["collar_x", "collar_y"]].to_numpy(float)
    holes["physical_hole_group"] = _collar_connected_components(xy, radius_m=5.0)
    groups = (
        holes.groupby("physical_hole_group", as_index=False)
        .agg(
            collar_x=("collar_x", "mean"),
            collar_y=("collar_y", "mean"),
            assay_support_count=("assay_support_count", "sum"),
        )
    )
    centred = groups[["collar_x", "collar_y"]].to_numpy(float)
    centred = centred - centred.mean(axis=0)
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    groups["principal_axis_score"] = centred @ vh[0]
    groups = groups.sort_values("principal_axis_score").reset_index(drop=True)
    midpoint_count = (
        groups["assay_support_count"].cumsum()
        - groups["assay_support_count"] / 2.0
    )
    groups["spatial_block"] = np.minimum(
        (
            midpoint_count
            / groups["assay_support_count"].sum()
            * n_blocks
        ).astype(int),
        n_blocks - 1,
    )
    holes = holes.merge(
        groups[["physical_hole_group", "principal_axis_score", "spatial_block"]],
        on="physical_hole_group",
        how="left",
    ).sort_values(["spatial_block", "principal_axis_score", "hole"])
    holes["partition_role"] = np.where(
        holes["spatial_block"].eq(blind_block), "BLIND", "DEVELOPMENT"
    )
    audit = pd.DataFrame(
        [
            {"check": "assay_support_rows", "value": int(len(support))},
            {"check": "assay_holes", "value": int(len(assay_holes))},
            {
                "check": "physical_hole_groups_5m",
                "value": int(holes["physical_hole_group"].nunique()),
            },
            {
                "check": "invalid_computed_lengths",
                "value": int((support["length_computed"] <= 0).sum()),
            },
            {
                "check": "excluded_assay_support_rows_missing_trajectory",
                "value": excluded_support_rows,
            },
            {
                "check": "excluded_holes_missing_trajectory",
                "value": "|".join(sorted(excluded_holes)),
            },
            {
                "check": "reported_vs_computed_length_max_abs_m",
                "value": float(
                    (support["length_reported"] - support["length_computed"])
                    .abs()
                    .max()
                ),
            },
            {
                "check": "blind_block",
                "value": int(blind_block),
            },
            {
                "check": "blind_holes",
                "value": int(holes["partition_role"].eq("BLIND").sum()),
            },
            {
                "check": "blind_assay_support_rows",
                "value": int(
                    holes.loc[
                        holes["partition_role"].eq("BLIND"), "assay_support_count"
                    ].sum()
                ),
            },
        ]
    )
    return holes.reset_index(drop=True), audit


def _load_assay_subset(
    mdb_path: Path, helper_path: Path, holes_json: Path
) -> pd.DataFrame:
    completed = subprocess.run(
        [
            sys.executable,
            str(helper_path),
            "--mdb",
            str(mdb_path),
            "--holes-json",
            str(holes_json),
        ],
        check=True,
        capture_output=True,
    )
    assay = _read_csv_bytes(completed.stdout)
    if assay.shape[1] != 8:
        raise ValueError(f"unexpected Sijiaying assay width: {assay.shape[1]}")
    assay.columns = [
        "hole",
        "from",
        "to",
        "length_reported",
        "TFe",
        "FeO",
        "SFe",
        "magnetic_rate",
    ]
    assay["hole"] = normalise_sijiaying_hole(assay["hole"])
    for column in [
        "from",
        "to",
        "length_reported",
        "TFe",
        "FeO",
        "SFe",
        "magnetic_rate",
    ]:
        assay[column] = pd.to_numeric(assay[column], errors="coerce")
    assay = assay.dropna(subset=["hole", "from", "to", "TFe"]).copy()
    assay["length"] = assay["to"] - assay["from"]
    return assay


def _load_lithology(mdb_path: Path, authorised_holes: set[str]) -> pd.DataFrame:
    lithology = export_mdb_table(mdb_path, "岩性表")
    lithology.columns = ["hole", "lith_from", "lith_to", "lithology"]
    lithology["hole"] = normalise_sijiaying_hole(lithology["hole"])
    lithology = lithology.loc[lithology["hole"].isin(authorised_holes)].copy()
    lithology["lith_from"] = pd.to_numeric(lithology["lith_from"], errors="coerce")
    lithology["lith_to"] = pd.to_numeric(lithology["lith_to"], errors="coerce")
    lithology["lithology"] = (
        lithology["lithology"].astype("string").fillna("UNKNOWN").str.strip()
    )
    return lithology.dropna(subset=["lith_from", "lith_to"])


def _align_lithology(
    intervals: pd.DataFrame, lithology: pd.DataFrame
) -> pd.DataFrame:
    output_frames: list[pd.DataFrame] = []
    for hole, frame in intervals.groupby("hole", sort=False):
        frame = frame.copy()
        logs = (
            lithology.loc[lithology["hole"].eq(hole)]
            .sort_values(["lith_from", "lith_to"])
            .reset_index(drop=True)
        )
        frame["lithology"] = "UNKNOWN"
        frame["lithology_unit_thickness_m"] = 0.0
        frame["lithology_relative_position"] = 0.5
        frame["lithology_nearest_boundary_distance_m"] = 0.0
        frame["lithology_known_indicator"] = 0
        if not logs.empty:
            starts = logs["lith_from"].to_numpy(float)
            ends = logs["lith_to"].to_numpy(float)
            mids = frame["mid_depth"].to_numpy(float)
            indices = np.searchsorted(starts, mids, side="right") - 1
            valid = (indices >= 0) & (indices < len(logs))
            clipped = np.clip(indices, 0, max(len(logs) - 1, 0))
            valid &= mids < ends[clipped] + 1e-9
            for row_position, log_position in zip(
                np.flatnonzero(valid), clipped[valid]
            ):
                start = float(starts[log_position])
                end = float(ends[log_position])
                thickness = max(end - start, 1e-9)
                midpoint = float(mids[row_position])
                frame.iloc[
                    row_position,
                    frame.columns.get_loc("lithology"),
                ] = str(logs.iloc[log_position]["lithology"])
                frame.iloc[
                    row_position,
                    frame.columns.get_loc("lithology_unit_thickness_m"),
                ] = thickness
                frame.iloc[
                    row_position,
                    frame.columns.get_loc("lithology_relative_position"),
                ] = np.clip((midpoint - start) / thickness, 0.0, 1.0)
                frame.iloc[
                    row_position,
                    frame.columns.get_loc(
                        "lithology_nearest_boundary_distance_m"
                    ),
                ] = min(midpoint - start, end - midpoint)
                frame.iloc[
                    row_position,
                    frame.columns.get_loc("lithology_known_indicator"),
                ] = 1
        output_frames.append(frame)
    return pd.concat(output_frames, ignore_index=True)


def load_authorised_intervals(
    mdb_path: Path,
    helper_path: Path,
    holes_json: Path,
    partition_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    authorised_holes = {
        str(value)
        .strip()
        .upper()
        .translate(str.maketrans({"（": "(", "）": ")", " ": ""}))
        for value in json.loads(holes_json.read_text(encoding="utf-8"))
    }
    collar = export_mdb_table(mdb_path, "定位表")
    survey = export_mdb_table(mdb_path, "测斜表")
    collar.columns = [
        "hole",
        "collar_x",
        "collar_y",
        "collar_z",
        "max_depth",
        "trajectory_type",
    ]
    survey.columns = ["hole", "depth", "azimuth", "dip"]
    for frame in (collar, survey):
        frame["hole"] = normalise_sijiaying_hole(frame["hole"])
    collar = collar.loc[collar["hole"].isin(authorised_holes)].copy()
    survey = survey.loc[survey["hole"].isin(authorised_holes)].copy()
    for column in ["collar_x", "collar_y", "collar_z", "max_depth"]:
        collar[column] = pd.to_numeric(collar[column], errors="coerce")
    for column in ["depth", "azimuth", "dip"]:
        survey[column] = pd.to_numeric(survey[column], errors="coerce")
    assay = _load_assay_subset(mdb_path, helper_path, holes_json)
    if set(assay["hole"]) - authorised_holes:
        raise RuntimeError("data firewall emitted an unauthorised hole")
    if assay.duplicated(["hole", "from", "to"]).any():
        raise ValueError("duplicate assay intervals detected")
    if (assay["length"] <= 0).any():
        raise ValueError("non-positive assay support detected")
    collar_one = collar.drop_duplicates("hole").set_index("hole")
    trajectories = {
        hole: HoleTrajectory.from_survey(survey.loc[survey["hole"].eq(hole)])
        for hole in sorted(set(assay["hole"]))
    }
    rows: list[dict[str, object]] = []
    for record in assay.to_dict(orient="records"):
        midpoint = (float(record["from"]) + float(record["to"])) / 2.0
        de, dn, dz, azimuth, dip = trajectories[str(record["hole"])].position(
            midpoint
        )
        collar_row = collar_one.loc[str(record["hole"])]
        row = dict(record)
        row.update(
            {
                "interval_id": (
                    f"{record['hole']}_{float(record['from']):.2f}_"
                    f"{float(record['to']):.2f}"
                ),
                "mid_depth": midpoint,
                "x": float(collar_row.collar_x) + de,
                "y": float(collar_row.collar_y) + dn,
                "z": float(collar_row.collar_z) + dz,
                "collar_x": float(collar_row.collar_x),
                "collar_y": float(collar_row.collar_y),
                "collar_z": float(collar_row.collar_z),
                "max_depth": float(collar_row.max_depth),
                "depth_ratio": midpoint / max(float(collar_row.max_depth), 1e-9),
                "azimuth_sin": math.sin(math.radians(azimuth)),
                "azimuth_cos": math.cos(math.radians(azimuth)),
                "dip_sin": math.sin(math.radians(dip)),
                "dip_cos": math.cos(math.radians(dip)),
            }
        )
        rows.append(row)
    intervals = pd.DataFrame(rows)
    lithology = _load_lithology(mdb_path, authorised_holes)
    intervals = _align_lithology(intervals, lithology)
    partition = pd.read_csv(partition_csv)
    partition["hole"] = normalise_sijiaying_hole(partition["hole"])
    intervals = intervals.merge(
        partition[
            [
                "hole",
                "physical_hole_group",
                "spatial_block",
                "partition_role",
            ]
        ],
        on="hole",
        how="left",
        validate="many_to_one",
    )
    if intervals["spatial_block"].isna().any():
        raise RuntimeError("authorised interval missing frozen partition")
    audit = pd.DataFrame(
        [
            {"check": "interval_count", "value": int(len(intervals))},
            {"check": "hole_count", "value": int(intervals["hole"].nunique())},
            {
                "check": "physical_hole_group_count",
                "value": int(intervals["physical_hole_group"].nunique()),
            },
            {
                "check": "lithology_known_rate",
                "value": float(intervals["lithology_known_indicator"].mean()),
            },
            {
                "check": "TFe_missing_count",
                "value": int(intervals["TFe"].isna().sum()),
            },
            {
                "check": "TFe_min",
                "value": float(intervals["TFe"].min()),
            },
            {
                "check": "TFe_max",
                "value": float(intervals["TFe"].max()),
            },
        ]
    )
    return intervals.sort_values(["hole", "from", "to"]).reset_index(drop=True), audit


def common_features() -> list[str]:
    return list(SPATIAL_FEATURES)
