from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.qidashan_strict import sha256_file
from src.sijiaying_external import build_blind_partition, load_partition_inputs


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config"
OUTPUT = ROOT / "results" / "sijiaying_blind_external"


def main() -> None:
    contract = json.loads(
        (CONFIG / "sijiaying_source_selection_freeze_20260723.json").read_text(
            encoding="utf-8"
        )
    )
    mdb_path = Path(contract["selected_source"]["path"])
    helper_path = ROOT / "src" / "mdb_assay_subset.py"
    blind_block = int(
        contract["blind_partition_freeze"][
            "selected_blind_block_index_zero_based"
        ]
    )
    if sha256_file(mdb_path) != contract["selected_source"]["sha256"]:
        raise RuntimeError("selected Sijiaying source hash changed")
    collar, survey, support = load_partition_inputs(mdb_path, helper_path)
    holes, audit = build_blind_partition(
        collar, survey, support, blind_block=blind_block, n_blocks=5
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    partition_path = OUTPUT / "sijiaying_frozen_hole_partition.csv"
    holes.to_csv(partition_path, index=False)
    audit.to_csv(OUTPUT / "sijiaying_partition_audit.csv", index=False)
    development_holes = holes.loc[
        holes["partition_role"].eq("DEVELOPMENT"), "hole"
    ].tolist()
    blind_holes = holes.loc[holes["partition_role"].eq("BLIND"), "hole"].tolist()
    (OUTPUT / "development_holes.json").write_text(
        json.dumps(development_holes, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT / "blind_holes_DO_NOT_OPEN_BEFORE_RELEASE.json").write_text(
        json.dumps(blind_holes, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    code_paths = [
        ROOT / "src" / "mdb_assay_subset.py",
        ROOT / "src" / "sijiaying_external.py",
        ROOT / "prepare_sijiaying_blind_partition.py",
        ROOT / "run_sijiaying_blind_external.py",
    ]
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "PARTITION_AND_CODE_FROZEN_TARGET_HIDDEN",
        "source_path": str(mdb_path),
        "source_sha256": sha256_file(mdb_path),
        "blind_block": blind_block,
        "development_hole_count": len(development_holes),
        "blind_hole_count": len(blind_holes),
        "partition_sha256": sha256_file(partition_path),
        "code_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in code_paths
        },
        "target_fields_used_for_partition": [],
        "entity_route": contract["entity_route"],
    }
    (OUTPUT / "partition_and_code_freeze_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
