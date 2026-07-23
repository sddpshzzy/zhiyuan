from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def normalise_hole(value: str) -> str:
    return (
        str(value)
        .strip()
        .upper()
        .translate(str.maketrans({"（": "(", "）": ")", " ": ""}))
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Data-firewall helper: emit only an authorised assay subset."
    )
    parser.add_argument("--mdb", type=Path, required=True)
    parser.add_argument("--holes-json", type=Path)
    parser.add_argument("--support-only", action="store_true")
    args = parser.parse_args()

    authorised: set[str] | None = None
    if args.holes_json is not None:
        authorised = {
            normalise_hole(value)
            for value in json.loads(args.holes_json.read_text(encoding="utf-8"))
        }
    if not args.support_only and authorised is None:
        raise SystemExit("--holes-json is required unless --support-only is used")

    process = subprocess.Popen(
        ["mdb-export", str(args.mdb), "化验表"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8-sig",
    )
    assert process.stdout is not None
    reader = csv.reader(process.stdout)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    header = next(reader)
    if len(header) < 4 or normalise_hole(header[0]) != normalise_hole("工程号"):
        process.kill()
        raise RuntimeError("unexpected assay table schema")
    if args.support_only:
        writer.writerow(header[:4])
        for row in reader:
            writer.writerow(row[:4])
    else:
        writer.writerow(header)
        for row in reader:
            if row and normalise_hole(row[0]) in authorised:
                writer.writerow(row)
    stderr = process.stderr.read() if process.stderr is not None else ""
    return_code = process.wait()
    if return_code:
        raise RuntimeError(stderr.strip() or f"mdb-export failed with {return_code}")


if __name__ == "__main__":
    main()
