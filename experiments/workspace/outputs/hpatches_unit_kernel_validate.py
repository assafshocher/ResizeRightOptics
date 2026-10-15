"""Check the new audit against unchanged old AA arms and invariants."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OLD_SOURCES = {
    8: HERE / "hpatches_matched_anchor_full" / "task_results.csv",
    12: HERE / "hpatches_matched_anchor_full" / "task_results.csv",
    16: HERE / "hpatches_matched_anchor_full" / "task_results.csv",
    20: HERE / "hpatches_matched_anchor_direct_x20_full" / "task_results.csv",
    24: HERE / "hpatches_matched_anchor_x24_full" / "task_results.csv",
    32: HERE / "hpatches_matched_anchor_x32_full" / "task_results.csv",
}
BASES = ("linear tent", "Keys cubic a=-0.5", "Lanczos-3")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def task_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return row["sequence"], row["target"], row["base"], row["factor"]


def score_signature(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        row["matches"], row["success_2_5_output_px"], row["success_5px"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs=3, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    new = {base: rows(directory / "task_results.csv") for base, directory in zip(BASES, args.runs)}
    report: dict = {"original_baseline": {}, "AA_vs_old": {}, "consistent_down_up": {}}

    baseline = {
        base: {
            (row["sequence"], row["target"]): score_signature(row)
            for row in entries if row["factor"] == "1"
        }
        for base, entries in new.items()
    }
    reference = baseline[BASES[0]]
    report["original_baseline"] = {
        "pairs": len(reference),
        "cross_base_mismatches": {
            base: sum(reference.get(key) != value for key, value in scores.items())
            for base, scores in baseline.items()
        },
    }

    for factor, old_path in OLD_SOURCES.items():
        old_rows = rows(old_path)
        for base in BASES[:2]:
            archived = {
                task_key(row): score_signature(row)
                for row in old_rows
                if row["factor"] == str(factor)
                and row["base"] == base
                and row["law"] == "AA stretch"
                and row["path"] == "full direct"
            }
            current = {
                task_key(row): score_signature(row)
                for row in new[base]
                if row["factor"] == str(factor)
                and row["law"] == "AA stretch"
                and row["path"] == "direct low"
            }
            report["AA_vs_old"][f"{base}:x{factor}"] = {
                "pairs": len(current),
                "missing_pairs": len(set(current) ^ set(archived)),
                "score_mismatches": sum(
                    current[key] != archived[key]
                    for key in current.keys() & archived.keys()
                ),
                "examples": [
                    {"key": key, "new": current[key], "old": archived[key]}
                    for key in sorted(current.keys() & archived.keys())
                    if current[key] != archived[key]
                ][:5],
            }

    roundtrip = {
        base: {
            (row["sequence"], row["target"], row["factor"]): (
                row["matches"], row["success_5px"]
            )
            for row in entries
            if row["law"] == "unit-kernel consistent"
            and row["path"] == "down then up"
        }
        for base, entries in new.items()
    }
    reference = roundtrip[BASES[0]]
    report["consistent_down_up"] = {
        "pairs": len(reference),
        "cross_base_score_mismatches": {
            base: sum(reference.get(key) != value for key, value in scores.items())
            for base, scores in roundtrip.items()
        },
    }
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
