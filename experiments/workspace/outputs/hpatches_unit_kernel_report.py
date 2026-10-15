"""Predeclared sequence-cluster summaries for the two unit-kernel tasks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from hpatches_geometry_make_report import estimate
from hpatches_matched_anchor_report import by_sequence, paired


LAWS = ("AA stretch", "unit-kernel consistent")
PATHS = ("direct low", "down then up")
METRICS = (
    "success_2_5_output_px",
    "success_5px",
    "matches",
    "enough_matches",
)
PRIMARY = {"direct low": "success_2_5_output_px", "down then up": "success_5px"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(results: Path) -> dict:
    protocol = json.loads((results / "protocol.json").read_text())
    tasks = read_csv(results / "task_results.csv")
    images = read_csv(results / "image_results.csv")
    for row in tasks:
        row["enough_matches"] = "1" if int(row["matches"]) >= 8 else "0"
    summary = {"protocol": protocol, "bases": {}}
    for base in protocol["bases"]:
        base_summary = {"original": {}, "factors": {}}
        for metric in METRICS:
            base_summary["original"][metric] = estimate(
                by_sequence(tasks, {"base": base, "factor": 1}, metric)
            )
        for factor in protocol["factors"]:
            factor_summary = {"paths": {}, "image_roundtrip": {}}
            for path in PATHS:
                path_summary = {"laws": {}, "difference": {}}
                for law in LAWS:
                    conditions = {
                        "base": base, "factor": factor,
                        "law": law, "path": path,
                    }
                    path_summary["laws"][law] = {
                        metric: estimate(by_sequence(tasks, conditions, metric))
                        for metric in METRICS
                    }
                path_summary["difference"] = {
                    metric: paired(
                        tasks,
                        {"base": base, "factor": factor,
                         "law": "unit-kernel consistent", "path": path},
                        {"base": base, "factor": factor,
                         "law": "AA stretch", "path": path},
                        metric,
                    )
                    for metric in METRICS
                }
                factor_summary["paths"][path] = path_summary
            factor_summary["image_roundtrip"] = {
                law: estimate(
                    by_sequence(
                        images,
                        {"base": base, "factor": factor, "law": law},
                        "roundtrip_nrmse_percent",
                    )
                )
                for law in LAWS
            }
            base_summary["factors"][str(factor)] = factor_summary
        summary["bases"][base] = base_summary
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.results)
    (args.results / "summary.json").write_text(json.dumps(summary, indent=2))
    for path in PATHS:
        print(f"\n{path}: {PRIMARY[path]}")
        for base in summary["protocol"]["bases"]:
            print(base)
            for factor in summary["protocol"]["factors"]:
                entry = summary["bases"][base]["factors"][str(factor)]["paths"][path]
                metric = PRIMARY[path]
                ours = entry["laws"]["unit-kernel consistent"][metric]["mean"]
                aa = entry["laws"]["AA stretch"][metric]["mean"]
                diff = entry["difference"][metric]
                print(
                    f"  {factor:2}x ours {100*ours:5.1f}% AA {100*aa:5.1f}% "
                    f"delta {100*diff['mean']:+5.1f} pp "
                    f"CI [{100*diff['low']:+.1f},{100*diff['high']:+.1f}]"
                )


if __name__ == "__main__":
    main()
