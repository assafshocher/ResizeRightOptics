"""Sequence-cluster bootstrap summary for the direct-only matched audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from hpatches_geometry_make_report import estimate
from hpatches_matched_anchor_report import by_sequence, paired


LAWS = ("AA stretch", "consistent telescope")
METRICS = ("success_2_5_output_px", "matches", "enough_matches")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(results: Path) -> dict:
    protocol = json.loads((results / "protocol.json").read_text())
    tasks = read_csv(results / "task_results.csv")
    anchors = read_csv(results / "anchor_results.csv")
    for row in tasks:
        row["enough_matches"] = "1" if float(row["matches"]) >= 8 else "0"
    summary = {"protocol": protocol, "bases": {}}
    for base in protocol["bases"]:
        base_summary = {"anchor": {}, "factors": {}}
        for metric in METRICS:
            base_summary["anchor"][metric] = estimate(
                by_sequence(tasks, {"base": base, "factor": 2}, metric)
            )
        base_summary["anchor"]["image_nrmse_percent"] = estimate(
            by_sequence(
                anchors, {"base": base}, "anchor_nrmse_percent"
            )
        )
        for factor in protocol["factors"]:
            factor_summary = {"laws": {}, "differences": {}}
            for law in LAWS:
                factor_summary["laws"][law] = {
                    metric: estimate(
                        by_sequence(
                            tasks,
                            {"base": base, "factor": factor, "law": law},
                            metric,
                        )
                    )
                    for metric in METRICS
                }
            factor_summary["differences"]["consistent_minus_AA"] = {
                metric: paired(
                    tasks,
                    {"base": base, "factor": factor, "law": "consistent telescope"},
                    {"base": base, "factor": factor, "law": "AA stretch"},
                    metric,
                )
                for metric in METRICS
            }
            base_summary["factors"][str(factor)] = factor_summary
        summary["bases"][base] = base_summary
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.results)
    (args.results / "direct_summary.json").write_text(json.dumps(summary, indent=2))
    for base, base_summary in summary["bases"].items():
        for factor, factor_summary in base_summary["factors"].items():
            ours = factor_summary["laws"]["consistent telescope"]["success_2_5_output_px"]["mean"]
            aa = factor_summary["laws"]["AA stretch"]["success_2_5_output_px"]["mean"]
            diff = factor_summary["differences"]["consistent_minus_AA"]["success_2_5_output_px"]
            print(
                f"{base:35} x{factor:2}: ours={100*ours:5.1f}% AA={100*aa:5.1f}% "
                f"delta={100*diff['mean']:+5.1f} pp "
                f"CI [{100*diff['low']:+.1f},{100*diff['high']:+.1f}]"
            )


if __name__ == "__main__":
    main()
