"""Combine disjoint base-family runs without changing their paired estimates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries = [json.loads(path.read_text()) for path in args.summaries]
    reference = summaries[0]["protocol"]
    for summary in summaries[1:]:
        protocol = summary["protocol"]
        for field in ("dataset", "sequences", "factors", "anchor_factor"):
            if protocol[field] != reference[field]:
                raise ValueError(f"cannot combine mismatched field {field}")
    merged = {"protocol": dict(reference), "bases": {}}
    merged["protocol"]["bases"] = []
    for summary in summaries:
        for base, result in summary["bases"].items():
            if base in merged["bases"]:
                raise ValueError(f"duplicate base family {base}")
            merged["protocol"]["bases"].append(base)
            merged["bases"][base] = result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2))


if __name__ == "__main__":
    main()
