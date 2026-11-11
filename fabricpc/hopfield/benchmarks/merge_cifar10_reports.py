"""Merge per-seed CIFAR-10 benchmark reports into one mean ± std summary.

    python -m fabricpc.hopfield.benchmarks.merge_cifar10_reports \
        reports/run_seed0 reports/run_seed1 --out reports/merged
"""

import argparse
import json
from pathlib import Path

from fabricpc.hopfield.benchmarks.cifar10_pc_prototype_retrieval import write_reports


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Merge CIFAR-10 benchmark reports.")
    parser.add_argument("report_dirs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    payloads = [
        json.loads((path / "results.json").read_text(encoding="utf-8"))
        for path in args.report_dirs
    ]
    seeds = [row for payload in payloads for row in payload["seeds"]]
    ids = [row["seed"] for row in seeds]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"Duplicate seeds across reports: {ids}")
    merged = {
        **payloads[0],
        "config": {**payloads[0]["config"], "seeds": ids},
        "seeds": seeds,
        "rows": [row for payload in payloads for row in payload["rows"]],
    }
    write_reports(merged, args.out)
    print(f"Merged {len(ids)} seeds -> {args.out}")


if __name__ == "__main__":
    main()
