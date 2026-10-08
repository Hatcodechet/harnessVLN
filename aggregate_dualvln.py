#!/usr/bin/env python3
"""Aggregate InternNav progress.json without inventing unavailable metrics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def mean(rows, field):
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return statistics.fmean(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    progress_path = args.run_dir / "progress.json"
    rows = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    csv_path = args.csv or args.run_dir / "episode_metrics.csv"
    summary_path = args.summary or args.run_dir / "summary.json"
    fields = ("scene_id", "episode_id", "success", "spl", "os", "ne", "ndtw", "steps", "episode_instruction")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "episode_count": len(rows),
        "success_rate": mean(rows, "success"),
        "spl": mean(rows, "spl"),
        "oracle_success": mean(rows, "os"),
        "navigation_error_m": mean(rows, "ne"),
        "ndtw": mean(rows, "ndtw"),
        "mean_steps": mean(rows, "steps"),
        "note": "A 20-episode Stage 1 diagnostic is not a publication-quality benchmark.",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
