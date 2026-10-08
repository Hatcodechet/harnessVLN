#!/usr/bin/env python3
"""Aggregate native SpatialStack result.json rows without inventing unavailable metrics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for line in args.result_json.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "scene_id" in row:
            rows.append(row)
    fields = [
        "scene_id", "episode_id", "success", "spl", "os", "ne", "steps",
        "peak_vggt_kv_mb", "peak_alloc_mb", "peak_reserved_mb", "mean_step_ms",
        "mean_vggt_ms", "episode_time_s", "episode_instruction",
    ]
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "sample_size": len(rows),
        "sr": statistics.fmean(row["success"] for row in rows) if rows else None,
        "spl": statistics.fmean(row["spl"] for row in rows) if rows else None,
        "oracle_success": statistics.fmean(row["os"] for row in rows) if rows else None,
        "navigation_error_m": statistics.fmean(row["ne"] for row in rows) if rows else None,
        "ndtw": None,
        "ndtw_status": "not supplied by this evaluator",
        "mean_steps": statistics.fmean(row["steps"] for row in rows) if rows else None,
        "mean_step_ms": statistics.fmean(row["mean_step_ms"] for row in rows) if rows else None,
        "p50_mean_step_ms": percentile([row["mean_step_ms"] for row in rows], 0.50),
        "p95_mean_step_ms": percentile([row["mean_step_ms"] for row in rows], 0.95),
        "total_episode_time_s": sum(row["episode_time_s"] for row in rows),
        "max_peak_alloc_mb": max((row["peak_alloc_mb"] for row in rows), default=None),
        "max_peak_reserved_mb": max((row["peak_reserved_mb"] for row in rows), default=None),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
