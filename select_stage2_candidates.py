#!/usr/bin/env python3
"""Select a frozen, non-cherry-picked Stage 2 pilot from the dev-20 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def row_key(row):
    return str(row["scene_id"]), str(row["episode_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dev20_run", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    validation = json.loads((args.dev20_run / "validation.json").read_text(encoding="utf-8"))
    if not validation.get("passed") or validation.get("episode_count") != 20:
        raise RuntimeError("dev-20 trace must pass validation before candidate selection")
    rows = read_jsonl(args.dev20_run / "progress.json")
    failures = [row for row in rows if float(row["success"]) == 0.0]
    successes = [row for row in rows if float(row["success"]) == 1.0]

    chosen: list[tuple[dict, str, float]] = []
    used = set()

    def take(pool, count, reason, fraction, sort_key, reverse=True):
        eligible = [row for row in pool if row_key(row) not in used]
        for row in sorted(eligible, key=sort_key, reverse=reverse)[:count]:
            chosen.append((row, reason, fraction))
            used.add(row_key(row))

    take(
        [row for row in failures if float(row.get("os", 0)) == 1.0],
        2,
        "oracle-success but terminal failure: inspect goal retention versus refresh near route completion",
        0.60,
        lambda row: int(row["steps"]),
    )
    take(
        failures,
        2,
        "long failed rollout: inspect stale-goal or repeated-plan behavior",
        0.50,
        lambda row: int(row["steps"]),
    )
    take(
        failures,
        2,
        "large terminal navigation error: inspect early wrong commitment",
        0.60,
        lambda row: float(row["ne"]),
    )
    take(
        successes,
        2,
        "successful control: test whether unnecessary replanning harms a working commitment",
        0.60,
        lambda row: float(row["spl"]),
    )

    candidates = []
    subset = []
    for index, (row, reason, fraction) in enumerate(chosen, 1):
        target = max(8, int(round(int(row["steps"]) * fraction)))
        candidate = {
            "candidate_id": f"c{index:02d}",
            "scene_id": str(row["scene_id"]),
            "episode_id": str(row["episode_id"]),
            "target_step": target,
            "source_success": float(row["success"]),
            "source_oracle_success": float(row.get("os", 0)),
            "source_navigation_error_m": float(row["ne"]),
            "source_steps": int(row["steps"]),
            "reason": reason,
        }
        candidates.append(candidate)
        subset.append({"scene_id": candidate["scene_id"], "episode_id": candidate["episode_id"]})

    output = {
        "dataset": "R2R-CE",
        "split": "val_unseen",
        "seed": 42,
        "selection_policy": (
            "Frozen automatic selection from validated dev-20: two oracle-only failures, "
            "two longest remaining failures, two largest-error remaining failures, and two high-SPL controls."
        ),
        "stage2_pilot_subset": subset,
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
