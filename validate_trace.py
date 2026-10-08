#!/usr/bin/env python3
"""Validate Stage 1 causal trace invariants and optionally compare result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_episode(episode_dir: Path) -> Dict[str, Any]:
    errors: List[str] = []
    observations = read_jsonl(episode_dir / "observations.jsonl")
    decisions = read_jsonl(episode_dir / "decisions.jsonl")
    execution = read_jsonl(episode_dir / "execution.jsonl")
    episode_path = episode_dir / "episode.json"
    episode = json.loads(episode_path.read_text()) if episode_path.exists() else {}

    state_ids = [row.get("state_idx") for row in observations]
    if state_ids != list(range(len(observations))):
        errors.append(f"non-contiguous state indices: {state_ids}")
    if len(observations) != len(execution) + 1:
        errors.append(
            f"N+1 violation: observations={len(observations)} execution={len(execution)}"
        )
    decision_ids = {row.get("decision_idx") for row in decisions}
    for row in decisions:
        state_idx = row.get("state_idx")
        if state_idx not in set(state_ids):
            errors.append(f"decision {row.get('decision_idx')} references missing state {state_idx}")
        history = row.get("history_state_indices") or []
        if any(index > state_idx for index in history):
            errors.append(f"decision {row.get('decision_idx')} has future history reference")
        if len(history) != len(row.get("model_input_rgb_sha256") or []):
            errors.append(f"decision {row.get('decision_idx')} input digest count mismatch")
    for index, row in enumerate(execution):
        if row.get("state_idx_before") != index or row.get("state_idx_after") != index + 1:
            errors.append(f"execution {index} has invalid state transition")
        if row.get("decision_idx") not in decision_ids:
            errors.append(f"execution {index} references missing decision")
        if row.get("generated_action") is None or row.get("issued_action") is None:
            errors.append(f"execution {index} missing generated/issued action")
    for row in observations:
        rgb_path = row.get("rgb_path")
        if rgb_path and not (episode_dir / rgb_path).is_file():
            errors.append(f"missing RGB file: {rgb_path}")
    if episode.get("status") != "complete":
        errors.append(f"episode status is {episode.get('status')!r}, expected 'complete'")

    result = {
        "episode_dir": str(episode_dir),
        "passed": not errors,
        "errors": errors,
        "counts": {
            "observations": len(observations),
            "decisions": len(decisions),
            "executed_actions": len(execution),
        },
    }
    (episode_dir / "trace_validation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def result_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = read_jsonl(path)
    return {
        f"{row['scene_id']}_{row['episode_id']}": row
        for row in rows
        if "scene_id" in row and "episode_id" in row
    }


def compare_results(left: Path, right: Path) -> Dict[str, Any]:
    left_rows, right_rows = result_rows(left), result_rows(right)
    errors = []
    stable_fields = ("success", "spl", "os", "ne", "steps", "episode_instruction")
    if set(left_rows) != set(right_rows):
        errors.append(
            f"episode keys differ: left={sorted(left_rows)} right={sorted(right_rows)}"
        )
    comparisons = {}
    for key in sorted(set(left_rows) & set(right_rows)):
        differences = {
            field: [left_rows[key].get(field), right_rows[key].get(field)]
            for field in stable_fields
            if left_rows[key].get(field) != right_rows[key].get(field)
        }
        comparisons[key] = {"passed": not differences, "differences": differences}
        if differences:
            errors.append(f"{key}: {differences}")
    return {"passed": not errors, "errors": errors, "episodes": comparisons}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_run", type=Path)
    parser.add_argument("--compare-left", type=Path)
    parser.add_argument("--compare-right", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    episodes = [
        validate_episode(path)
        for path in sorted(args.trace_run.iterdir())
        if path.is_dir() and (path / "episode.json").exists()
    ]
    report: Dict[str, Any] = {
        "passed": bool(episodes) and all(row["passed"] for row in episodes),
        "episode_count": len(episodes),
        "episodes": episodes,
    }
    if args.compare_left or args.compare_right:
        if not (args.compare_left and args.compare_right):
            parser.error("both --compare-left and --compare-right are required")
        report["result_equivalence"] = compare_results(args.compare_left, args.compare_right)
        report["passed"] = report["passed"] and report["result_equivalence"]["passed"]
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
