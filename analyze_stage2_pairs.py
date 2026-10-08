#!/usr/bin/env python3
"""Validate synchronized Stage 2 pairs and summarize intervention outcomes."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def episode_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["scene_id"]), str(row["episode_id"])


def trace_cost(run_dir: Path, scene: str, episode: str) -> dict[str, Any]:
    episode_dir = run_dir / "traces" / f"{scene}_{episode}"
    events = read_jsonl(episode_dir / "events.jsonl")
    metadata = json.loads((episode_dir / "episode.json").read_text(encoding="utf-8"))
    return {
        "s2_calls": sum(event["event_type"] == "system2_generation" for event in events),
        "s1_calls": sum(event["event_type"] == "system1_trajectory" for event in events),
        "wall_time_s": float(metadata["wall_time_ms"]) / 1000,
    }


def classify(commit: dict[str, Any], replan: dict[str, Any]) -> str:
    success_delta = float(replan["success"]) - float(commit["success"])
    if success_delta > 0:
        return "REPLAN_BETTER"
    if success_delta < 0:
        return "COMMIT_BETTER"
    ne_delta = float(replan["ne"]) - float(commit["ne"])
    spl_delta = float(replan["spl"]) - float(commit["spl"])
    if spl_delta > 0.05 or ne_delta < -0.5:
        return "REPLAN_BETTER"
    if spl_delta < -0.05 or ne_delta > 0.5:
        return "COMMIT_BETTER"
    return "TIE_OR_INCONCLUSIVE"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--commit-run", required=True, type=Path)
    parser.add_argument("--replan-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    candidates = {row["candidate_id"]: row for row in spec["candidates"]}
    commit_events = {row["candidate_id"]: row for row in read_jsonl(args.commit_run / "branch_events.jsonl")}
    replan_events = {row["candidate_id"]: row for row in read_jsonl(args.replan_run / "branch_events.jsonl")}
    commit_progress = {episode_key(row): row for row in read_jsonl(args.commit_run / "progress.json")}
    replan_progress = {episode_key(row): row for row in read_jsonl(args.replan_run / "progress.json")}

    errors = []
    expected = set(candidates)
    if set(commit_events) != expected:
        errors.append(f"commit branch events mismatch: {sorted(set(commit_events) ^ expected)}")
    if set(replan_events) != expected:
        errors.append(f"replan branch events mismatch: {sorted(set(replan_events) ^ expected)}")

    rows = []
    for candidate_id in sorted(expected & set(commit_events) & set(replan_events)):
        candidate = candidates[candidate_id]
        commit_event = commit_events[candidate_id]
        replan_event = replan_events[candidate_id]
        if commit_event["mode"] != "commit" or replan_event["mode"] != "replan":
            errors.append(f"{candidate_id}: wrong branch modes")
        if commit_event["actual_step"] != replan_event["actual_step"]:
            errors.append(f"{candidate_id}: actual branch steps differ")
        if commit_event["snapshot_signature"] != replan_event["snapshot_signature"]:
            errors.append(f"{candidate_id}: pre-intervention snapshot signatures differ")

        key = (candidate["scene_id"], candidate["episode_id"])
        if key not in commit_progress or key not in replan_progress:
            errors.append(f"{candidate_id}: missing terminal progress")
            continue
        commit = commit_progress[key]
        replan = replan_progress[key]
        commit_cost = trace_cost(args.commit_run, *key)
        replan_cost = trace_cost(args.replan_run, *key)
        rows.append(
            {
                "candidate_id": candidate_id,
                "scene_id": key[0],
                "episode_id": key[1],
                "requested_step": candidate["target_step"],
                "actual_step": commit_event["actual_step"],
                "reason": candidate["reason"],
                "commit_success": commit["success"],
                "replan_success": replan["success"],
                "commit_spl": commit["spl"],
                "replan_spl": replan["spl"],
                "commit_ne": commit["ne"],
                "replan_ne": replan["ne"],
                "commit_steps": commit["steps"],
                "replan_steps": replan["steps"],
                "commit_s2_calls": commit_cost["s2_calls"],
                "replan_s2_calls": replan_cost["s2_calls"],
                "commit_s1_calls": commit_cost["s1_calls"],
                "replan_s1_calls": replan_cost["s1_calls"],
                "commit_wall_time_s": commit_cost["wall_time_s"],
                "replan_wall_time_s": replan_cost["wall_time_s"],
                "outcome": classify(commit, replan),
            }
        )

    counts = {
        label: sum(row["outcome"] == label for row in rows)
        for label in ("REPLAN_BETTER", "COMMIT_BETTER", "TIE_OR_INCONCLUSIVE")
    }
    summary = {
        "passed": not errors,
        "errors": errors,
        "pair_count": len(rows),
        "snapshot_match_count": len(rows) if not errors else sum(
            commit_events[cid].get("snapshot_signature") == replan_events[cid].get("snapshot_signature")
            for cid in expected & set(commit_events) & set(replan_events)
        ),
        "outcome_counts": counts,
        "commit_sr": statistics.fmean(float(row["commit_success"]) for row in rows) if rows else None,
        "replan_sr": statistics.fmean(float(row["replan_success"]) for row in rows) if rows else None,
        "commit_spl": statistics.fmean(float(row["commit_spl"]) for row in rows) if rows else None,
        "replan_spl": statistics.fmean(float(row["replan_spl"]) for row in rows) if rows else None,
        "mean_extra_s2_calls": statistics.fmean(
            row["replan_s2_calls"] - row["commit_s2_calls"] for row in rows
        ) if rows else None,
        "interpretation": "Pilot only; repeat disputed pairs across matched seeds before causal claims.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paired_results.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if rows:
        with (args.output_dir / "paired_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
