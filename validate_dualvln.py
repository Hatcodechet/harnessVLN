#!/usr/bin/env python3
"""Validate exact episode selection, DualVLN traces, and optional run equivalence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRIC_FIELDS = ("success", "spl", "os", "ne", "ndtw", "steps")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["scene_id"]), str(row["episode_id"])


def validate_run(run_dir: Path, require_trace: bool) -> dict[str, Any]:
    errors: list[str] = []
    selected_path = run_dir / "selected_episodes.json"
    progress_path = run_dir / "progress.json"
    if not selected_path.exists():
        errors.append(f"missing {selected_path}")
        selected = []
    else:
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
    progress = read_jsonl(progress_path)
    selected_keys = [key(row) for row in selected]
    progress_keys = [key(row) for row in progress]
    if progress_keys != selected_keys:
        errors.append(f"episode selection/order mismatch: selected={selected_keys}, progress={progress_keys}")

    trace_reports = []
    if require_trace:
        for scene, episode in selected_keys:
            episode_dir = run_dir / "traces" / f"{scene}_{episode}"
            metadata_path = episode_dir / "episode.json"
            events = read_jsonl(episode_dir / "events.jsonl")
            episode_errors = []
            if not metadata_path.exists():
                episode_errors.append("missing episode.json")
                metadata = {}
            else:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("status") != "complete":
                episode_errors.append(f"status is {metadata.get('status')!r}")
            indices = [event.get("event_idx") for event in events]
            if indices != list(range(len(events))):
                episode_errors.append("event indices are not contiguous")
            event_types = {event.get("event_type") for event in events}
            for required in ("observation", "system2_generation", "executed_action"):
                if required not in event_types:
                    episode_errors.append(f"missing event type {required}")
            system1_events = sum(
                event.get("event_type") == "system1_trajectory" for event in events
            )
            trace_reports.append(
                {
                    "episode": f"{scene}_{episode}",
                    "passed": not episode_errors,
                    "errors": episode_errors,
                    "event_count": len(events),
                    "system1_trajectory_count": system1_events,
                }
            )
            errors.extend(f"{scene}_{episode}: {error}" for error in episode_errors)

    return {
        "run_dir": str(run_dir),
        "passed": not errors,
        "errors": errors,
        "episode_count": len(progress),
        "trace_reports": trace_reports,
    }


def compare_runs(left: Path, right: Path) -> dict[str, Any]:
    left_rows = {key(row): row for row in read_jsonl(left / "progress.json")}
    right_rows = {key(row): row for row in read_jsonl(right / "progress.json")}
    errors = []
    if set(left_rows) != set(right_rows):
        errors.append(f"episode keys differ: {sorted(left_rows)} != {sorted(right_rows)}")
    differences = {}
    for episode in sorted(set(left_rows) & set(right_rows)):
        delta = {
            field: [left_rows[episode].get(field), right_rows[episode].get(field)]
            for field in METRIC_FIELDS
            if left_rows[episode].get(field) != right_rows[episode].get(field)
        }
        if delta:
            differences["/".join(episode)] = delta
            errors.append(f"{episode}: {delta}")
    return {"passed": not errors, "errors": errors, "differences": differences}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--require-trace", action="store_true")
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_run(args.run_dir, args.require_trace)
    if args.compare:
        report["equivalence"] = compare_runs(args.compare, args.run_dir)
        report["passed"] = report["passed"] and report["equivalence"]["passed"]
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
