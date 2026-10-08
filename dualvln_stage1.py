#!/usr/bin/env python3
"""Run a frozen, manifest-selected InternNav DualVLN Stage 1 evaluation.

The upstream evaluator has no episode-limit CLI.  This runner initializes the
official evaluator, selects exact (scene_id, episode_id) pairs before evaluation,
and optionally installs return-preserving tracing hooks.  It does not change the
prompt, model generation, action conversion, rewards, or termination behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--subset", required=True, choices=("smoke_subset", "development_subset"))
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--save-trace-images", action="store_true")
    parser.add_argument(
        "--allow-resume",
        action="store_true",
        help="Allow an output directory containing progress.json to resume.",
    )
    return parser.parse_args()


def load_eval_cfg(config_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("dualvln_stage1_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load config: {config_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.eval_cfg


def scene_id(episode: Any) -> str:
    return str(episode.scene_id).rstrip("/").split("/")[-2]


def episode_key(episode: Any) -> tuple[str, str]:
    return scene_id(episode), str(episode.episode_id)


def git_sha(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    args.config = args.config.resolve()
    args.manifest = args.manifest.resolve()
    args.model_path = args.model_path.resolve()
    args.dataset_manifest = args.dataset_manifest.resolve()
    args.output_path = args.output_path.resolve()

    for path, label in (
        (args.config, "config"),
        (args.manifest, "episode manifest"),
        (args.model_path, "model checkpoint"),
        (args.dataset_manifest, "R2R dataset manifest"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    progress_path = args.output_path / "progress.json"
    if progress_path.exists() and not args.allow_resume:
        raise RuntimeError(
            f"Refusing to mix results with existing {progress_path}. "
            "Use a new RUN_NAME or pass --allow-resume explicitly."
        )

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    dataset_sha256 = sha256_file(args.dataset_manifest)
    expected_sha256 = manifest.get("source_manifest_sha256")
    if expected_sha256 and dataset_sha256 != expected_sha256:
        raise RuntimeError(
            f"Dataset manifest checksum mismatch: {dataset_sha256} != {expected_sha256}"
        )
    requested_rows = manifest[args.subset]
    requested = [(str(row["scene_id"]), str(row["episode_id"])) for row in requested_rows]
    if len(requested) != len(set(requested)):
        raise ValueError(f"Duplicate episode key in {args.subset}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    from internnav.evaluator import Evaluator

    cfg = load_eval_cfg(args.config)
    cfg.agent.model_settings["model_path"] = str(args.model_path)
    cfg.eval_settings["output_path"] = str(args.output_path)

    evaluator = Evaluator.init(cfg)
    available = {episode_key(episode): episode for episode in evaluator.env.episodes}
    completed = {
        (str(row["scene_id"]), str(row["episode_id"]))
        for row in (
            json.loads(line)
            for line in progress_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    } if progress_path.exists() else set()
    missing = [key for key in requested if key not in available and key not in completed]
    if missing:
        suffix = " (already completed episodes are omitted during resume)" if args.allow_resume else ""
        raise RuntimeError(f"Requested episodes unavailable: {missing}{suffix}")

    evaluator.env.episodes = [available[key] for key in requested if key in available]
    evaluator.env._current_episode_index = 0
    evaluator.env.is_running = True

    args.output_path.mkdir(parents=True, exist_ok=True)
    root = Path.cwd()
    run_metadata = {
        "stage": "Stage 1 frozen baseline",
        "model_variant": "InternVLA-N1 DualVLN",
        "subset_name": args.subset,
        "episode_count": len(requested),
        "episodes": [
            {"scene_id": scene, "episode_id": episode} for scene, episode in requested
        ],
        "completed_before_start": [
            {"scene_id": scene, "episode_id": episode}
            for scene, episode in requested
            if (scene, episode) in completed
        ],
        "seed": args.seed,
        "trace_enabled": args.trace,
        "save_trace_images": args.save_trace_images,
        "config_path": str(args.config),
        "manifest_path": str(args.manifest),
        "dataset_manifest_path": str(args.dataset_manifest),
        "dataset_manifest_sha256": dataset_sha256,
        "model_path": str(args.model_path),
        "internnav_root": str(root),
        "internnav_git_sha": git_sha(root),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    write_json(args.output_path / "run_metadata.json", run_metadata)
    write_json(args.output_path / "selected_episodes.json", run_metadata["episodes"])

    if args.trace:
        harness_root = Path(os.environ["HARNESS_ROOT"]).resolve()
        if str(harness_root) not in sys.path:
            sys.path.insert(0, str(harness_root))
        from dualvln_trace import install_dualvln_tracing

        install_dualvln_tracing(
            evaluator,
            trace_root=args.output_path / "traces",
            run_metadata=run_metadata,
            save_images=args.save_trace_images,
        )

    print(
        f"Running {len(requested)} exact episodes from {args.subset}; "
        f"trace={'on' if args.trace else 'off'}; output={args.output_path}",
        flush=True,
    )
    result = evaluator.eval()
    write_json(args.output_path / "stage1_result.json", result)


if __name__ == "__main__":
    main()
