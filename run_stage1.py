#!/usr/bin/env python3
"""Run SpatialStack's native evaluator with optional, disabled-by-default tracing."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spatialstack_root",
        default=os.environ.get("SPATIALSTACK_ROOT"),
        help="SpatialStack checkout (or set SPATIALSTACK_ROOT)",
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--geometry_encoder_path", default="")
    parser.add_argument("--habitat_config_path", default="configs/vln_r2r_local.yaml")
    parser.add_argument("--eval_split", default="val_unseen")
    parser.add_argument("--scene_ids", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--save_video_ratio", type=float, default=0.05)
    parser.add_argument("--trace", action="store_true", help="Enable causal JSONL tracing")
    parser.add_argument("--trace_root", default="artifacts/stage1/runs")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--trace_images", action="store_true")
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", default="1111")
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def git_sha(repo: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def checkpoint_revision(checkpoint: Path) -> str | None:
    metadata = checkpoint / ".cache/huggingface/download/model.safetensors.metadata"
    if not metadata.is_file():
        return None
    lines = metadata.read_text().splitlines()
    return lines[0].strip() if lines else None


def checkpoint_blob_sha256(checkpoint: Path) -> str | None:
    metadata = checkpoint / ".cache/huggingface/download/model.safetensors.metadata"
    if not metadata.is_file():
        return None
    lines = metadata.read_text().splitlines()
    return lines[1].strip() if len(lines) > 1 else None


def main() -> None:
    args = parse_args()
    if not args.spatialstack_root:
        raise SystemExit("Set SPATIALSTACK_ROOT or pass --spatialstack_root")
    spatialstack_root = Path(args.spatialstack_root).resolve()
    if not (spatialstack_root / "src/evaluation.py").is_file():
        raise SystemExit(f"Not a SpatialStack checkout: {spatialstack_root}")
    sys.path.insert(0, str(spatialstack_root / "src"))
    import evaluation

    scene_filter = {item.strip() for item in args.scene_ids.split(",") if item.strip()}
    if not scene_filter:
        raise ValueError("--scene_ids must include at least one scene")
    if args.trace and not args.run_id:
        raise ValueError("--run_id is required with --trace")

    evaluation.set_seed(args.seed)
    evaluation.init_distributed_mode(args)
    geometry_path = args.geometry_encoder_path or os.environ.get("GEOMETRY_ENCODER_PATH")
    model = evaluation.SpatialStackVLN_Inference(
        args.model_path,
        device=f"cuda:{args.local_rank}",
        geometry_encoder_path=geometry_path or None,
    )

    if args.trace:
        from stage1_trace import TraceController, install_tracing

        controller = TraceController(
            Path(args.trace_root),
            args.run_id,
            {
                "run_id": args.run_id,
                "trace_enabled": True,
                "trace_schema_version": "stage1-spatialstack-v1",
                "model_variant": "SpatialStack-Qwen3.5-4B-VGGT-direct-discrete-action",
                "architecture_contract": "architecture_mismatch_not_InternVLA_N1_not_DualVLN",
                "model_path": str(Path(args.model_path).resolve()),
                "geometry_encoder_path": str(Path(geometry_path).resolve()) if geometry_path else None,
                "habitat_config_path": str(Path(args.habitat_config_path).resolve()),
                "dataset_split": args.eval_split,
                "scene_ids": sorted(scene_filter),
                "seed": args.seed,
                "max_steps": args.max_steps,
                "num_history": args.num_history,
                "oracle_stop": os.environ.get("VLN_ORACLE_STOP", "0"),
                "teacher_forced": os.environ.get("VLN_TEACHER_FORCED", "0"),
                "projected_geometry_cache": os.environ.get("VLN_PROJECTED_GEOMETRY_CACHE", "0"),
                "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                "checkpoint_hf_revision": checkpoint_revision(Path(args.model_path)),
                "checkpoint_model_blob_sha256": checkpoint_blob_sha256(Path(args.model_path)),
                "spatialstack_git_sha": git_sha(spatialstack_root),
            },
            save_images=args.trace_images,
        )
        install_tracing(evaluation, model, controller)

    evaluation.evaluate(model, args, scene_filter=scene_filter)


if __name__ == "__main__":
    main()
