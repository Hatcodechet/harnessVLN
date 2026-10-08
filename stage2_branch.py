"""Controlled COMMIT/REPLAN intervention hook for the DualVLN pilot.

This module is imported by a tiny pinned-InternNav hook only when
``DUALVLN_BRANCH_SPEC`` is set.  It records a pre-intervention signature and
returns the requested choice; all policy-state mutation stays in the upstream
evaluator hook where the relevant local variables exist.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _digest_bytes(parts: list[bytes]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def _array_digest(value: Any) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


def _rng_signature() -> dict[str, str]:
    numpy_state = np.random.get_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return {
        "python": hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
        "numpy": _digest_bytes(
            [
                str(numpy_state[0]).encode(),
                np.ascontiguousarray(numpy_state[1]).tobytes(),
                repr(numpy_state[2:]).encode(),
            ]
        ),
        "torch_cpu": hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
        "torch_cuda": _digest_bytes([state.cpu().numpy().tobytes() for state in cuda_states]),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class Stage2BranchController:
    def __init__(self, spec_path: Path, mode: str, output_path: Path) -> None:
        if mode not in {"commit", "replan"}:
            raise ValueError(f"DUALVLN_BRANCH_MODE must be commit or replan, got {mode!r}")
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        self.mode = mode
        self.candidates = {
            (str(row["scene_id"]), str(row["episode_id"])): row
            for row in spec["candidates"]
        }
        self.triggered: set[tuple[str, str]] = set()
        self.output_path = output_path / "branch_events.jsonl"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists():
            self.output_path.unlink()

    @classmethod
    def from_environment(cls, evaluator_output_path: str) -> "Stage2BranchController":
        return cls(
            Path(os.environ["DUALVLN_BRANCH_SPEC"]).resolve(),
            os.environ["DUALVLN_BRANCH_MODE"].strip().lower(),
            Path(evaluator_output_path).resolve(),
        )

    def maybe_branch(
        self,
        *,
        scene_id: str,
        episode_id: int,
        step_id: int,
        observations: dict[str, Any],
        pixel_goal: Any,
        local_actions: Any,
        action_seq: Any,
        forward_action: int,
    ) -> str | None:
        key = (str(scene_id), str(episode_id))
        candidate = self.candidates.get(key)
        if candidate is None or key in self.triggered:
            return None
        if step_id < int(candidate["target_step"]):
            return None
        # The intervention is meaningful only while a real pixel goal is active.
        if pixel_goal is None:
            return None

        event = {
            "candidate_id": candidate["candidate_id"],
            "scene_id": key[0],
            "episode_id": key[1],
            "requested_step": int(candidate["target_step"]),
            "actual_step": int(step_id),
            "mode": self.mode,
            "reason": candidate["reason"],
            "timestamp_ns": time.time_ns(),
            "snapshot_signature": {
                "rgb_sha256": _array_digest(observations.get("rgb")),
                "depth_sha256": _array_digest(observations.get("depth")),
                "pixel_goal": _jsonable(pixel_goal),
                "local_actions": _jsonable(list(local_actions)),
                "action_seq": _jsonable(list(action_seq)),
                "forward_action": int(forward_action),
                "rng": _rng_signature(),
            },
        }
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        self.triggered.add(key)
        print(
            f"STAGE2_BRANCH candidate={candidate['candidate_id']} mode={self.mode} "
            f"requested_step={candidate['target_step']} actual_step={step_id}",
            flush=True,
        )
        return self.mode
