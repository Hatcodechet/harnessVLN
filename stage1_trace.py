"""Causal tracing hooks for SpatialStack's native Habitat VLN evaluator.

The hooks are installed only when explicitly requested by ``run_stage1.py``.
They wrap the existing model call and Habitat environment; they do not replace
the evaluator loop, prompt, model inputs, action parser, or termination logic.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np


ACTION_NAMES = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")


def _digest_array(array: np.ndarray) -> str:
    view = np.ascontiguousarray(array)
    return hashlib.sha256(view.tobytes()).hexdigest()


def _scene_id(episode: Any) -> str:
    scene_path = str(episode.scene_id).rstrip("/")
    return Path(scene_path).parent.name


def _pose(env: Any) -> Optional[Dict[str, Any]]:
    try:
        state = env.sim.get_agent_state()
        rotation = state.rotation
        if hasattr(rotation, "components"):
            rotation = list(rotation.components)
        else:
            rotation = [rotation.x, rotation.y, rotation.z, rotation.w]
        return {
            "position": np.asarray(state.position).tolist(),
            "rotation_xyzw_or_library_components": _jsonable(rotation),
            "provenance": "habitat_sim_ground_truth",
            "usage": "annotation_only",
        }
    except Exception:
        return None


class TraceController:
    def __init__(
        self,
        trace_root: Path,
        run_id: str,
        provenance: Dict[str, Any],
        save_images: bool = False,
    ) -> None:
        self.run_dir = trace_root / run_id
        self.run_id = run_id
        self.provenance = provenance
        self.save_images = save_images
        self.episode = None
        self.episode_dir: Optional[Path] = None
        self.state_idx = -1
        self.decision_idx = -1
        self.last_decision: Optional[Dict[str, Any]] = None
        self.started_monotonic_ns = 0
        self.path_length_m = 0.0
        self.previous_position: Optional[np.ndarray] = None
        self.last_issued_action: Optional[str] = None
        self.last_metrics: Dict[str, Any] = {}
        self.run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(self.run_dir / "run.json", provenance)

    def prepare_episode(self, episode: Any) -> None:
        if self.episode is not None:
            self.finalize_episode(self.last_metrics, terminal_reason="next_episode")
        self.episode = episode
        episode_key = f"{_scene_id(episode)}_{episode.episode_id}"
        self.episode_dir = self.run_dir / episode_key
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        # A resumed evaluator may retry an episode that failed mid-step. Reset only
        # that episode's generated streams so records never concatenate attempts.
        # Completed episodes are skipped by the native result.json resume logic.
        for filename in (
            "decisions.jsonl",
            "execution.jsonl",
            "observations.jsonl",
            "trace_validation.json",
        ):
            generated_path = self.episode_dir / filename
            if generated_path.exists():
                generated_path.unlink()
        self.state_idx = -1
        self.decision_idx = -1
        self.last_decision = None
        self.started_monotonic_ns = time.monotonic_ns()
        self.path_length_m = 0.0
        self.previous_position = None
        self.last_issued_action = None
        self.last_metrics = {}
        instruction = getattr(getattr(episode, "instruction", None), "instruction_text", None)
        goals = [getattr(goal, "position", None) for goal in getattr(episode, "goals", [])]
        _write_json(
            self.episode_dir / "episode.json",
            {
                **self.provenance,
                "status": "running",
                "scene_id": _scene_id(episode),
                "episode_id": str(episode.episode_id),
                "trajectory_id": getattr(episode, "trajectory_id", None),
                "instruction": instruction,
                "goal_positions": goals,
                "goal_positions_usage": "evaluation_only_not_model_input",
                "started_wall_time_ns": time.time_ns(),
                "started_monotonic_ns": self.started_monotonic_ns,
            },
        )

    def begin_episode(self, observations: Dict[str, Any], metrics: Dict[str, Any], env: Any) -> None:
        self.record_observation(observations, metrics, env)

    def record_observation(self, observations: Dict[str, Any], metrics: Dict[str, Any], env: Any) -> None:
        assert self.episode_dir is not None
        self.state_idx += 1
        now_mono = time.monotonic_ns()
        rgb = observations.get("rgb")
        depth = observations.get("depth")
        rgb_path = None
        rgb_digest = None
        if rgb is not None:
            rgb_array = np.asarray(rgb)
            rgb_digest = _digest_array(rgb_array)
            if self.save_images:
                from PIL import Image

                rel = Path("rgb") / f"{self.state_idx:06d}.png"
                (self.episode_dir / rel).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb_array).save(self.episode_dir / rel)
                rgb_path = str(rel)

        pose = _pose(env)
        if pose is not None:
            position = np.asarray(pose["position"], dtype=np.float64)
            if self.previous_position is not None:
                self.path_length_m += float(np.linalg.norm(position - self.previous_position))
            self.previous_position = position

        record = {
            "run_id": self.run_id,
            "episode_id": str(self.episode.episode_id),
            "scene_id": _scene_id(self.episode),
            "state_idx": self.state_idx,
            "timestamp_wall_ns": time.time_ns(),
            "timestamp_monotonic_ns": now_mono,
            "rgb_path": rgb_path,
            "rgb_sha256": rgb_digest,
            "rgb_shape": list(np.asarray(rgb).shape) if rgb is not None else None,
            "rgb_dtype": str(np.asarray(rgb).dtype) if rgb is not None else None,
            "depth_path": None,
            "depth_sha256": _digest_array(np.asarray(depth)) if depth is not None else None,
            "depth_shape": list(np.asarray(depth).shape) if depth is not None else None,
            "depth_usage": "analysis_only_not_model_input" if depth is not None else None,
            "simulator_pose": pose,
            "evaluation_metrics": _jsonable(metrics),
        }
        _append_jsonl(self.episode_dir / "observations.jsonl", record)

    def record_decision(
        self,
        task: str,
        step_id: int,
        frame_indices: Optional[Iterable[int]],
        observations: Iterable[Any],
        raw_outputs: Any,
        parsed_actions: Any,
        started_ns: int,
        ended_ns: int,
        error: Optional[str] = None,
    ) -> None:
        assert self.episode_dir is not None
        self.decision_idx += 1
        indices = list(frame_indices or [])
        goal_id = None
        parsed = list(parsed_actions or [])
        image_digests = []
        for image in observations:
            image_digests.append(_digest_array(np.asarray(image)))
        record = {
            "run_id": self.run_id,
            "episode_id": str(self.episode.episode_id),
            "scene_id": _scene_id(self.episode),
            "decision_idx": self.decision_idx,
            "decision_id": f"d_{self.decision_idx:06d}",
            "state_idx": self.state_idx,
            "step_idx": step_id,
            "timestamp_monotonic_ns": ended_ns,
            "history_state_indices": indices,
            "model_input_rgb_sha256": image_digests,
            "instruction": task,
            "policy_call_invoked": True,
            "policy_call_type": "single_vlm_direct_discrete_action_generation",
            "system2_invoked": False,
            "system2_interface_available": False,
            "system2_semantics": "not_applicable_no_distinct_system2_in_this_evaluator",
            "raw_decoded_outputs": raw_outputs,
            "parsed_actions": parsed,
            "goal_id": goal_id,
            "goal_type": "not_applicable_direct_discrete_action_policy",
            "goal_coordinate_frame": None,
            "goal_carried_across_steps": False,
            "goal_absence_reason": "model_output_is_an_immediate_discrete_action",
            "system1_invoked": False,
            "system1_interface_available": False,
            "trajectory_output": None,
            "generation": {"temperature": 0, "num_beams": 1, "max_new_tokens": 24},
            "model_latency_ms": (ended_ns - started_ns) / 1_000_000.0,
            "status": "error" if error else "ok",
            "error": error,
        }
        self.last_decision = record
        _append_jsonl(self.episode_dir / "decisions.jsonl", record)

    def record_step(self, action: Any, observations: Dict[str, Any], env: Any, started_ns: int) -> None:
        assert self.episode_dir is not None
        issued_idx = int(action)
        issued_name = ACTION_NAMES.get(issued_idx, f"UNKNOWN_{issued_idx}")
        generated_name = None
        if self.last_decision and self.last_decision.get("parsed_actions"):
            generated_name = self.last_decision["parsed_actions"][0]
        metrics = env.get_metrics()
        self.last_metrics = _jsonable(metrics)
        before = self.state_idx
        self.record_observation(observations, metrics, env)
        ended_ns = time.monotonic_ns()
        record = {
            "run_id": self.run_id,
            "episode_id": str(self.episode.episode_id),
            "scene_id": _scene_id(self.episode),
            "step_idx": before,
            "decision_idx": self.last_decision["decision_idx"] if self.last_decision else None,
            "decision_id": self.last_decision["decision_id"] if self.last_decision else None,
            "state_idx_before": before,
            "state_idx_after": self.state_idx,
            "generated_action": generated_name,
            "issued_action": issued_name,
            "executed_action": issued_name,
            "executed_action_observability": "assumed_from_successful_habitat_env_step_return",
            "action_conversion": "fixed_name_to_habitat_index",
            "environment_step_latency_ms": (ended_ns - started_ns) / 1_000_000.0,
            "environment_done": bool(env.episode_over),
            "timestamp_monotonic_ns": ended_ns,
        }
        self.last_issued_action = issued_name
        _append_jsonl(self.episode_dir / "execution.jsonl", record)
        if env.episode_over:
            reason = "stop_action" if issued_idx == 0 else "environment_termination"
            self.finalize_episode(metrics, terminal_reason=reason)

    def finalize_episode(self, metrics: Dict[str, Any], terminal_reason: str) -> None:
        if self.episode is None or self.episode_dir is None:
            return
        episode_path = self.episode_dir / "episode.json"
        existing = json.loads(episode_path.read_text()) if episode_path.exists() else {}
        existing.update(
            {
                "status": "complete",
                "terminal_reason": terminal_reason,
                "stopped_by_model": bool(
                    self.last_issued_action == "STOP"
                    and self.last_decision
                    and self.last_decision.get("parsed_actions", [None])[0] == "STOP"
                ),
                "final_metrics": _jsonable(metrics),
                "actions_executed": self.state_idx,
                "states_recorded": self.state_idx + 1,
                "decisions_recorded": self.decision_idx + 1,
                "traveled_path_length_m_from_gt_pose": self.path_length_m,
                "traveled_path_length_usage": "evaluation_diagnostic_only",
                "episode_wall_time_ms": (time.monotonic_ns() - self.started_monotonic_ns) / 1_000_000.0,
            }
        )
        _write_json(episode_path, existing)
        self.episode = None


class TracedEnv:
    def __init__(self, env: Any, controller: TraceController) -> None:
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "_controller", controller)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_env", "_controller"}:
            object.__setattr__(self, name, value)
            return
        setattr(self._env, name, value)
        if name == "current_episode":
            self._controller.prepare_episode(value)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        observations = self._env.reset(*args, **kwargs)
        self._controller.begin_episode(observations, self._env.get_metrics(), self._env)
        return observations

    def step(self, action: Any, *args: Any, **kwargs: Any) -> Any:
        started_ns = time.monotonic_ns()
        observations = self._env.step(action, *args, **kwargs)
        self._controller.record_step(action, observations, self._env, started_ns)
        return observations

    def close(self) -> Any:
        self._controller.finalize_episode(self._env.get_metrics(), terminal_reason="environment_close")
        return self._env.close()


def install_tracing(evaluation_module: Any, model: Any, controller: TraceController) -> None:
    """Install return-preserving runtime hooks around native evaluator objects."""
    original_config_env = evaluation_module.VLNEvaluator.config_env

    def traced_config_env(evaluator_self: Any) -> TracedEnv:
        return TracedEnv(original_config_env(evaluator_self), controller)

    evaluation_module.VLNEvaluator.config_env = traced_config_env

    original_decode = model.processor.batch_decode
    decode_capture: Dict[str, Any] = {"outputs": None}

    def traced_decode(*args: Any, **kwargs: Any) -> Any:
        outputs = original_decode(*args, **kwargs)
        decode_capture["outputs"] = list(outputs)
        return outputs

    model.processor.batch_decode = traced_decode
    original_call = model.call_model

    def traced_call(
        observations: Any,
        task: str,
        step_id: int,
        gen_kwargs: Optional[dict] = None,
        frame_indices: Any = None,
    ) -> Any:
        decode_capture["outputs"] = None
        started_ns = time.monotonic_ns()
        try:
            parsed = original_call(
                observations,
                task,
                step_id,
                gen_kwargs=gen_kwargs,
                frame_indices=frame_indices,
            )
        except Exception as exc:
            ended_ns = time.monotonic_ns()
            controller.record_decision(
                task,
                step_id,
                frame_indices,
                observations,
                decode_capture["outputs"],
                [],
                started_ns,
                ended_ns,
                error=repr(exc),
            )
            raise
        ended_ns = time.monotonic_ns()
        controller.record_decision(
            task,
            step_id,
            frame_indices,
            observations,
            decode_capture["outputs"],
            parsed,
            started_ns,
            ended_ns,
        )
        return parsed

    model.call_model = traced_call
