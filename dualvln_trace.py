"""Disabled-by-default tracing hooks for InternNav's DualVLN evaluator."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ACTION_NAMES = {
    0: "STOP",
    1: "MOVE_FORWARD",
    2: "TURN_LEFT",
    3: "TURN_RIGHT",
    4: "LOOK_UP",
    5: "LOOK_DOWN",
}


def jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "values": tensor.tolist()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(payload), sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def scene_id(episode: Any) -> str:
    return str(episode.scene_id).rstrip("/").split("/")[-2]


def digest_array(value: Any) -> str | None:
    if value is None:
        return None
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def tensor_summary(value: Any, include_values: bool = False) -> Any:
    if not isinstance(value, torch.Tensor):
        return jsonable(value)
    tensor = value.detach()
    result: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }
    if include_values:
        result["values"] = tensor.float().cpu().tolist()
    return result


def compact_metrics(metrics: Any) -> Any:
    """Keep scalar metrics while avoiding multi-megabyte top-down maps per step."""
    if isinstance(metrics, dict):
        compact = {}
        for key, value in metrics.items():
            if key == "top_down_map":
                if value is not None:
                    compact[key] = {"omitted": True, "reason": "large visualization-only payload"}
                continue
            compact[str(key)] = compact_metrics(value)
        return compact
    if isinstance(metrics, (np.ndarray, torch.Tensor)):
        element_count = metrics.size if isinstance(metrics, np.ndarray) else metrics.numel()
        if element_count > 32:
            return {"shape": list(metrics.shape), "dtype": str(metrics.dtype), "values_omitted": True}
    return jsonable(metrics)


class DualVLNTracer:
    def __init__(self, trace_root: Path, run_metadata: dict[str, Any], save_images: bool) -> None:
        self.trace_root = trace_root
        self.run_metadata = run_metadata
        self.save_images = save_images
        self.episode: Any = None
        self.episode_dir: Path | None = None
        self.event_idx = 0
        self.sim_step_idx = 0
        self.s2_call_idx = 0
        self.traj_call_idx = 0
        self.started_ns = 0
        self.last_metrics: dict[str, Any] = {}
        self.trace_root.mkdir(parents=True, exist_ok=True)
        write_json(self.trace_root / "run.json", run_metadata)

    def emit(self, event_type: str, **fields: Any) -> None:
        if self.episode_dir is None:
            return
        append_jsonl(
            self.episode_dir / "events.jsonl",
            {
                "event_idx": self.event_idx,
                "event_type": event_type,
                "timestamp_wall_ns": time.time_ns(),
                "timestamp_monotonic_ns": time.monotonic_ns(),
                "scene_id": scene_id(self.episode),
                "episode_id": str(self.episode.episode_id),
                **fields,
            },
        )
        self.event_idx += 1

    def begin_episode(self, episode: Any, observations: dict[str, Any]) -> None:
        self.finalize("next_episode")
        self.episode = episode
        self.episode_dir = self.trace_root / f"{scene_id(episode)}_{episode.episode_id}"
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        event_file = self.episode_dir / "events.jsonl"
        if event_file.exists():
            event_file.unlink()
        self.event_idx = 0
        self.sim_step_idx = 0
        self.s2_call_idx = 0
        self.traj_call_idx = 0
        self.started_ns = time.monotonic_ns()
        self.last_metrics = {}
        write_json(
            self.episode_dir / "episode.json",
            {
                **self.run_metadata,
                "status": "running",
                "scene_id": scene_id(episode),
                "episode_id": str(episode.episode_id),
                "instruction": episode.instruction.instruction_text,
                "started_wall_ns": time.time_ns(),
            },
        )
        self.record_observation(observations, source="reset")

    def record_observation(self, observations: dict[str, Any], source: str) -> None:
        rgb = observations.get("rgb")
        depth = observations.get("depth")
        rgb_path = None
        if self.save_images and rgb is not None and self.episode_dir is not None:
            from PIL import Image

            relative = Path("rgb") / f"{self.sim_step_idx:06d}_{source}.png"
            (self.episode_dir / relative).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.asarray(rgb)).save(self.episode_dir / relative)
            rgb_path = str(relative)
        self.emit(
            "observation",
            source=source,
            sim_step_idx=self.sim_step_idx,
            rgb_sha256=digest_array(rgb),
            rgb_shape=list(np.asarray(rgb).shape) if rgb is not None else None,
            rgb_path=rgb_path,
            depth_sha256=digest_array(depth),
            depth_shape=list(np.asarray(depth).shape) if depth is not None else None,
            depth_provenance="online_habitat_sensor",
        )

    def record_env_step(self, action: Any, result: tuple[Any, Any, Any, Any], latency_ms: float) -> None:
        observations, reward, done, info = result
        action_int = int(action)
        self.sim_step_idx += 1
        self.last_metrics = compact_metrics(info)
        self.emit(
            "executed_action",
            sim_step_idx=self.sim_step_idx,
            issued_action_id=action_int,
            issued_action=ACTION_NAMES.get(action_int, f"UNKNOWN_{action_int}"),
            executed_action_observability="successful HabitatEnv.step return",
            reward=reward,
            done=bool(done),
            metrics=self.last_metrics,
            environment_step_latency_ms=latency_ms,
        )
        self.record_observation(observations, source="step")
        if done:
            self.finalize("habitat_episode_over")

    def finalize(self, reason: str) -> None:
        if self.episode_dir is None:
            return
        path = self.episode_dir / "episode.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update(
            {
                "status": "complete",
                "terminal_reason": reason,
                "simulator_steps_recorded": self.sim_step_idx,
                "system2_calls_recorded": self.s2_call_idx,
                "trajectory_calls_recorded": self.traj_call_idx,
                "final_metrics": self.last_metrics,
                "wall_time_ms": (time.monotonic_ns() - self.started_ns) / 1_000_000,
            }
        )
        write_json(path, payload)
        self.episode = None
        self.episode_dir = None


class TracedEnv:
    def __init__(self, env: Any, tracer: DualVLNTracer) -> None:
        object.__setattr__(self, "_wrapped", env)
        object.__setattr__(self, "_tracer", tracer)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_wrapped", "_tracer"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self._wrapped, name, value)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        observations = self._wrapped.reset(*args, **kwargs)
        if observations is not None and self._wrapped.is_running:
            self._tracer.begin_episode(self._wrapped.get_current_episode(), observations)
        return observations

    def step(self, action: Any, *args: Any, **kwargs: Any) -> Any:
        started = time.monotonic_ns()
        result = self._wrapped.step(action, *args, **kwargs)
        self._tracer.record_env_step(action, result, (time.monotonic_ns() - started) / 1_000_000)
        return result

    def close(self) -> Any:
        self._tracer.finalize("evaluator_close")
        return self._wrapped.close()


def install_dualvln_tracing(
    evaluator: Any,
    trace_root: Path,
    run_metadata: dict[str, Any],
    save_images: bool = False,
) -> DualVLNTracer:
    """Wrap native objects without changing returned values or control flow."""
    tracer = DualVLNTracer(trace_root, run_metadata, save_images)
    evaluator.env = TracedEnv(evaluator.env, tracer)

    original_generate = evaluator.model.generate

    def traced_generate(*args: Any, **kwargs: Any) -> Any:
        started = time.monotonic_ns()
        output = original_generate(*args, **kwargs)
        ended = time.monotonic_ns()
        sequences = output.sequences if hasattr(output, "sequences") else output
        input_ids = kwargs.get("input_ids")
        prompt_tokens = int(input_ids.shape[1]) if isinstance(input_ids, torch.Tensor) else 0
        decoded = evaluator.processor.tokenizer.decode(
            sequences[0][prompt_tokens:], skip_special_tokens=True
        )
        coords = [int(value) for value in re.findall(r"\d+", decoded)]
        pixel_goal = [coords[1], coords[0]] if len(coords) >= 2 else None
        tracer.emit(
            "system2_generation",
            system2_call_idx=tracer.s2_call_idx,
            prompt_tokens=prompt_tokens,
            input_image_tensor=tensor_summary(kwargs.get("pixel_values")),
            decoded_output=decoded,
            parsed_pixel_goal_xy=pixel_goal,
            generation_kwargs={
                key: jsonable(kwargs.get(key))
                for key in ("max_new_tokens", "do_sample", "use_cache")
            },
            latency_ms=(ended - started) / 1_000_000,
        )
        tracer.s2_call_idx += 1
        return output

    evaluator.model.generate = traced_generate

    if hasattr(evaluator.model, "generate_latents"):
        original_generate_latents = evaluator.model.generate_latents

        def traced_generate_latents(*args: Any, **kwargs: Any) -> Any:
            started = time.monotonic_ns()
            output = original_generate_latents(*args, **kwargs)
            tracer.emit(
                "goal_conditioning",
                output=tensor_summary(output),
                latency_ms=(time.monotonic_ns() - started) / 1_000_000,
            )
            return output

        evaluator.model.generate_latents = traced_generate_latents

    if hasattr(evaluator.model, "generate_traj"):
        from internnav.model.utils.vln_utils import traj_to_actions

        original_generate_traj = evaluator.model.generate_traj

        def traced_generate_traj(*args: Any, **kwargs: Any) -> Any:
            started = time.monotonic_ns()
            output = original_generate_traj(*args, **kwargs)
            try:
                converted_actions = [int(action) for action in traj_to_actions(output)]
            except Exception as error:  # tracing must never break native inference
                converted_actions = [f"trace_conversion_error: {error}"]
            tracer.emit(
                "system1_trajectory",
                trajectory_call_idx=tracer.traj_call_idx,
                output=tensor_summary(output, include_values=True),
                converted_actions=converted_actions,
                latency_ms=(time.monotonic_ns() - started) / 1_000_000,
            )
            tracer.traj_call_idx += 1
            return output

        evaluator.model.generate_traj = traced_generate_traj

    return tracer
