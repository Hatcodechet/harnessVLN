import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from stage1_trace import TraceController
from validate_trace import validate_episode


def write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def minimal_episode(tmp_path: Path) -> Path:
    episode = tmp_path / "scene_1"
    episode.mkdir()
    (episode / "episode.json").write_text(json.dumps({"status": "complete"}))
    write_jsonl(
        episode / "observations.jsonl",
        [{"state_idx": 0, "rgb_path": None}, {"state_idx": 1, "rgb_path": None}],
    )
    write_jsonl(
        episode / "decisions.jsonl",
        [{
            "decision_idx": 0,
            "state_idx": 0,
            "history_state_indices": [0],
            "model_input_rgb_sha256": ["abc"],
        }],
    )
    write_jsonl(
        episode / "execution.jsonl",
        [{
            "decision_idx": 0,
            "state_idx_before": 0,
            "state_idx_after": 1,
            "generated_action": "STOP",
            "issued_action": "STOP",
        }],
    )
    return episode


def test_minimal_valid_trace(tmp_path):
    assert validate_episode(minimal_episode(tmp_path))["passed"]


def test_missing_post_action_state_is_rejected(tmp_path):
    episode = minimal_episode(tmp_path)
    write_jsonl(episode / "observations.jsonl", [{"state_idx": 0, "rgb_path": None}])
    result = validate_episode(episode)
    assert not result["passed"]
    assert any("N+1 violation" in error for error in result["errors"])


def test_future_history_reference_is_rejected(tmp_path):
    episode = minimal_episode(tmp_path)
    write_jsonl(
        episode / "decisions.jsonl",
        [{
            "decision_idx": 0,
            "state_idx": 0,
            "history_state_indices": [1],
            "model_input_rgb_sha256": ["abc"],
        }],
    )
    result = validate_episode(episode)
    assert not result["passed"]
    assert any("future history" in error for error in result["errors"])


def test_direct_policy_call_is_not_mislabeled_as_system2(tmp_path):
    controller = TraceController(tmp_path, "run", {"trace_enabled": True})
    episode = SimpleNamespace(
        scene_id="mp3d/scene/scene.glb",
        episode_id="1",
        trajectory_id=1,
        instruction=SimpleNamespace(instruction_text="go forward"),
        goals=[],
    )
    controller.prepare_episode(episode)
    controller.state_idx = 0
    controller.record_decision(
        "go forward",
        0,
        [0],
        [np.zeros((2, 2, 3), dtype=np.uint8)],
        ["MOVE_FORWARD"],
        ["MOVE_FORWARD"],
        1,
        2,
    )
    decision = json.loads(
        (tmp_path / "run/scene_1/decisions.jsonl").read_text().splitlines()[0]
    )
    assert decision["policy_call_invoked"] is True
    assert decision["system2_invoked"] is False
    assert decision["goal_carried_across_steps"] is False
