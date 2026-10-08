# harnessVLN — Stage 1 feasibility audit

This repository implements the Stage 1 audit and disabled-by-default causal
instrumentation from [`vln_harness_plan.md`](vln_harness_plan.md).

The audited local backbone is **SpatialStack Qwen3.5-4B + VGGT-1B**, not
InternVLA-N1/DualVLN. Its Habitat evaluator calls one VLM policy at every simulator
step and directly emits `STOP`, `MOVE_FORWARD`, `TURN_LEFT`, or `TURN_RIGHT`. It has
no persistent local goal and no distinct System 2 → goal → System 1 interface.

Therefore the current Stage 1 verdict is **NO-GO for COMMIT vs REPLAN on this
backbone**. The code intentionally does not implement a commitment gate. It provides:

- a wrapper that leaves the native evaluator unchanged unless `--trace` is passed;
- causal per-episode JSONL traces for observations, policy decisions, and actions;
- validators for N-actions/N+1-states and causal history references;
- native result aggregation without fabricating unsupported metrics;
- the measured audit and failure evidence.

See [`baseline_audit.md`](baseline_audit.md) and
[`failure_cases.md`](failure_cases.md).

## Requirements

Use the environment already supported by the SpatialStack checkout. The measured
environment was Python 3.10, PyTorch 2.8/CUDA 12.8, Transformers 5.3, Habitat-Lab
0.2.4, and Habitat-Sim 0.2.4. The wrapper does not vendor or reinstall those projects.

Set paths explicitly:

```bash
export SPATIALSTACK_ROOT=/absolute/path/to/SpatialStack
export SPATIALSTACK_DATA_ROOT="$SPATIALSTACK_ROOT/data"
export CHECKPOINT="$SPATIALSTACK_ROOT/checkpoints/spatialstack_janus_vln_train-gate-scale-4B-loss-3"
export GEOMETRY_ENCODER_PATH="$SPATIALSTACK_ROOT/checkpoints/VGGT-1B"
```

For long episodes, use the repository-supported allocator and projected geometry cache:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLN_PROJECTED_GEOMETRY_CACHE=1
```

## Trace run

Tracing is disabled unless `--trace` is supplied:

```bash
MAX_EPISODES=1 torchrun --nproc_per_node=1 run_stage1.py \
  --spatialstack_root "$SPATIALSTACK_ROOT" \
  --model_path "$CHECKPOINT" \
  --geometry_encoder_path "$GEOMETRY_ENCODER_PATH" \
  --habitat_config_path configs/vln_r2r_local.yaml \
  --eval_split val_unseen \
  --scene_ids 2azQ1b91cZZ,8194nk5LbLH,EU6Fwq7SyZv \
  --output_path artifacts/stage1/results/smoke_trace_on \
  --max_steps 3 --seed 42 \
  --trace --trace_root artifacts/stage1/runs --run_id smoke_trace_on
```

Without `--trace`, `run_stage1.py` calls the native model and evaluator without
installing hooks.

## Validation

```bash
python3 -m pytest -q tests/test_trace_validation.py
python3 validate_trace.py artifacts/stage1/runs/smoke_trace_on
```

Large traces, images, datasets, checkpoints, and result directories are ignored.

## InternVLA-N1 DualVLN on the remote workstation

The remote setup uses the isolated `dualvln_harness` Conda environment and the
clean InternNav checkout at `/storage/anhdh35/InternNav_harness`. Run the scripts
from this repository in order:

```bash
bash scripts/dualvln/check_dualvln.sh
bash scripts/dualvln/smoke_dualvln_model.sh
RUN_FULL_EVAL=1 bash scripts/dualvln/run_dualvln_eval.sh
```

The full evaluation script deliberately requires `RUN_FULL_EVAL=1` because the
official config evaluates all of R2R `val_unseen`. Common overrides include:

```bash
GPU_INDEX=1 MASTER_PORT=2345 RUN_FULL_EVAL=1 \
  bash scripts/dualvln/run_dualvln_eval.sh
```

Paths can be overridden with `CONDA_ROOT`, `CONDA_ENV`, `INTERNNAV_ROOT`,
`CHECKPOINT_ROOT`, `JANUSVLN_ROOT`, `R2R_ROOT`, and `SCENES_ROOT`.
