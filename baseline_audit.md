# Stage 1 baseline audit

## Verdict

**NO-GO for the proposed COMMIT-vs-REPLAN method on the audited SpatialStack
checkpoint.** This is an architecture mismatch, not a failed implementation.

The runnable local model is a geometry-augmented Qwen3.5 direct-action policy. It
does not expose InternVLA-N1, DualVLN, a persistent local goal, a separate System 1,
or a separately scheduled System 2. Creating those concepts in the wrapper would
manufacture the intervention that Stage 1 is meant to verify.

## Repository and environment

- Repository: `https://github.com/phamquandung/SpatialStack.git`
- Branch/SHA: `accelerate-eval` / `844cb09577b0c3d8d0cfddbdd9d5d5590a01ad01`
- Dirty files present before this work: `config/vln_dagger_rxr.yaml`,
  `scripts/dagger.sh`, `src/dagger.py`, and untracked `checkpoints/`. They were not
  modified by this harness repository.
- Habitat-Lab SHA: `1639e1ae732ba1e84199a1a04b79c7243c3f8586`
- Habitat-Sim SHA: `f179b584bcd713c5a2a998132211e2cae881d6d1`
- Python 3.10.20; PyTorch 2.8.0+cu128; Transformers 5.3.0; NumPy 1.26.4;
  Habitat-Lab/Sim 0.2.4.
- GPU used: NVIDIA GeForce RTX 5090, 32,607 MiB, driver 580.178.04.
- Dataset: R2R-CE `val_unseen`, 1,839 episodes in 11 scenes; source manifest SHA-256
  `1767a407e2c8a011fbb7abece76cd64c5b39ff9fa0e9e340ebdce5a490d167c3`.

## Checkpoint identity

- Model directory: `spatialstack_janus_vln_train-gate-scale-4B-loss-3`
- Architecture: `Qwen3_5ForConditionalGenerationWithGeometry`
- Precision/device placement: bfloat16, entire model on one CUDA device.
- Model blob: 14,716,806,192 bytes; Hugging Face metadata revision
  `618d4c28922fc3695620dbb87bb5da80ba4cc932`, blob SHA-256
  `66ce6315a78e429ef15790ab6d68174caea4a747b097e4c05483cc9b4e188e6a`.
- Geometry encoder: VGGT-1B, 5,026,367,224 bytes; local file SHA-256
  `f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e`.
- This checkpoint is **not** InternVLA-N1 or DualVLN.
- Model loading printed many `UNEXPECTED` geometry/fusion keys. Although inference
  completed, checkpoint/module compatibility remains an integrity warning that must
  be resolved before publication claims.

## Actual execution contract

Observed source contract at the audited SHA:

```text
RGB observation + instruction + evenly sampled RGB history (up to 9 frames)
        ↓
Qwen3.5/VGGT call_model() — invoked every Habitat step
        ↓
decoded text parsed as one of four discrete action names
        ↓
fixed name→Habitat integer mapping
        ↓
env.step(action)
        ↓
new RGB observation; repeat until STOP/environment limit
```

Relevant code locations in `src/evaluation.py`:

- Prompt/history construction: lines 135–147 and 577–587.
- Model input and current-frame VGGT geometry: lines 297–330.
- Generation and action parsing: lines 332–390 and 150–158.
- Discrete action mapping: lines 442–447.
- Per-step model call followed by `env.step`: lines 574–631.
- Max-step forced STOP: lines 624–628.

RGB enters the model. The configured depth, GPS, compass, simulator pose, goal
position, and evaluator distances do not enter `call_model`; tracing labels those as
analysis/evaluation-only. The VGGT streaming KV is model memory, not a grounded
navigation goal or System 1 trajectory.

## Required feasibility questions

1. **Is COMMIT a real action? — No.** There is no retained goal or trajectory to
   continue. Every simulator step first calls the same VLM policy.
2. **Does REPLAN produce a meaningful updated goal at a new observation? — No.** A
   new observation produces another immediate discrete action, not a refreshed goal.
3. **Are failures attributable to wrong replan timing? — Not identifiable here.**
   There is no replan schedule. Observed failures can support direct-policy,
   perception, execution, or STOP analysis, but not goal-commitment claims.
4. **Can evaluator states be restored safely for paired branching? — Not yet.** No
   serializer exists for Habitat task/measure state, RGB history, VGGT KV cache,
   projected-geometry buffer, generation state, and all RNG states. Restoring only
   agent pose would leak/mismatch hidden policy state.

## Baseline and trace evidence

The genuine native evaluator ran three model-controlled R2R-CE episodes with seed 42
and a smoke-only three-step bound followed by forced STOP. Results are stored in
`artifacts/stage1/baseline_smoke_result.json`. SR/SPL/OS were 0 for all three and mean
final NE was 7.725 m; these bounded smoke values are not benchmark results.

The matched trace-on run preserved episode IDs, generated action sequences, issued
actions, step counts, SR, SPL, OS, and NE. All three traces passed the N-actions/N+1
states, monotonic index, causal-history, and reference checks. Timing and CUDA memory
values differed as expected and were excluded from behavioral equivalence.

Tracing is disabled by default and wraps the native model call and Habitat environment
only after `--trace`. It records direct policy calls explicitly; it does not label them
as System 2 or invent goal/S1 fields.

## Revised interface-level hypothesis

Before studying learned goal commitment, obtain an actual InternVLA-N1/DualVLN or
other evaluator with a verified persistent goal and independently callable executor.
If that backbone is unavailable, revise the research question to selective invocation
of a direct-action VLM under a separately specified action-repeat or lightweight
policy interface. That is a different intervention and requires a new fairness and
safety analysis; it must not be called COMMIT-vs-REPLAN of an existing goal.

