# Research & Implementation Plan — Outcome-Aware Goal Commitment for VLN

**Status:** Revised proposal / implementation handoff (2026-10-08)  
**Working title:** *Outcome-Aware Goal Commitment for Vision-Language Navigation* (temporary)  
**Base:** InternNav / InternVLA-N1 (prefer DualVLN configuration if locally runnable), Habitat Sim  
**Research scope:** a **separate VLN harness paper**, independent of DETaP. The group's CoT-VLN work is complementary but **not required for Stage 1**.

> **Core question:** Given the current observation, goal, execution history, and available task-state evidence, **when should a VLN agent continue executing its existing goal (COMMIT), and when should it trigger System 2 to refresh the goal (REPLAN)?** Can the decision be learned from actual downstream navigation outcomes, rather than fixed schedules or uncertainty thresholds?

## 1. Problem

Even an initially reasonable grounded local goal can become stale or inconsistent with physical progress. In long-horizon VLN, repeated reasoning can also be wasteful or destabilize an otherwise correct execution. Offline CoT supervision on expert trajectories does not establish that the resulting model will recognize these cases under its own closed-loop execution.

**Hypothesis (not a fact):** for a nontrivial set of runtime decision states, the future execution outcome differs between `COMMIT` and `REPLAN`; a policy trained on paired outcome data can select better than fixed-period, confidence-based, and event-trigger heuristics at matched inference budget.

**Essential architectural precondition:** This problem is meaningful only if the real model/evaluator exposes (or naturally supports) a local goal/plan that persists for more than one execution step, or a distinct S2 invocation schedule. **Audit this before implementing any controller.** If S2 already runs every step and no plan can persist, do not manufacture an artificial commitment interface. Report the blocker and reassess the research question.

## 2. Current work and related literature

### Existing group CoT-VLN work (do not conflate with this paper)

The group currently has a **data-generation pipeline**, not yet a fine-tuned CoT-VLN model. It replays expert Habitat trajectories and generates:
- Semantic subtasks and refined boundaries via VLM-assisted segmentation;
- deterministic traveled/remaining geometry and future-expert pixel-goal labels (with visibility/depth checks);
- sparse 2–4-sentence grounded reasoning at selected start/middle/end anchors.

These are **teacher labels**. Future expert waypoints, expert boundaries, oracle remaining distances, and simulator ground-truth poses must **not be used as runtime harness inputs**. They can support annotation and offline evaluation. The group’s separate proposal studies landmark–event–clause grounding and task-state revision; the proposed harness studies **when to use/revise an already generated navigation goal**.

### Nearby papers that constrain novelty

- **NavHarness** — adaptive local goals, explicit goal-completion verification, history compression: https://arxiv.org/abs/2609.39915
- **HarnessVLN** — training-free embodied agent harness, spatial-evidence checks, subgoal consistency, recovery: https://arxiv.org/abs/2609.15195
- **InternNav / InternVLA-N1 / DualVLN** — open VLN evaluation and model stack: https://github.com/InternRobotics/InternNav
- **InternNav installation/evaluation docs:** https://internrobotics.github.io/user_guide/internnav/quick_start/installation.html
- **InternNav releases (evaluator variants may differ):** https://github.com/InternRobotics/InternNav/releases
- **VerNav** and **DIRECT** — relevant selective verification / compute-routing baselines; **verify exact versions/URLs and mechanism details before citing in the manuscript**. Do not claim that triggering CoT, goal verification, or goal re-planning by itself is novel.

**Potential contribution to test:** *execution-outcome-supervised goal commitment decisions* under a frozen VLN model and matched runtime budgets. This remains a hypothesis until proven against strong schedulers, learned context routers, and existing methods.

## 3. Proposed method: COMMIT vs REPLAN

```text
Instruction + causal observation history (RGB[/D])
                     |
             VLN System 2
                     |
        Current grounded local goal
                     |
       Harness goal-commitment gate <--- execution feedback
            /                    \
    COMMIT                      REPLAN
   retain goal             invoke S2 on updated evidence
            \                    /
              System 1 / executor
                     |
         real actions, new observations
                     +----------> gate
```

**COMMIT:** keep the currently valid goal/trajectory commitment and execute the next bounded portion of it using the normal System 1 and action-conversion path.

**REPLAN:** invoke the existing System 2 on **the latest causal observation/history**, update the local goal through the *same validated goal-conditioning interface*, then continue via the *same* System 1. REPLAN is not 'ask identical model with identical input again'.

**Decision gate inputs:** only fields available online: goal ID/age, recent executed actions, goal/trajectory progress, state changes, stagnation, current RGB/features, and (later, if validated) predicted clause/event state or confidence from the CoT-VLN model. Use pose/depth only if supplied in the declared online configuration, and provide information-matched baselines.

**Decision gate output:** `COMMIT` or `REPLAN`, plus log of decision time, reason features, and cost. Fixed safety/terminal constraints apply before the gate; the gate must not bypass environment STOP rules or cause unbounded repeated replanning.

**Important control:** Distinguish *normal goal refresh imposed by the original evaluator* from *extra early REPLAN decided by the harness*. Every comparison must use a clearly documented maximum goal lifetime and equal execution interfaces.

## 4. Scientific method and learning objective (after feasibility pilot)

At a synchronized closed-loop checkpoint `h_t`, restore simulator state **and** policy context (S2 history/cache, active goal, S1 state if recurrent, random seeds, termination state). Then evaluate two branches:

- `COMMIT`: continue the valid current goal;
- `REPLAN`: refresh the goal using an updated observation and the existing S2 call;

Keep follow-up policy, execution horizon, action conversion and stopping criteria matched after the first branch choice. Collect repeated paired seeds where stochasticity matters.

Define the *estimated intervention advantage*:

`A(h_t) = E[G | h_t, REPLAN] - E[G | h_t, COMMIT]`

where `G` is a horizon-limited return computed from actual execution: terminal or subgoal success, validated route progress, false clause transitions, movement failures, and measured reasoning cost/delay. Report quality and cost **separately** before combining them, to avoid obscuring navigation failures with token savings.

If the pilot yields meaningful, heterogeneous branch outcomes, train a **small goal-commitment/value model** predicting `A(h_t)` (or two separate action values) from causal online features. Select `REPLAN` only when the predicted benefit outweighs measured cost and safety constraints; otherwise `COMMIT`.

**No causal overclaim:** two simulator branches approximate intervention effects only under valid restoration, matched downstream conditions and enough repeated trials. Do not treat a single lucky branch as ground truth.

## 5. Stage 1 — Baseline audit and instrumentation (DO THIS FIRST)

**Objective:** reproduce an unmodified pretrained InternVLA-N1/DualVLN Habitat closed-loop run, understand the exact S2-to-S1 scheduling and goal lifetime, and collect trustworthy failure traces. **Do not train or build the commitment model yet.**

### Stage 1A — Inspect actual repository and environment

1. Locate actual checked-out repository (may be local SpatialStack fork rather than upstream InternNav); record `git remote -v`, branch, SHA, dirty files, Python, CUDA/PyTorch, Habitat version, GPU, model checkpoint paths and benchmark configs.
2. Inspect the evaluator/config and identify the **actually runnable** variant (InternVLA-N1 S2 + executor, DualVLN, NavDP*, etc.). Do not call any variant `DualVLN` without verifying checkpoint and configuration.
3. Trace exact execution: `observation -> S2 invocation -> S2 output -> goal conditioning -> S1 trajectory -> action conversion -> executed simulator actions -> next observation / stopping`.
4. Answer: **How often is S2 called? Is a goal retained across multiple actions? Can S1 continue that goal without re-calling S2? What invalidates a goal? What is the S1 horizon?**
5. Locate existing evaluation commands/documentation and dataset assets (R2R-CE or RxR-CE). Prefer repository commands over invented commands.

### Stage 1B — Reproduce baseline

1. Run smallest supported smoke test (e.g., 1–3 episodes) with original configs, fixed seeds; record exact commands and failures.
2. If feasible, run a **fixed 20-episode development subset** for diagnostics; larger runs only after reproducibility and throughput are known. A 20-episode SR is diagnostic, not a publication-quality estimate.
3. Preserve model weights, prompts, observation availability, navigation behavior, action conversion and termination for the canonical baseline.

### Stage 1C — Add instrumentation (disabled by default)

Log at each decision/action boundary:
- `episode_id`, scene, step, timestamp; model/config versions and seeds;
- instruction reference, observation path/hash and its source timestamp;
- S2 call flag, invocation time, input history size, outputs and latency;
- local goal ID/type, coordinate frame, creation time, expiration/refresh, whether carried across steps;
- S1 input/trajectory, action conversion, proposed and **actually executed** actions;
- movement progress, collision/stuck signal (if available), stop reason, episode outcome;
- any available runtime task-state fields **with explicit provenance** (`predicted`, `rule_based`, `oracle_for_annotation_only`).

Use JSONL per episode and lightweight summary JSON/CSV. Avoid dumping full images or hidden expert trajectories into each record. Verify timestamp and coordinate consistency. Tracing must not change predicted actions or baseline outcome.

### Stage 1D — Failure analysis and feasibility gate

Classify a small manually inspected sample of failures:
- stale goal despite changed observation;
- premature / unnecessary S2 refresh;
- perception or landmark error;
- incorrect grounding / pixel goal;
- S1/trajectory execution failure;
- incorrect clause advance / premature STOP;
- other / undetermined.

**Stage 1 questions that must be answered explicitly:**
1. Is `COMMIT` a real, reproducible action in this codebase?
2. Does `REPLAN` produce a meaningful updated goal at **new observations**?
3. Are there consequential failures plausibly attributable to **wrong replan timing** rather than S1/vision limitations?
4. Can evaluator states be restored for future paired branching without hidden state leakage?

**Decision:** If (1) or (2) is false, STOP this version of the method and write a revised interface-level hypothesis instead of forcing the gate into the model.

**Stage 1 deliverables:** `baseline_audit.md` (repo/env, S2/S1 call graph, goal persistence answer), reproducible baseline command/config + result JSON, instrumented traces, `failure_cases.md` with linked samples, feasibility verdict and blockers.

## 6. Stage 2 — Minimal causal feasibility pilot (ONLY after Stage 1 gate)

- Choose **20–30** informative runtime checkpoints (nominal progress, stale-goal candidates, landmark ambiguity, blocked motion, suspected premature refresh) across multiple scenes/routes.
- Branch `COMMIT` and `REPLAN` under matched replay conditions; collect **40–60 primary branches**, plus targeted repeated seeds on disputed cases.
- For each pair, compare: new goal change, route progress, false clause transitions, eventual/horizon-limited success, elapsed seconds, S2 count and tokens when measurable.
- Check for *heterogeneity*: states where COMMIT helps and states where REPLAN helps; check stability across repeated seeds.
- **Go/no-go:** only expand if branch selection is meaningful, outcomes are not dominated by uncontrolled simulator variance, and measurable causal online features predict which branch benefits.

## 7. Later work — conditional research roadmap

**Stage 3 / Data:** Expand to hundreds of checkpoints **only if pilot passes**. Save per-checkpoint causal feature snapshots, alternative branch results and provenance. Split scenes/routes before creating branch variants. Collect on-policy checkpoints after training to address distribution shift.

**Stage 4 / Learning:** Train lightweight advantage/value gate with supervised outcome regression or pairwise ranking. Freeze S2 and S1. Compare against `always COMMIT`, `always REPLAN` where safe/meaningful, fixed S2 interval, uncertainty threshold, event-triggered heuristic, failure predictor and learned context-only router.

**Stage 5 / Main evaluations:** R2R-CE and RxR-CE unseen scenes with frozen evaluator and exact shared routes/seeds. Primary: **SR**; secondary: SPL/nDTW, false transitions, premature STOP, latency (wall clock), S2 calls, token cost, and recovery of previously failed trajectories. Report **matched compute** and **matched success** comparisons with uncertainty aggregated by route/scene, not by frame.

**Stage 6 / CoT-VLN integration:** After the group actually trains a validated CoT checkpoint, compare `pretrained`, `pretrained + gate`, `CoT-VLN`, `CoT-VLN + gate`. Integrate predicted event/clause state, not expert boundary/ground-truth future. Check whether the gate adds benefit even after grounded CoT.

**Key ablations:** no paired outcome supervision; no execution feedback; no goal-age/history; no event/clause features (once available); no cost constraint; no on-policy data aggregation; value regression vs pairwise ranking.

## 8. Failure modes, safeguards and claims

- **Architecture mismatch:** no persistent goal / continuous S2 replanning → the proposal may be inapplicable as written.
- **Redundant replanning:** identical observation leads to identical S2 output → need *new evidence* for a meaningful branch.
- **Unfair comparison:** forcing a long COMMIT where the original model would replan normally creates an artificial baseline; document schedules and match budgets.
- **Oracle leakage:** expert `remaining_m`, future pixel-goal labels, true subtask boundary and simulator pose cannot enter online harness unless independently available to both systems in an explicit oracle experiment.
- **Noisy return:** single branched rollout does not prove causal benefit; repeat samples and evaluate horizon choices.
- **Trivial benefit:** if fixed scheduling performs just as well, do not assert a learned-harness contribution.
- **Novelty overlap:** NavHarness verifies goal completion; HarnessVLN performs rule-guided consistency/recovery; selective compute routing exists. Proposed contribution must be **paired execution-outcome learning of a goal commitment gate**, confirmed experimentally.

## 9. Instructions for Codex

1. Read this file fully; audit current repo first. **Do not invent local paths, configs, model capabilities or evaluation commands.** Existing local group paths were not verified by this document.
2. Implement **Stage 1 only** unless specifically instructed otherwise. Do not introduce `COMMIT` / `REPLAN` policy modifications during canonical baseline reproduction.
3. Preserve original eval metrics and behavior, add minimal reversible structured logs; leave tracing disabled by default.
4. Produce a short evidence-backed answer to the four Stage 1 feasibility questions, and show actual commands/results/blockers.
5. If assets or checkpoint are missing, document exact missing dependencies instead of fabricating completion.
6. Do not train the group CoT model, do not call expert replay 'closed-loop model evaluation', and do not claim unpublished benchmark improvements.

## 10. Minimal acceptance checklist

- [ ] Exact repo SHA, environment and checkpoint variant recorded.
- [ ] Genuine model-driven closed-loop Habitat smoke test run (or blocker explicitly documented).
- [ ] S2 invocation schedule, goal persistence and S1 continuation behavior established from code + traces.
- [ ] Unmodified baseline evaluation command and frozen subset recorded.
- [ ] Trace-on / trace-off checked for behavior equivalence where runnable.
- [ ] At least several traced example episodes inspected, including failures if present.
- [ ] Clear **GO / NO-GO** verdict on real COMMIT-vs-REPLAN feasibility.

---

**Bottom line:** The first scientific contribution is *not* another four-agent orchestration diagram. It is a falsifiable hypothesis: **learning when to preserve versus refresh an existing navigation goal from downstream execution outcomes can improve closed-loop success**. Stage 1 first determines whether this intervention exists in the actual InternVLA-N1 implementation and whether observed failures justify studying it.
