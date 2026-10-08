# Stage 1 failure cases

These cases come from a seven-episode partial diagnostic run using the projected
geometry cache. The run was intentionally stopped once the revised plan's NO-GO gate
was established; it is not a 20-episode benchmark and no aggregate SR is claimed.
Simulator distance and collision fields are annotation/evaluation-only evidence.

## `2azQ1b91cZZ_10`: passed the goal, then stopped far away

- 78 actions: 59 forward, 12 right, 6 left, 1 STOP.
- Distance-to-goal: 7.113 m initially, minimum 0.245 m, 7.217 m at STOP.
- Oracle success was reached during execution, but model STOP success was 0.
- Final collision count: 1.

Classification: **incorrect STOP/progress behavior**. The trace shows overshoot after
entering the success radius. It cannot be classified as stale-goal or bad replan timing
because this model exposes neither a goal nor replanning.

## `8194nk5LbLH_221`: timeout and route divergence

- 401 actions (the evaluator's 400-step bound plus forced STOP).
- 173 left, 107 right, 120 forward, 1 forced STOP.
- Distance-to-goal: 11.806 m initially, never improved below 11.806 m, and ended at
  22.903 m.
- Final collision count: 36; oracle success and model success were both 0.

Classification: **direct-policy navigation failure; possible perception/grounding or
collision-loop failure**. More manual visual inspection is needed to distinguish them.

## `EU6Fwq7SyZv_128`: partial approach, premature model STOP

- 69 actions: 30 forward, 26 left, 12 right, 1 STOP.
- Distance-to-goal: 4.615 m initially, minimum 2.748 m, 3.498 m at STOP.
- Oracle success was 1 but terminal success was 0; collision count was 1.

Classification: **incorrect STOP/progress behavior**. The model briefly entered the
3 m success radius and stopped after moving back outside it.

## `QUCTc6BB5sX_19`: overshoot after substantial progress

- 163 actions: 72 forward, 46 left, 44 right, 1 STOP.
- Distance-to-goal: 13.205 m initially, minimum 1.429 m, 5.097 m at STOP.
- Oracle success was 1 but terminal success was 0; collision count was 6.

Classification: **incorrect STOP/progress behavior**, with perception/grounding still
an unverified alternative explanation.

## Interpretation limit

None of these failures supplies evidence for a goal-commitment gate. They demonstrate
closed-loop direct-action failures, but the causal variable proposed by the plan does
not exist in the audited architecture. Any stale-goal or premature-S2-refresh label
would be fabricated.

