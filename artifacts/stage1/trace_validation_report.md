# Trace validation report

Date: 2026-10-08. Seed: 42. Split: R2R-CE `val_unseen`.

Three episodes were run through the untouched native evaluator with trace absent,
then through the wrapper with `--trace`. Both runs used the same checkpoint, scenes,
history length, three-step smoke bound, and disabled oracle stop.

## Behavioral comparison

| Episode | Native generated actions | Trace-on generated actions | Metrics/steps |
|---|---|---|---|
| `2azQ1b91cZZ_10` | FWD, FWD, FWD, FWD | FWD, FWD, FWD, FWD | exact match |
| `8194nk5LbLH_220` | LEFT, LEFT, LEFT, LEFT | LEFT, LEFT, LEFT, LEFT | exact match |
| `EU6Fwq7SyZv_127` | RIGHT, FWD, FWD, FWD | RIGHT, FWD, FWD, FWD | exact match |

The fourth generated action is overridden by the native evaluator's max-step STOP;
the trace records generated, issued, and executed actions separately. For all three
episodes, success, SPL, oracle success, navigation error, and step count matched
exactly. Timing and allocator measurements were not required to match.

## Structural validation

Each episode recorded four decisions, four executed actions, and five simulator
states. All checks passed:

- contiguous state and step indices;
- N executed actions to N+1 observations;
- each execution references an existing decision;
- no history input references a future state;
- input digest counts match history frame counts;
- generated and issued actions are both present;
- complete terminal episode metadata.

Synthetic unit tests also verify rejection of a missing post-action state and a
future history reference. A schema test ensures the direct SpatialStack VLM call is
recorded as a policy call, not mislabeled as System 2.

Command:

```bash
python3 -m pytest -q tests/test_trace_validation.py
# 4 passed
```

