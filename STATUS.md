# Current Status

## Current phase

Stage 2 task-graph semantics: pure builder implemented and tested; neural
integration is paused at an environment/specification conflict.

## Completed

- Stage 0 audit.
- Stage 1 three-seed ACAC `ovcA7` reproduction at the official 20M-step budget.
- Leakage-free immutable task-graph input contract.
- Pure NumPy `TaskGraphBuilder` with all required output tensors.
- V1 local macro-observation decoder for `ovcA7` and `ovcAR7`.
- Ten required semantic scenarios plus additional plate, leakage, real-reset,
  and random-reset tests; 15 task-graph tests pass.

## Verified

- Actual V1 action indices and macro termination behavior drive task mapping.
- Hidden targets expose neither location nor dynamic state and are infeasible.
- Busy agents receive no new task decision.
- Wait prevents an empty masked-softmax row for ready agents.
- No task-graph or neural code has been connected to baseline ACAC training.

## Current blocker

The specification masks locally invisible targets and defines no exploration
or movement task. The real `ovcA7` initial observation gives agents 0 and 1 only
Wait; after agent 2 picks an ingredient, knives may remain invisible. Since Wait
does not change observation, the six-type graph can deadlock. The original
environment exposes `right/down/left/up`, so dropping them changes reachability.

## Recommended resolution

Add four directional Explore/Move task nodes mapped to the unchanged primitive
movement macros. Their feasibility should use the perspective robot's position
and static map topology, without hidden dynamic entity locations. Keep the six
specified semantic task types and Wait unchanged. This is the smallest option
that preserves local observability and the original executor/action coverage.

## Not completed

- Resolution and tests for exploration semantics.
- TG-ACAC-NoDependency neural integration.
- Full TG-ACAC dependency propagation.
- ACAC-MLP and pilot experiments.
