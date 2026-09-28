# Current Status

## Current phase

Stage 1 complete. Ready to begin Stage 2 task-graph semantics.

## Completed

- Stage 0 read-only audit and minimal modification plan.
- Windows-safe project-local output paths, timestamps, and checkpoint rotation.
- Seed 0 main-process initialization fix with regression coverage.
- ACAC smoke test and timed GPU/disk calibration on `ovcA7`.
- Full ACAC `ovcA7` baseline for seeds 0, 1, and 2 at the official 20M-step budget.
- Three-seed `Eval/ReturnAverage` aggregation and mean/95% CI curve.

## Verified

- All three baseline runs exceeded 20,000,000 environment timesteps and saved final checkpoints.
- `variant.json` records seeds 0, 1, and 2 and identical training/evaluation settings.
- No NaN or Inf values were found in any baseline `progress.csv`.
- Final returns are consistent with the official released ACAC results.
- No baseline Python training processes remain active.

## Not completed

- Pure-function `TaskGraphBuilder` and its ten required scenario tests.
- TG-ACAC-NoDependency and TG-ACAC.
- ACAC-MLP parameter-matched control.
- `ovcAR7` pilot and later formal experiments.

## Current risks and information gaps

- The original option `total_epi` defines `env_terminate_step * total_epi`
  environment steps; it is not a strict episode count. This behavior must be
  preserved for fair comparison.
- `Eval/NumDeliveryAverage` remains zero because the runner expects an info key
  that the current Overcooked wrapper does not provide. Returns are valid, but
  delivery-count reporting must be repaired before the pilot metrics are used.
- A leakage-free hard feasibility mask cannot use hidden global state. Stage 2
  must derive graph fields from the acting agent's observable state and static
  environment metadata only.

## Next action

Begin Stage 2 by freezing the task-graph input contract from actual environment
objects and observations. Implement the pure `TaskGraphBuilder` and the ten
required scenario tests before connecting any graph encoder to training.
