# Current Status

## Current phase

Stage 4 ACAC-MLP and four-method profiling are complete.
Next: prepare Stage 5 metrics and resource estimates before pilot.

## Completed

- Stage 0 audit, Stage 1 baseline, Stage 2 task graphs, Stage 3 TG.
- Separate acac_mlp algorithm; only original post-GRU head is enlarged.
- Configurable automatic/manual width and parameter matching tolerance.
- Four-method parameter, CPU/GPU inference and CPU training cost report.

## Verified

- 35 unit tests pass, including preserved MLP encoder, finite gradients,
  macro-action outputs and no graph access in MLP forward.
- MLP width 126: three Actors 44,100 vs full TG 44,064 (0.0817% difference).
- All Critic counts remain 290,497 trainable parameters.
- Four ovcAR7 seed-0 runs: each 16 episodes / 3,200 steps, identical settings.
- All CSV metrics, network and optimizer tensors finite; checkpoints restore.
- CPU train steps/s: ACAC 237.6; MLP 233.2; NoDependency 98.0; TG 91.6.
- Common-input CPU/GPU inference measured; warm/cold caches reported.
- Original environment, macros, models, memory, runner and PPO/GAE hashes unchanged.

## Current blocker

No blocker for completed Stage 4 V1 map-A scope.

## Pending verification and limitations

- Short runs establish pipeline/cost, not learning effectiveness.
- Full TG costs about 2.59x more than ACAC in this CPU throughput check.
- GTX 1050 Ti small-batch inference is slower than CPU; GPU training
  throughput and peak training memory remain unverified.
- Pilot task/conflict/idle metrics and delivery logger correction pending;
  original baseline CSVs must remain unchanged.
- Resource estimates required before any run expected above two hours.
- TG/MLP matching supports V1 map A; map B and V0 scale are pending.
- TG obs_last_action=True remains unsupported.
- Original budget and evaluation-interval step accounting unchanged.

## Not completed

- Stage 5 three-seed ovcAR7 pilot.
- Stage 6 formal experiments after pilot continuation criteria pass.

## Stage 4 evidence

- analysis/STAGE4_REPORT.md
- data/stage4/profile.json and summary.csv
- data/stage4/reuse_audit.json and failures/

## Repository layout maintenance (2026-09-29)

### Completed

- Flattened the former three-level repeated hierarchy into the single `acac/` directory.
- Updated runtime path setup, package discovery, tests, ignore rules, and all
  documentation references to the new locations.

### Verified

- `acac` imports from `acac/__init__.py` and resolves `PROJECT_DIR` to `acac/`.
- The project `acac` Conda environment passes all 16 unit tests.

No new blocker or pending work was introduced by the layout change.


## Stage 3 preflight (2026-10-08)

Verified with original V1 macros on fixed states:
- get tomato successfully retrieves chopped food from a knife, while the
  builder masks get:tomato because chopped/on-resource means completed;
- get plate 1 successfully adds a chopped ingredient to a partially filled
  plate, while the builder masks get:plate1 because contents means completed.

The existing 19 passing unit tests do not verify the complete assembly chain.
Initial Explore reachability is verified, but Stage 2 semantic completeness
must be revisited. Neural integration and training have not started.
Evidence: analysis/audit_task_graph_assembly.py and
data/audit/stage3_preflight.json. Await semantic resolution before integration.


The Stage 3 preflight findings below are historical; D016 resolves D015.
