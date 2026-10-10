# Current Status

## Current phase

Stage 5 preflight metrics, delivery repair and resource calibration are complete.
The three-seed pilot is NOT STARTED. Address TG PPO runtime cost before launching.

## Completed

- Stages 0-4: baseline reproduction, task graphs, TG variants, matched MLP.
- Opt-in delivery/local-feasibility/resource-contention/Wait/inference metrics.
- Detached TG raw pre-mask invalid mass and weighted-counter aggregation.
- Read-only common-support AUC and 12-run three-seed analysis tool.
- Eight short CPU/CUDA resource calibrations; proposed 25% pilot plan.

## Verified

- 45 unit tests pass; original observations, rewards, done and RNG unchanged.
- Existing baseline replay: delivery [0,0] -> [1,1], unchanged returns
  [226.6,226.6], episode lengths and discounted returns.
- All eight successful calibration CSVs and checkpoint/optimizer tensors finite;
  network weights restore. TG pre-mask columns exist and lie in [0,1].
- Final CUDA runs include actual EpisodeCount and raw pre-mask mass sums.
- Original GRU models, controller, environment, macro executors, memory,
  PPO/GAE learner and ACAC learn loop are unchanged.
- MLP/TG parameter matching remains verified by existing regression tests.

## Current blocker

No metric correctness blocker. Operational constraint: current TG PPO costs
make a 10M-step pilot seed roughly 86-94 hours in short-rate extrapolation.
Long pilot remains unlaunched pending runtime profiling/optimization or cost review.

## Resources and planning

- Proposed pilot: ovcAR7, four methods, seeds 0/1/2, total_epi=50000:
  10M-step target per seed (25% of formal 40M); original budget accounting retained.
- CPU point estimates per seed including eval proxy: ACAC 12.9h, MLP 10.9h,
  NoDependency 86.2h, full TG 89.6h. GPU is not faster for TG in this calibration.
- Serial 12-run point estimate about 599 CPU hours (25 days).
  These are planning extrapolations, not reliable long-run bounds.
- Full TG CUDA peak allocated 115.1 MiB, reserved 150 MiB; device-free ~3163 MiB.
  Peaks exclude context/display. Short runs fit GTX 1050 Ti 4 GiB.
- RSS sum across process tree roughly 3.6-4.2 GiB. Reserve 12 GiB outputs,
  24 GiB including working headroom. Current disk free ~506 GiB.
- Prefer serial runs; first profile TG graph preparation/cache work in PPO.

## Pending verification and limitations

- Real three-seed pilot results, learning advantage and continuation criteria.
- Long-run timing/memory behavior; current resource forecasts use early-policy runs.
- AUC aggregation tested with fixtures, not real pilot data.
- Invalid rate uses local initiation feasibility; contention is target competition;
  IdleRatio is explicit Wait. Mask-induced zero invalid rate alone is insufficient.
- Current graph diagnostics/TG/MLP matching support V1 map A; map B/V0 pending.
- TG obs_last_action=True unsupported.
- Old baseline delivery CSV fields remain unchanged and are known missing-event data.
- Some earlier CPU calibration files precede final auxiliary EpisodeCount additions.

## Evidence

- analysis/PILOT_PREFLIGHT_REPORT.md and analysis/pilot_plan.json
- data/pilot_preflight/calibration.json and resource_estimates.json
- data/pilot_preflight/delivery_replay.json, source_reuse.json and failures/
- EXPERIMENT_PROTOCOL.md: exact metric definitions and AUC/checkpoint rules

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


## Container preparation (2026-10-10)

Completed: Dockerfile, .dockerignore, requirements-container.txt, CONTAINER.md.
Shared Python 3.9.25/PyTorch 2.1.0+cu121 image for all four algorithms; outputs
at /data. Build gates include unit tests, pip check and CLI help.
Verified: official base tag/digest exists; direct package pins match local versions.
Pending: actual Linux build, container smoke and GPU calibration.
Current blocker: docker executable unavailable in the agent shell.
Algorithm/environment code unchanged. No upload or long training started.

Local verification: 45 unit tests passed with OMP/MKL threads=1 and dummy SDL;
git diff --check passed. This is Windows verification, not a Linux image build.
