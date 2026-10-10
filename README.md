# TG-ACAC Overcooked research workspace

This project studies task-graph-conditioned decisions on the official ACAC
Overcooked implementation. The current phase is RQ1 public-environment
validation; see `PROJECT_SPEC.md`, `EXPERIMENT_PROTOCOL.md`, and `STATUS.md` for
scope and progress.

## Source provenance

The code under `acac/` comes from the official
[LGAI-Research/acac](https://github.com/LGAI-Research/acac) repository at
commit `5ca6afb5e00f281c2be1f4cf98f3493c80714d4e`. Its BSD-3-Clause
license and notices remain in `acac/License.md` and
`acac/Notice.md`. Local adaptations are recorded in `DECISIONS.md`.

## Local setup and smoke command

Create the `acac` Python environment using the instructions in
`acac/README.md`. From `acac/`, run:

```powershell
python scripts/acac_main.py --exp_name audit_acac_smoke --alg_name acac --env_name ovcA7 --seed 1 --total_epi 16 --eval_freq 8 --eval_num_epi 2
```

Outputs default to this project's `data/` directory. The local `data/`, the
official bundled `icml25_result/`, and paper PDFs are intentionally excluded
from Git. Run details and known limitations are in `REPRODUCIBILITY.md`.


## TG-ACAC Stage 3 entry

From the repository root, with the acac Conda environment active:

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
python acac/scripts/tg_acac_main.py --alg_name tg_acac_no_dependency --exp_name tg_nodep_smoke100 --env_name ovcA7 --seed 0 --total_epi 100 --n_env 4 --train_freq 4 --eval_freq 20 --eval_num_epi 4 --device cpu
python acac/scripts/tg_acac_main.py --alg_name tg_acac --exp_name tg_full_smoke100 --env_name ovcA7 --seed 0 --total_epi 100 --n_env 4 --train_freq 4 --eval_freq 20 --eval_num_epi 4 --device cpu
```

Use a new experiment name for a new run to avoid mixing checkpoints.
These are debugging budgets, not pilot commands. Current TG runtime supports
V1 map A (ovcA7/ovcAR7); map B and V0 scale support are pending.
Original ACAC remains available via acac/scripts/acac_main.py --alg_name acac.

## Stage 4: ACAC-MLP and runtime profiling

ACAC-MLP uses --alg_name acac_mlp via acac/scripts/acac_main.py.
Only the post-GRU macro head is enlarged. mlp_head_width=0 automatically
matches full TG Actor parameters within mlp_match_tolerance=0.01.
Current V1 map-A settings select width 126; the GRU encoder is unchanged.

Repository root, acac environment, one OMP/MKL thread:

```powershell
python analysis/profile_stage4.py --train --run-prefix stage4_repeat1 --output-dir data/stage4_repeat1
```

This is short cost calibration. Use fresh names for repeats. See
analysis/STAGE4_REPORT.md; outputs stay in data/. GPU training remains unverified.


## Pilot preflight diagnostics

Use --record_pilot_metrics true for common diagnostic metrics and corrected
delivery counting; --record_delivery_metrics true enables delivery alone.
Both default false; --profile_resources true records CUDA allocator peaks.
Metric definitions: EXPERIMENT_PROTOCOL.md. Resource report:
analysis/PILOT_PREFLIGHT_REPORT.md. Current 10M-step TG pilot seeds are
estimated at roughly 86-94 hours; profile PPO graph preparation before launch.
Existing result CSVs remain untouched. Pilot has not started.
