# TG-ACAC Overcooked research workspace

This project studies task-graph-conditioned decisions on the official ACAC
Overcooked implementation. The current phase is RQ1 public-environment
validation; see `PROJECT_SPEC.md`, `EXPERIMENT_PROTOCOL.md`, and `STATUS.md` for
scope and progress.

## Source provenance

The code under `acac/acac/` comes from the official
[LGAI-Research/acac](https://github.com/LGAI-Research/acac) repository at
commit `5ca6afb5e00f281c2be1f4cf98f3493c80714d4e`. Its BSD-3-Clause
license and notices remain in `acac/acac/License.md` and
`acac/acac/Notice.md`. Local adaptations are recorded in `DECISIONS.md`.

## Local setup and smoke command

Create the `acac` Python environment using the instructions in
`acac/acac/README.md`. From `acac/acac/`, run:

```powershell
python scripts/acac_main.py --exp_name audit_acac_smoke --alg_name acac --env_name ovcA7 --seed 1 --total_epi 16 --eval_freq 8 --eval_num_epi 2
```

Outputs default to this project's `data/` directory. The local `data/`, the
official bundled `icml25_result/`, and paper PDFs are intentionally excluded
from Git. Run details and known limitations are in `REPRODUCIBILITY.md`.
