# Reproducibility Record

## Code

- ACAC repository URL:
- ACAC commit:
- Current branch:
- Local modifications:

## Machine

- OS:
- CPU:
- GPU:
- GPU driver:
- CUDA:
- RAM:

## Python environment

- Python:
- PyTorch:
- NumPy:
- Gym:
- Pygame:

## Commands

在这里逐条记录实际成功运行过的命令。

## Results

每个run记录：
- experiment name
- algorithm
- environment
- seed
- training budget
- output directory
- completion status

## Failed attempts

- 2026-09-22, `ovcA7`, seed 1, smoke command reported by user:
  `python scripts/acac_main.py --exp_name audit_acac_smoke --alg_name acac --env_name ovcA7 --seed 1 --total_epi 16 --eval_freq 8 --eval_num_epi 2`.
  Failed before training because the log directory timestamp contained `:`,
  which Windows rejects in a directory name (`WinError 123`).

## Verified smoke run (user executed)

- Command: `python scripts/acac_main.py --exp_name audit_acac_smoke --alg_name acac --env_name ovcA7 --seed 1 --total_epi 16 --eval_freq 8 --eval_num_epi 2`
- Working directory: `E:\myrepo\tg_acac_overcooked\acac`
- Progress: `data/log/audit_acac_smoke/ovcA7/s1_260922_112708/progress.csv`
  (2 rows at episodes 8 and 16; no NaN or Inf strings found).
- Checkpoint: `data/performance/audit_acac_smoke/ovcA7/ckpt/`
  (generic state, 3 actors, and 8 worker RNG states present; generic state
  loads with episode count 16 and 2 evaluation returns).
- Limitation: Unix `cp` failed during checkpoint history rotation on Windows.
  The latest `_1.tar` files exist, but historical `_2.tar` and later backups
  were not preserved. Rotation has since been changed to Python `copy2` and
  verified separately; no new training run has tested that change yet.

## Repository preparation (2026-09-22)

- Project Git branch: `main`
- Local initial commit: `430895ce604fb97a9994b72ed5e45ad58286ac1d`
- GitHub remote: `https://github.com/SSSiseven/acac.git` (configured locally; not pushed)
- Official ACAC source commit: `5ca6afb5e00f281c2be1f4cf98f3493c80714d4e`
- Tracked: 108 source and documentation files. Excluded: `data/`, bundled
  `icml25_result/`, local paper PDFs, and `upstream/acac-git-backup/`.

## Smoke resource estimate (2026-09-22)

- `ovcA7`, seed 1, 16 episodes: 3200 environment steps in 18.25 seconds
  according to `progress.csv`; sampling 11.88 seconds, training 1.74 seconds.
- Direct scaling to 100,000 episodes gives about 31.7 hours per seed. This is
  only a planning estimate because startup and evaluation schedules differ.
- One checkpoint generation is about 7.31 MB. Eleven versions would use
  roughly 80 MB per seed; reserve at least 0.5 GB per seed for run artifacts.
- GPU: NVIDIA GeForce GTX 1050 Ti, 4096 MiB; driver 581.80. Peak training
  VRAM was not captured by the smoke log. E: had about 591 GB free.
- Before full training, run a timed 200-episode calibration and record peak
  GPU memory. No long training was started during repository preparation.

## Timed ACAC calibration (user executed, 2026-09-22)

- Working directory: `E:\myrepo\tg_acac_overcooked\acac`.
- Command: `python scripts/acac_main.py --exp_name acac_calib200_s1 --alg_name acac --env_name ovcA7 --seed 1 --total_epi 200 --eval_freq 200 --eval_num_epi 20 --device cuda` (the supplied PowerShell command used the Conda environment Python executable).
- GPU sampling command: `nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 1 > ..\..\data\calibration\gpu_ovcA7_seed1.csv`.
- Progress: `data/log/acac_calib200_s1/ovcA7/s1_260922_123108/progress.csv`; complete rows at episodes 200 and 400, no NaN/Inf text. `variant.json` confirms the command flags and seed 1.
- Checkpoint: `data/performance/acac_calib200_s1/ovcA7/ckpt/`; 132 files, 80,480,650 bytes (76.75 MiB), including 11 rotated generations. Checkpoint rotation was exercised successfully on Windows.
- The run actually trained 400 episodes and 79,776 environment steps in 190.02 seconds. At episode 200 it had 39,879 steps, below the 40,000 step target. `total_timesteps` is only updated in `log_diagnostics()` at the 200-episode evaluation boundary, so the next stopping check occurred after episode 400. This is existing ACAC behavior, not a requested 400-episode budget.
- The GPU sampler overlaps the log file lifetime (11:31:08 to 11:34:20 local file time): 189 samples, 1,659 to 1,867 MiB total card memory, mean 1,797 MiB; peak utilization 62%, mean 8%. Pre-run sample was about 1,632 MiB. These are whole-card measurements, not process-attributed VRAM.
- Linear planning estimate at this setting: roughly 13.2 hours for 100,000 episodes on `ovcA7`, subject to run length, other system load, evaluation/checkpoint cost, and the stopping overshoot. Do not apply this estimate to other maps without calibration.
- `Eval/ReturnAverage` was -12.9167 at episode 200 and -12.7083 at 400. `Eval/NumDeliveryAverage` remained zero; the previously identified delivery-metric mapping bug means this field is not yet reliable.

## Seed 0 initialization fix (2026-09-22)

- Code: `acac/scripts/acac_main.py` uses `if args.seed is not None:`.
- Regression command: `C:\Users\705\.conda\envs\acac\python.exe -m unittest discover -s tests -p test_seed_zero.py` from the project root.
- Before fix: test failed, with the observed Python/NumPy/PyTorch first draws matching seed 123 rather than seed 0.
- After fix: one test passed. The test intercepts execution immediately after seeding, so it does not run training or create experiment outputs.
- Paper says each experiment used five random seeds. The released official result bundle has populated ACAC `0.csv` files for `ovcA7` and `ovcAR7`; the official README also explicitly shows `--seed=0`. This supports using seed 0 in the project's baseline protocol, while the paper text does not name each seed individually.

## Full ACAC ovcA7 baseline reproduction (2026-09-22 to 2026-09-28)

All commands were run from `E:\myrepo\tg_acac_overcooked\acac` with the
`acac` Conda environment. The three commands differed only in `--seed`:

`python scripts/acac_main.py --exp_name acac_ovcA7_baseline --alg_name acac --env_name ovcA7 --seed <0|1|2> --total_epi 100000 --eval_freq 200 --eval_num_epi 20 --device cuda`

| Seed | Log directory | Final episodes | Final timesteps | Final Eval/ReturnAverage | Total time |
|---:|---|---:|---:|---:|---:|
| 0 | `data/log/acac_ovcA7_baseline/ovcA7/s0_260922_163350` | 373,800 | 20,000,482 | 226.6000 | 19.47 h |
| 1 | `data/log/acac_ovcA7_baseline/ovcA7/s1_260923_120327` | 342,600 | 20,003,301 | 226.0957 | 18.77 h |
| 2 | `data/log/acac_ovcA7_baseline/ovcA7/s2_260927_140540` | 371,000 | 20,002,075 | 226.6000 | 19.85 h |

Validation:
- Every `variant.json` contains its expected seed and the requested settings.
- No NaN or Inf token occurs in any `progress.csv`.
- The final generic checkpoints contain matching episode counts and complete
  evaluation histories of 1,869, 1,713, and 1,855 entries.
- Three policy actors and the checkpoint set exist for each seed. The combined
  checkpoint directory is 284.53 MiB.
- The raw final three-seed return mean is approximately 226.432. Official
  released ACAC seed endpoints range from 225.8 to 226.6, so the reproduced
  endpoint is consistent with the release.
- `total_epi=100000` preserves official code semantics: a target of
  `200 * 100000 = 20,000,000` environment timesteps. The higher episode counts
  result from learned policies ending episodes early.

Aggregation command from the project root:

`C:\Users\705\.conda\envs\acac\python.exe analysis\plot_ovcA7_baseline.py --log-root data\log\acac_ovcA7_baseline\ovcA7 --output-dir data\analysis\acac_ovcA7_baseline`

Outputs:
- `data/analysis/acac_ovcA7_baseline/eval_return_three_seed.csv`
- `data/analysis/acac_ovcA7_baseline/eval_return_three_seed.png`
- `data/analysis/acac_ovcA7_baseline/run_summary.csv`

The aggregation interpolates each seed by environment timesteps on a 40,000-step
grid, then reports the mean, sample standard deviation, standard error, and a
Student-t 95% confidence interval for three seeds. It does not edit source CSVs.

## Stage 2 task-graph semantics (2026-09-28)

Implemented without connecting to training:
- `acac/acac_marl/task_graph/builder.py`: deterministic NumPy-only
  graph construction and feasibility masks;
- `acac/acac_marl/task_graph/v1_adapter.py`: leakage-free decoder for
  V1 macro observations;
- `tests/test_task_graph_builder.py`: all ten required state scenarios plus
  empty/complete plate and hidden-state validation;
- `tests/test_task_graph_v1_adapter.py`: real `ovcA7`, deterministic `ovcAR7`,
  and visible co-location decoding checks.

Validation command from the project root:

`C:\Users\705\.conda\envs\acac\python.exe -m unittest discover -s tests -p "test_task_graph*.py" -v`

Result: 15 tests passed in 0.029 seconds. The Gym deprecation message is emitted
by the unchanged dependency and is not a test failure. No training was run.

The real `ovcA7` reset check found executable tasks by agent:
- agent 0: `wait`;
- agent 1: `wait`;
- agent 2: `get:lettuce`, `get:onion`, `get:tomato`, `wait`.

This demonstrates that the documented hard visibility mask plus the initial six
task types omits the original directional exploration actions and can deadlock.
Neural integration is paused pending an explicit semantic resolution.

## ACAC source-tree flattening (2026-09-29)

The former three-level repeated hierarchy was flattened to `acac/`. The
single directory now contains `__init__.py`, `acac_marl/`, `env/`,
`scripts/`, `setup.py`, the upstream README, and license notices.

Validation from the project root:

`C:\Users\705\.conda\envs\acac\python.exe -m unittest discover -s tests -v`

Result: all 16 tests passed. The Gym deprecation message is emitted by the
unchanged dependency and is not a test failure.

Import and project-root validation:

`C:\Users\705\.conda\envs\acac\python.exe -c "import sys; sys.path.insert(0, r'E:\myrepo\tg_acac_overcooked'); import acac; from acac.acac_marl import PROJECT_DIR; print(acac.__file__); print(PROJECT_DIR)"`

Observed paths:

- `E:\myrepo\tg_acac_overcooked\acac\__init__.py`
- `E:\myrepo\tg_acac_overcooked\acac`

`setup.py --name` also completed and returned `acac_marl`. No training or
experiment-output modification was performed.


## Stage 2 Explore resolution (record updated 2026-10-08)

The user-approved Explore implementation has one task type and four
directional instances. The current graph has 23 task nodes, 20 dependency
edges, and 28-wide task features. Static walkable cells are passed explicitly
through TaskGraphMetadata; the existing V1 observation decoder is unchanged.

Successful validation command from the project root:

`C:\Users\705\.conda\envs\acac\python.exe -m unittest discover -s tests -p "test_task_graph*.py" -v`

Previous validation: all 18 task-graph tests passed in 0.032 seconds.
Initial failed validation had five errors from a static test helper referring
to self, and one failure from checking robot row 0 for all perspectives.
Both test issues were fixed before the successful rerun.

Real deterministic ovcA7 reset executable tasks:

- agent 0: explore:right, explore:down, explore:left, wait;
- agent 1: explore:right, explore:left, explore:up, wait;
- agent 2: all three visible GetIngredient tasks plus
  explore:right, explore:down, explore:left, wait.

All four direction nodes remain present; masks select a feasible subset.
No training or experiment-output modification was performed.


## Stage 3 assembly preflight (2026-10-08)

Successful command from the project root:

`C:\Users\705\.conda\envs\acac\python.exe analysis/audit_task_graph_assembly.py`

The diagnostic builds fixed V1 states, refreshes original observations,
decodes local macro observations, and compares the builder mask with original
macro execution. Both chopped-food retrieval and partial-plate addition had
task_mask=false and original_macro_succeeded=true.
Output: data/audit/stage3_preflight.json (no training performed).

First diagnostic attempt did not refresh the primitive observation cache after
fixture placement: pickup falsely appeared to fail. Calling _get_obs before
capturing the macro observation fixed the fixture; both original macros then
succeeded. The reproduction script contains this refresh.


## Retrieval and assembly resolution (2026-10-08)

Successful command from the project root:

`C:\Users\705\.conda\envs\acac\python.exe -m unittest discover -s tests -v`

Result: 24 tests passed in 0.133 seconds. Tests cover collection visibility,
chopped status, holding/containment, plate occupancy, duplicate ingredients,
busy robots, partial-plate pickup, and direct collection into a held plate.
The original V1 fixed-state cooperative chain reached successful delivery;
every graph task selected in the chain passed its local feasibility mask.

Failed full-chain attempts retained in this record: the first assumed that
get-plate stops immediately after depositing food; original execution also
picks up the plate. The test was corrected to a cooperative cook/plate-holder
chain. The next attempted GoToKnife from outside local visibility; adding
an Explore step to reveal the knife corrected the fixture plan.

No training was run. Existing experiment CSVs and preflight evidence were
not modified. Graph dimensions are now 32 nodes, 32 edges, and width 30.


## Stage 3 TG-ACAC smoke validation (2026-10-08)

All commands below ran from the project root with the acac Conda Python.
Set OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1 before running.
All 32 unit tests passed in 0.272 seconds. No raw CSV was modified.

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
C:\Users\705\.conda\envs\acac\python.exe -m unittest discover -s tests -v
C:\Users\705\.conda\envs\acac\python.exe acac/scripts/tg_acac_main.py --alg_name tg_acac_no_dependency --exp_name tg_nodep_calib8 --env_name ovcA7 --seed 0 --total_epi 8 --n_env 4 --train_freq 4 --eval_freq 8 --eval_num_epi 2 --device cpu
C:\Users\705\.conda\envs\acac\python.exe acac/scripts/tg_acac_main.py --alg_name tg_acac_no_dependency --exp_name tg_nodep_smoke100 --env_name ovcA7 --seed 0 --total_epi 100 --n_env 4 --train_freq 4 --eval_freq 20 --eval_num_epi 4 --device cpu
C:\Users\705\.conda\envs\acac\python.exe acac/scripts/tg_acac_main.py --alg_name tg_acac --exp_name tg_full_smoke100 --env_name ovcA7 --seed 0 --total_epi 100 --n_env 4 --train_freq 4 --eval_freq 20 --eval_num_epi 4 --device cpu
C:\Users\705\.conda\envs\acac\python.exe acac/scripts/acac_main.py --alg_name acac --exp_name acac_stage3_regression8 --env_name ovcA7 --seed 0 --total_epi 8 --n_env 4 --train_freq 4 --eval_freq 8 --eval_num_epi 2 --device cpu
C:\Users\705\.conda\envs\acac\python.exe analysis/verify_stage3_smoke.py tg_nodep_smoke100 tg_full_smoke100 tg_nodep_calib8 acac_stage3_regression8
```

| Experiment | Episodes | Steps | Training/eval seconds | Checkpoint MiB |
|---|---:|---:|---:|---:|
| tg_nodep_smoke100 | 100 | 20000 | 204.08 | 63.37 |
| tg_full_smoke100 | 120 | 23906 | 254.41 | 78.46 |
| tg_nodep_calib8 | 8 | 1600 | 22.70 | 14.08 |
| acac_stage3_regression8 | 8 | 1600 | 10.05 | 13.89 |

Time is the original logger TotalTime; process setup and final checkpoint
overhead are excluded. Disk totals count checkpoints only, excluding policies
and logs. Full TG had 19,906 steps at the 100-episode evaluation point and
23,906 at the next point, so it completed 120 episodes before stopping.
This preserves original step-budget accounting.

Configurations and raw progress paths:
- data\log\tg_nodep_smoke100\ovcA7\s0_261008_121701\progress.csv (variant.json beside it)
- data\log\tg_full_smoke100\ovcA7\s0_261008_122350\progress.csv (variant.json beside it)
- data\log\tg_nodep_calib8\ovcA7\s0_261008_121538\progress.csv (variant.json beside it)
- data\log\acac_stage3_regression8\ovcA7\s0_261008_123013\progress.csv (variant.json beside it)

Complete stdout/stderr: data/stage3/<experiment>.log.
Checkpoints: data/performance/<experiment>/ovcA7/ckpt/.
Policies: data/policy_nns/<experiment>/ovcA7/.
Verification report: data/stage3/verified_smokes.json.
Original-source reuse audit hashes: data/stage3/reuse_audit.json.

CSV values, network tensors, and optimizer state tensors are all finite.
Actual TG actor checkpoints restore and produce finite outputs. Early
NoDependency checkpoints lack the later static dependency_edge_index buffer;
loading reconstructs only this immutable index from metadata, not weights.

No training run failed. One initial dependency test failed because initial
feasible Explore/Wait tasks have no incoming edges. Changing the test to a
held-food state with feasible GoToKnife confirmed probability changes and
finite nonzero dependency gradients.

CPU smoke tests establish runnable training, not effectiveness or GPU
requirements. Unsupported map B, V0 scale, and obs_last_action=True are
guarded explicitly. No pilot, formal training, or git push was performed.

## Stage 4: ACAC-MLP and profiling (2026-10-08)

Repository-root commands, existing acac Conda environment:

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
python -m unittest discover -s tests -v
python analysis/profile_stage4.py --train
```

Initial invocation completed inference, then rejected an extra max_checkpoint
CLI argument before training. Failure evidence: data/stage4/failures/.
After correcting the profiling command, this successful command reused the
inference evidence and completed all four timing runs:

```powershell
python analysis/profile_stage4.py --train --training-only
```

Exact successful training commands:

```powershell
python acac/scripts/acac_main.py --alg_name acac --exp_name stage4_acac_timing16 --env_name ovcAR7 --seed 0 --total_epi 16 --n_env 4 --train_freq 4 --eval_freq 8 --eval_num_epi 2 --device cpu
python acac/scripts/acac_main.py --alg_name acac_mlp --exp_name stage4_acac_mlp_timing16 --env_name ovcAR7 --seed 0 --total_epi 16 --n_env 4 --train_freq 4 --eval_freq 8 --eval_num_epi 2 --device cpu
python acac/scripts/acac_main.py --alg_name tg_acac_no_dependency --exp_name stage4_tg_acac_no_dependency_timing16 --env_name ovcAR7 --seed 0 --total_epi 16 --n_env 4 --train_freq 4 --eval_freq 8 --eval_num_epi 2 --device cpu
python acac/scripts/acac_main.py --alg_name tg_acac --exp_name stage4_tg_acac_timing16 --env_name ovcAR7 --seed 0 --total_epi 16 --n_env 4 --train_freq 4 --eval_freq 8 --eval_num_epi 2 --device cpu
```

All four: 16 episodes / 3,200 steps; finite metrics and network/optimizer
tensors; checkpoints restore. Original CSVs unchanged. 35 unit tests passed.

| Method | Train steps/s | Process wall s | Checkpoints MiB |
|---|---:|---:|---:|
| acac | 237.6 | 21.93 | 27.77 |
| acac_mlp | 233.2 | 22.26 | 28.40 |
| tg_acac_no_dependency | 98.0 | 47.24 | 28.17 |
| tg_acac | 91.6 | 50.51 | 28.53 |

Full tables and scope: analysis/STAGE4_REPORT.md.
Configurations, exact interpreter commands and CSV paths: data/stage4/profile.json.
Derived results: data/stage4/summary.csv. Source hashes: reuse_audit.json.
Run stdout/stderr: data/stage4/stage4_*_timing16.log.
Checkpoints: data/performance/stage4_<method>_timing16/ovcAR7/ckpt/.

Fresh independent repeat (existing experiment names are rejected):

```powershell
python analysis/profile_stage4.py --train --run-prefix stage4_repeat1 --output-dir data/stage4_repeat1
```

CPU uses one thread; GPU measurements are inference only. Training rate excludes
evaluation/saving; process wall includes them. No learning claim, pilot, formal
training or git push. Pilot metric verification/resource estimates remain pending.
