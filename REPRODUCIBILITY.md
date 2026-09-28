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
- Working directory: `E:\myrepo\tg_acac_overcooked\acac\acac`
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

- Working directory: `E:\myrepo\tg_acac_overcooked\acac\acac`.
- Command: `python scripts/acac_main.py --exp_name acac_calib200_s1 --alg_name acac --env_name ovcA7 --seed 1 --total_epi 200 --eval_freq 200 --eval_num_epi 20 --device cuda` (the supplied PowerShell command used the Conda environment Python executable).
- GPU sampling command: `nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 1 > ..\..\data\calibration\gpu_ovcA7_seed1.csv`.
- Progress: `data/log/acac_calib200_s1/ovcA7/s1_260922_123108/progress.csv`; complete rows at episodes 200 and 400, no NaN/Inf text. `variant.json` confirms the command flags and seed 1.
- Checkpoint: `data/performance/acac_calib200_s1/ovcA7/ckpt/`; 132 files, 80,480,650 bytes (76.75 MiB), including 11 rotated generations. Checkpoint rotation was exercised successfully on Windows.
- The run actually trained 400 episodes and 79,776 environment steps in 190.02 seconds. At episode 200 it had 39,879 steps, below the 40,000 step target. `total_timesteps` is only updated in `log_diagnostics()` at the 200-episode evaluation boundary, so the next stopping check occurred after episode 400. This is existing ACAC behavior, not a requested 400-episode budget.
- The GPU sampler overlaps the log file lifetime (11:31:08 to 11:34:20 local file time): 189 samples, 1,659 to 1,867 MiB total card memory, mean 1,797 MiB; peak utilization 62%, mean 8%. Pre-run sample was about 1,632 MiB. These are whole-card measurements, not process-attributed VRAM.
- Linear planning estimate at this setting: roughly 13.2 hours for 100,000 episodes on `ovcA7`, subject to run length, other system load, evaluation/checkpoint cost, and the stopping overshoot. Do not apply this estimate to other maps without calibration.
- `Eval/ReturnAverage` was -12.9167 at episode 200 and -12.7083 at 400. `Eval/NumDeliveryAverage` remained zero; the previously identified delivery-metric mapping bug means this field is not yet reliable.

## Seed 0 initialization fix (2026-09-22)

- Code: `acac/acac/scripts/acac_main.py` uses `if args.seed is not None:`.
- Regression command: `C:\Users\705\.conda\envs\acac\python.exe -m unittest discover -s tests -p test_seed_zero.py` from the project root.
- Before fix: test failed, with the observed Python/NumPy/PyTorch first draws matching seed 123 rather than seed 0.
- After fix: one test passed. The test intercepts execution immediately after seeding, so it does not run training or create experiment outputs.
- Paper says each experiment used five random seeds. The released official result bundle has populated ACAC `0.csv` files for `ovcA7` and `ovcAR7`; the official README also explicitly shows `--seed=0`. This supports using seed 0 in the project's baseline protocol, while the paper text does not name each seed individually.

## Full ACAC ovcA7 baseline reproduction (2026-09-22 to 2026-09-28)

All commands were run from `E:\myrepo\tg_acac_overcooked\acac\acac` with the
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
- `acac/acac/acac/acac_marl/task_graph/builder.py`: deterministic NumPy-only
  graph construction and feasibility masks;
- `acac/acac/acac/acac_marl/task_graph/v1_adapter.py`: leakage-free decoder for
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
