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
