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
