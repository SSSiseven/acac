# Pilot preflight: metrics, delivery repair and resources

Date: 2026-10-09. Pilot remains NOT STARTED.

## Completed and verified

- Opt-in local-feasibility, contention, Wait, inference and delivery diagnostics.
- TG detached pre-mask invalid probability mass; raw counts for pooled rates.
- Read-only 12-run/three-seed common-support AUC summarizer.
- 45 tests pass; observations, rewards, done and RNG trajectory unchanged.
- Existing baseline replay: delivery [0,0] -> [1,1], return [226.6,226.6]
  unchanged, as are lengths and discounted returns. Old files untouched.
- Eight CPU/CUDA short runs finish; new primary metrics finite and in [0,1].
- All checkpoints/optimizer tensors finite and network weights restore.
- Original models, critic/GAE/PPO learner, macros and learn loop unchanged.

## Metric definitions

See EXPERIMENT_PROTOCOL.md for denominators and logging names.
Selected-macro feasibility uses the same local task contract for every method.
TG pre-mask mass is network softmax before masking and epsilon; it is TG-only.
Target contention is not physical collision; IdleRatio measures explicit Wait,
not every stall. Mask-induced zero invalid rates alone do not prove learning.
Evaluation pooled ratios weight counts by actual episode counts.

## Hardware and short calibration

Intel Core i5-9400, 6 cores; 15.9 GiB RAM; GTX 1050 Ti 4 GiB.
Disk free at calibration start: 506.45 GiB. One OMP/MKL/PyTorch CPU thread.
n_env=8, train_freq=8, target 6,400 steps, eval_freq=32, eval_num_epi=20.
Diagnostics enabled for all methods. Original budget accounting retained;
CPU MLP reaches 12,797 steps because first interval finished three steps below
the target and the next full interval ran. Rates use actual steps.

| Method | Device | Train steps/s | Sampling s | PPO s | GPU allocated/reserved MiB | RSS sum GiB | Output MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| acac | cpu | 227.8 | 21.0 | 7.1 | N/A | 3.59 | 14.1 |
| acac_mlp | cpu | 271.5 | 33.0 | 14.1 | N/A | 3.60 | 43.0 |
| tg_acac_no_dependency | cpu | 33.0 | 49.1 | 145.0 | N/A | 3.63 | 33.3 |
| tg_acac | cpu | 31.7 | 50.8 | 151.0 | N/A | 3.65 | 33.5 |
| acac | cuda | 240.2 | 22.8 | 3.8 | 74.0/94.0 | 4.13 | 14.1 |
| acac_mlp | cuda | 247.4 | 22.4 | 3.5 | 74.9/94.0 | 4.15 | 14.5 |
| tg_acac_no_dependency | cuda | 31.3 | 67.6 | 137.1 | 92.6/128.0 | 4.22 | 33.3 |
| tg_acac | cuda | 30.4 | 69.8 | 140.5 | 115.1/150.0 | 4.21 | 33.5 |

RSS is summed across parent and eight workers and can double-count shared
pages. CUDA allocator peaks exclude driver/context/display memory; full TG
reported 3,163 MiB device-free memory at the end. This is a short-run observation.

## Proposed pilot budget and time estimates

25% of formal 40M steps = 10M steps per method/seed (total_epi=50000).
Four methods, seeds 0/1/2: 12 runs. Use n_env=8/train_freq=8/eval_freq=200/
eval_num_epi=20 and one common device. Choose final checkpoint consistently.

| Method | Device | Training hours | Eval proxy hours | Point total hours | Planning range hours |
|---|---|---:|---:|---:|---:|
| acac | cpu | 12.2 | 0.7 | 12.9 | 10-26 |
| acac_mlp | cpu | 10.2 | 0.7 | 10.9 | 8-22 |
| tg_acac_no_dependency | cpu | 84.3 | 2.0 | 86.2 | 65-172 |
| tg_acac | cpu | 87.6 | 2.0 | 89.6 | 67-179 |
| acac | cuda | 11.6 | 0.7 | 12.3 | 9-25 |
| acac_mlp | cuda | 11.2 | 0.7 | 11.9 | 9-24 |
| tg_acac_no_dependency | cuda | 88.8 | 2.9 | 91.8 | 69-184 |
| tg_acac | cuda | 91.3 | 2.8 | 94.1 | 71-188 |

Serial 12-run point estimate: CPU 599 hours (24.9 days);
CUDA 630 hours (26.3 days).
Planning range 0.75x-2x is a resource allowance, NOT a confidence interval.
Rates can change substantially as episode length and ready-decision counts change.
Evaluation proxy is EpochTime minus SamplingTime/TrainingTime per logged interval;
it includes other interval work and is not a separately instrumented eval timer.
Startup and checkpoint I/O variations are not precisely extrapolated.

## Disk, memory and recommendation

- Reserve 1 GiB per seed, 12 GiB for pilot outputs; 24 GiB with working headroom.
- Original checkpoint max_save is 11; best-policy files overwrite. Return histories
  and logs grow; do not scale short output size linearly with number of episodes.
- Observed RSS sum is around 3.6-4.2 GiB per run. Current 16 GiB machine supports
  a single calibration run. Prefer serial runs on this six-core machine.
- Observed CUDA peaks fit 4 GiB. Keep at least 1 GiB free device memory as
  working headroom; no claim that 115 MiB captures all GPU process memory.
- Current GPU is not a remedy for TG runtime. PPO is the dominant measured cost.
- Do not start the 12 long runs yet. Profile graph preparation/repeated rebuilding
  and cache use in TG PPO first, then recalibrate without changing rewards,
  encoder, PPO, budgets or evaluation. CPU is the current common-device preference
  from these short rates; that preference is not a precise hardware ranking.

## Evidence and known limits

- data/pilot_preflight/calibration.json: commands, paths, metrics and raw resources.
- data/pilot_preflight/resource_estimates.json: estimates and assumptions.
- data/pilot_preflight/delivery_replay.json: old weights with diagnostics on/off.
- data/pilot_preflight/source_reuse.json: unchanged source audit.
- data/pilot_preflight/failures/: empty-plate and logger-filter diagnostics.
- Original failed/incomplete CSVs and logs retained at pilotcal*/ directories.
- analysis/pilot_plan.json: proposed plan; no launch script has been executed.
- analysis/summarize_pilot.py: runs only on 12 completed matching pilot runs.

Some early CPU calibration files precede the final EpisodeCount/pre-mask-sum
logging addition. All primary rates exist; latest CUDA runs verify the final
counter schema. AUC output is tested with fixtures, not actual pilot results.
V1 map A diagnostics only; V0 scale/map B graph coverage remains pending.
No historical CSV edits, git commit or git push.
