# Stage 4: parameter matching and runtime measurements

Date: 2026-10-08. Environment: ovcAR7, three robots, V1 map A.
CPU inference/training: one PyTorch/OMP/MKL thread.
GPU inference: GTX 1050 Ti, PyTorch 2.1.0, CUDA 12.1.

## Completed and verified

- Separate acac_mlp algorithm; only post-GRU head expanded to width 126.
- Full TG matching: 14,700 vs 14,688 trainable parameters per robot.
- Actor mismatch 0.0817%; combined actor+critic mismatch 0.0108%.
- 35 tests passed: preserved encoder, macro output, no graph forward, gradients.
- Four runs finish 16 episodes / 3,200 steps with identical short-run settings.
- All CSV and checkpoint/optimizer tensors finite; checkpoints restore.
- Original models, memory, runner, environment and PPO/GAE hashes unchanged.

## Trainable parameters

All three online actors and one online centralized critic. Unique parameters
counted once; target copies, optimizers and frozen sinusoidal embeddings excluded.

| Method | Actor | Critic | Total | Including frozen |
|---|---:|---:|---:|---:|
| acac | 30,282 | 290,497 | 320,779 | 322,423 |
| acac_mlp | 44,100 | 290,497 | 334,597 | 336,241 |
| tg_acac_no_dependency | 37,440 | 290,497 | 327,937 | 329,581 |
| tg_acac | 44,064 | 290,497 | 334,561 | 336,205 |

NoDependency omits the dependency module. MLP targets full TG; no dummy parameters.

## Inference latency

Median milliseconds per joint decision for three ready robots, batch=1.
Common 64 local observations from original random-macro rollout, seed 17;
five repeats, 320 samples, timing index=1, epsilon=0.05.
Includes GRU, head, masking, sampling, graph/cache work and device transfers.
CUDA synchronized. Cache clear excluded; cold includes rebuilding graphs.
Warm primed with entire pool. These are controlled decisions, not trained-policy
ready/busy frequencies. Means and p95 values are in profile.json.

| Method | CPU warm ms | CPU cold ms | GPU warm ms | GPU cold ms |
|---|---:|---:|---:|---:|
| acac | 2.086 | 2.076 | 4.283 | 4.216 |
| acac_mlp | 2.090 | 2.125 | 4.301 | 4.285 |
| tg_acac_no_dependency | 3.135 | 8.640 | 12.874 | 19.601 |
| tg_acac | 3.764 | 9.238 | 13.493 | 20.557 |

Small-batch GPU latency exceeds CPU on this device; GPU training is unverified.

## CPU short training throughput

Seed 0; target 16*200 steps; n_env=4; train_freq=4; eval_freq=8;
eval_num_epi=2; original checkpoint saving and five PPO repeats.
Training rate = actual steps / sum(SamplingTime+TrainingTime).
Wall rate includes startup, evaluation and saves. Original CSVs are unchanged.

| Method | Sampling s | PPO s | Wall s | Train steps/s | Wall steps/s | Checkpoints MiB |
|---|---:|---:|---:|---:|---:|---:|
| acac | 9.07 | 4.40 | 21.93 | 237.6 | 145.9 | 27.77 |
| acac_mlp | 8.96 | 4.77 | 22.26 | 233.2 | 143.8 | 28.40 |
| tg_acac_no_dependency | 22.25 | 10.42 | 47.24 | 98.0 | 67.7 | 28.17 |
| tg_acac | 23.19 | 11.76 | 50.51 | 91.6 | 63.4 | 28.53 |

One short timing run per method; rates vary with hardware, episode length and
ready-decision frequency. No effectiveness conclusion. Full TG is approximately
2.59 times slower than ACAC by the training-rate measure.

## Pending and limitations

- GPU training throughput and peak training memory remain unverified.
- Verify pilot task/conflict/idle metrics and delivery logger before Stage 5.
- Estimate time/disk/GPU needs before a run above two hours.
- Current TG/MLP matching supports V1 map A; map B and V0 scale remain pending.
- No pilot, formal run, commit or git push.

## Evidence

- analysis/profile_stage4.py: unified profiling and checkpoint verification.
- data/stage4/profile.json: full config, commands, paths and timing results.
- data/stage4/summary.csv: derived summary; source experiment CSVs untouched.
- data/stage4/reuse_audit.json: unchanged source hashes.
- data/stage4/failures/: retained test/CLI diagnostics.

The first command was rejected before training: original CLI has no
--max_checkpoint option. Removed the profiling option; saved diagnostics.
An initial test expected width 131 from a hand estimate; actual nearest width
is 126. Fixed only the test expectation. Checkpoint verifier restores online
and target actors from the saved online weights, as the original loader does.
