# Experiment Protocol

## Stage E0

复现原始ACAC：
- env: ovcA7
- seeds: 0, 1, 2
- output: progress.csv、均值曲线、环境版本

## Stage P0: Graph correctness

固定状态构造任务图，检查：
- 节点数量
- dependency
- feasible mask
- task-to-action mapping
- 图随状态变化
- busy agent行为

## Stage P1: Smoke test

每个方法：
- 1 seed
- 10至100 episodes
- 检查NaN、shape、mask、梯度和日志

## Stage P2: Pilot

Environment:
- ovcAR7

Methods:
- ACAC
- ACAC-MLP
- TG-ACAC-NoDependency
- TG-ACAC

Seeds:
- 0, 1, 2

Budget:
- 正式训练预算的20%至30%

## Pilot continuation criteria

满足以下条件才进入正式实验：
- 无NaN和死锁；
- TG-ACAC至少2个seed表现出学习；
- TG-ACAC相对ACAC在AUC、最终return或无效任务率中至少一项稳定改善；
- 图构造耗时可接受；
- dependency消融产生合理变化。

## Stage P3: Final experiments

Environments:
- ovcAR7
- ovcBR7
- ovcAR11_N6

Seeds:
- 至少5个

Metrics:
- Eval/ReturnAverage
- Eval/NumDeliveryAverage
- Eval/EpiLenAverage
- learning-curve AUC
- invalid task proposal rate
- resource conflict rate
- idle ratio
- inference latency
- parameter count

## Pilot metric definitions and diagnostic switches (2026-10-09)

Pilot planning budget: 25% of ovcAR7 formal 40M environment steps:
total_epi=50000, env_terminate_step=200, target=10M steps.
Preserve n_env=8, train_freq=8, eval_freq=200, eval_num_epi=20 and seeds 0/1/2.
total_epi is the original step-budget multiplier, not an exact episode count.
Original evaluation-interval budget updates and any overshoot remain recorded.

record_pilot_metrics=true enables evaluator-only diagnostics and corrected
delivery counting. record_delivery_metrics=true can enable delivery alone.
Both default false for historical runs. profile_resources=true enables peak
CUDA allocator reporting. New switches accept explicit true/false values.

- NumDelivery: number of successful V1 deliveries during an episode. Record
  the original successful termination and matching dish on the delivery shelf.
  Wrong dishes, timeouts and ordinary steps count zero. Do not infer from reward.
- InvalidTaskProposalRate: among ready decisions, fraction of selected macros
  with no feasible task mapping under the SAME locally observed task graph.
  ACAC macros that map to several task types are valid if ANY mapping is feasible.
  This is observable initiation feasibility, not eventual macro execution success.
  Busy continuation steps are excluded; invisible targets are infeasible.
- PreMaskInvalidTaskMass (TG variants only): raw task-score softmax probability
  assigned to infeasible task nodes BEFORE masking and BEFORE epsilon mixing.
  It is diagnostic only, detached, and never modifies sampling or PPO gradients.
  ACAC/MLP lack task-node logits, so this metric is N/A for them.
  Masked TG has structurally low selected-action invalid rate; that alone cannot
  establish learned graph reasoning or justify proceeding to final experiments.
- ResourceConflictRate: conflicted resource-directed robot-steps / all
  resource-directed robot-steps. A conflict means at least two active macros
  target the same named knife or plate, including busy continuations.
  GoToKnife/GetPlate use named resources; Chop resolves the original direction
  order; food collection identifies its occupied knife/plate. Nonresource moves
  and Wait are excluded. This is target contention, not physical collision.
  Centralized resource state is used only in evaluator accounting, never Actor.
- IdleRatio: Wait macro robot-steps / all robot-steps. Busy moving/chopping is
  not idle. This does not count every physical stall or ineffective interaction.
- InferenceTimeMs: synchronized controller wall milliseconds divided across
  active environment slots, then episode-averaged per joint environment step.
  Includes busy slots, Actor graph work, masks and sampling; excludes evaluator
  worker diagnostics. Ready frequency differs by policy. Stage 4 controlled
  three-ready-robot cold/warm inference remains a separate measurement.

Log raw counters (DecisionCount, InvalidProposalCount, AgentSteps, IdleAgentSteps,
ResourceAgentSteps, ConflictAgentSteps), actual EpisodeCount, and TG pre-mask
mass sum/decision count. Offline pooled rates weight interval means by the
actual number of eval episodes, rather than averaging episode percentages.
An empty denominator is reported as zero with its zero count visible.

AUC integrates Eval/ReturnAverage against Timesteps over shared observed
support: start=max(first eval timestep across all 12 runs), end=10M target.
Linearly interpolate only between recorded points; never invent a return at
step zero or extrapolate. Report raw AUC and AUC/(end-start).
Require all four methods and seeds 0/1/2, unique runs and matching settings.
Final-return summaries use the final logged evaluation/checkpoint of each run
(first evaluation interval reaching the budget); retain actual overshoot.
Existing best-policy exports are not substituted for these final checkpoints.

No historical CSVs are corrected in place. Old zero delivery fields remain
identified as missing-event statistics. Pilot continuation still requires
multi-seed learning and evidence beyond a mechanically enforced mask.
