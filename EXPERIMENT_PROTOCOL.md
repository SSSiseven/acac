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