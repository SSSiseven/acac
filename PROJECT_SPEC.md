# Project Specification

## Research question

在宏动作异步完成、环境配置随机变化、任务具有前置依赖和资源冲突的条件下，
显式任务图是否能够帮助ACAC更有效地选择任务，并改善样本效率、最终回报和泛化能力？

## Baseline

Agent-Centric Actor-Critic, ICML 2025。

## Proposed method

TG-ACAC = ACAC异步学习机制
        + 动态任务节点集合
        + 机器人—任务关系编码
        + 任务依赖边
        + ready-conditioned task pointer
        + feasibility mask

## Central comparison

- ACAC
- ACAC-MLP：参数量匹配
- TG-ACAC-NoDependency
- TG-ACAC

## Primary environment

- 调试：ovcA7
- Pilot：ovcAR7
- 扩展：ovcAR11_N6

## Success criteria

TG-ACAC至少满足：
1. 多个seed上训练稳定；
2. 相对ACAC提高学习曲线AUC或最终回报；
3. 降低无效任务、资源冲突或重复选择；
4. 去掉依赖边后性能下降；
5. 优势不是单纯来自参数量增加。