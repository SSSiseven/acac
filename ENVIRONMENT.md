# Environment Contract

## Environment

gym-macro-overcooked through the ACAC repository.

## Official environment names

- ovcA7
- ovcAR7
- ovcAR11_N6

## Existing macro actions

需要Codex从实际代码中核验并填写：
- action index
- action name
- initiation condition
- target object/resource
- termination condition
- failure condition

## State entities

需要从代码核验：
- agents
- tomato/lettuce/onion
- plates
- knives
- delivery station
- current recipe
- object status
- agent holding state

## Constraints

1. 不改变原始奖励。
2. 不改变原始primitive dynamics。
3. 不改变原始宏动作执行器。
4. 任务图必须由当前环境状态确定性构造。
5. 任务图不能读取智能体本来不可观察的状态。