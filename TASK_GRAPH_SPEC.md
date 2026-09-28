# Task Graph Specification

## Robot nodes

Features:
- agent_id
- position or observable relative position
- held_object
- ready/busy
- current_macro_action
- elapsed_duration
- observable local entity state
- recurrent history embedding由ACAC网络内部产生，不在环境中伪造

## Task nodes

Initial task types:
- GetIngredient(object_id)
- GetPlate(plate_id)
- GoToKnife(object_id, knife_id)
- Chop(object_id, knife_id)
- Deliver(recipe_id)
- Wait(agent_id)

每个任务节点包含：
- task_id
- task_type
- target_entity_id
- status
- prerequisite_status
- resource_id
- observable location
- estimated distance
- valid/invalid

## Task dependency edges

Examples:
- GetIngredient → GoToKnife
- GoToKnife → Chop
- Chop(required ingredients) → Deliver
- GetPlate → Deliver

实际依赖必须根据当前Overcooked代码的食谱和执行逻辑核验。

## Robot-task edges

Features:
- observable distance
- executable
- target visible
- object compatibility
- resource occupancy
- estimated macro duration

## Feasibility mask

A robot-task pair is invalid when:
- prerequisite未满足；
- 机器人正忙且当前时刻不需要重新决策；
- 机器人持有不兼容物体；
- 目标不存在或已经完成；
- 必需资源不可用；
- 任务不在机器人的局部可观测范围且原环境不允许知道它；
- 相同任务已被互斥占用；
- episode状态下该任务没有意义。

## Output interface

TaskGraphObservation:
- robot_features
- robot_mask
- task_features
- task_mask
- robot_task_edge_features
- robot_task_mask
- task_dependency_edge_index
- task_dependency_edge_features
- task_to_macro_action
## Audited V1 semantics for Stage 2

For `ovcA7` and `ovcAR7`, the actual V1 macro-action table is:

`stay`, `get tomato`, `get lettuce`, `get onion`, `get plate 1`,
`get plate 2`, `go to knife 1`, `go to knife 2`, `deliver`, `chop`,
`right`, `down`, `left`, `up`.

`chop` acts only on an adjacent knife and does not navigate. `go to knife N`
navigates while an agent holds an ingredient. `deliver` is meaningful only for
an agent holding a plate whose contents exactly match the recipe. A busy agent
keeps executing its current macro action and must not choose another task.

The V1 macro observation lists each item in a fixed slot. An entity outside the
agent's radius is represented by zero coordinates and zero dynamic state. The
observation does not expose a hidden entity's current position, holder,
container, knife occupancy, or chop progress. Co-located visible slots are
sufficient to infer a held ingredient, plate contents, and knife occupancy.

## Concrete pure-builder contract

`TaskGraphBuilder` consumes immutable `RobotObservation`, `EntityObservation`,
and `TaskGraphMetadata` values. It does not receive an environment object.
Hidden robots and entities must have no position or dynamic state; invalid
inputs are rejected. `decode_v1_macro_observation` produces this contract from
one agent's raw V1 macro observation and that agent's own ready/current-action
state.

For the six-entry ingredient vocabulary used by the current public scope:

- robot feature width: 16 (10 base fields plus held-ingredient vocabulary);
- task feature width: 23 (task type, ingredient, status, visibility,
  prerequisite/resource state, location, progress, and stable ordinals);
- robot-task edge width: 8;
- dependency-edge width: 3.

For the three-ingredient V1 recipe with two plates and two knives, the builder
creates 19 nodes: three GetIngredient, two GetPlate, six GoToKnife, six Chop,
one Deliver, and one Wait. It creates 20 dependency edges. Completed tasks stay
in the graph but cannot be selected. Busy robots have no selectable task. Wait
is the fallback for a ready robot when no productive task is feasible.

## Unresolved exploration conflict

The hard-mask rule says a target outside local observation is infeasible. The
initial six task types contain no task mapped to the environment's four
primitive movement macros. On the real `ovcA7` reset, the leakage-free builder
finds:

- agent 0 sees delivery and both knives: only Wait is executable;
- agent 1 sees delivery: only Wait is executable;
- agent 2 sees the three ingredients: GetIngredient and Wait are executable.

After agent 2 obtains an ingredient, a knife can remain outside its radius, so
GoToKnife is masked and Wait cannot change the observation. This can deadlock
before any Chop or Deliver task. Training integration must remain paused until
the task vocabulary covers exploration, or the visibility constraint is
changed explicitly. Adding four directional Explore/Move nodes mapped to
`right/down/left/up` is the smallest leakage-free option; allowing hidden target
positions would violate the actor observation boundary.
