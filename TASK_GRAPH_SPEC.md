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