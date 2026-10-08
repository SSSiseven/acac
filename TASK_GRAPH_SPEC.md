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
- CollectChoppedIngredient(object_id)
- AddToPlate(object_id, plate_id)
- Deliver(recipe_id)
- Explore(direction): one type, four directional node instances
- Wait

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
- task feature width: 30 (nine task types, ingredient, status, visibility,
  prerequisite/resource state, location, progress, stable ordinals, and
  a four-way direction encoding);
- robot-task edge width: 8;
- dependency-edge width: 3.

For the three-ingredient V1 recipe with two plates and two knives, the builder
creates 32 nodes: three GetIngredient, two GetPlate, six GoToKnife, six Chop,
three CollectChoppedIngredient, six AddToPlate, one Deliver, four Explore
instances, and one Wait. It creates 32 dependency edges. Completed tasks stay
in the graph but cannot be selected. Busy robots have no selectable task. Wait
is the fallback for a ready robot when no productive task is feasible.

## Resolved exploration semantics

Explore is one task type with four task-node instances:
`explore:right`, `explore:down`, `explore:left`, and `explore:up`.
Each maps directly to the same-named original V1 macro action.
The term describes the ability to move and reveal targets; it adds no
novelty reward or guarantee of visiting a new location.

All four nodes exist in the graph. The robot-task feasibility mask checks
whether the adjacent destination belongs to static `walkable_positions`
metadata and whether the robot is ready. If topology is absent, Explore
edges are masked. Positions of entities outside local observation are never
used. Dynamic collisions remain handled by the original executor.

V1 uses coordinate deltas right=(0,1), down=(1,0), left=(0,-1), up=(-1,0).
For deterministic ovcA7 reset:

- agent 0: right/down/left feasible, up masked;
- agent 1: right/left/up feasible, down masked;
- agent 2: right/down/left feasible, up masked.

The masked direction points to a boundary counter. Moving changes position,
so the feasible subset may change on subsequent decisions. Explore has no
task-dependency edges; the graph has 32 dependency edges after the assembly extension.


## Retrieval and assembly semantics (approved and implemented)

CollectChoppedIngredient has one instance per recipe ingredient. It maps to
the same original get-ingredient macro. It requires visible chopped food that
is neither held by a robot nor already inside a plate. An empty-handed robot
can retrieve it; a robot holding a plate can collect it directly into the
plate when that ingredient type is absent. It is completed when the food is
held or contained. Raw acquisition and post-chop retrieval are separate tasks.

AddToPlate has one instance per ingredient/plate pair. It maps to that
plate's original get-plate macro. The robot must hold the matching chopped
ingredient; the target plate must be visible, unheld, and lack that ingredient
type. Completion means the ingredient is inside this particular plate.
The original macro deposits food and then picks up the partial plate.

GetPlate is reserved for empty-handed acquisition. A nonempty unheld plate
can still be acquired. Chop requires empty hands so that the original
interaction actually chops rather than placing a held object.

Dependencies: GetIngredient -> GoToKnife (6), GoToKnife -> Chop (6),
Chop -> CollectChoppedIngredient (6 alternative knife branches),
CollectChoppedIngredient -> AddToPlate (6), AddToPlate -> Deliver
(6 alternative plate branches), and GetPlate -> Deliver (2).
Incoming alternatives do not imply that both knives or plates must be used.
Feasibility uses observable state and the exact assembled recipe, not an AND
over every incoming dependency. Direct collection into a held plate also
completes the corresponding AddToPlate milestone without selecting it.

A fixed legal V1 fixture verifies a complete cooperative delivery using
only graph-feasible decisions and the unchanged original macro executor.


## Stage 3 neural integration

TaskGraphActor preserves the original MLP/GRU history encoder and time
embedding. A history query scores task embeddings combined with robot-task
relation features. Feasibility masking precedes softmax; epsilon exploration
is uniform over feasible task nodes. Zero padding has only Wait feasible.

The controller samples only for ready robots, maps task IDs to original
macros, and updates hidden state only on decisions. Busy robots return -1 and
retain their previous task and hidden state. The TG runner records selected
task indices in the original actor-action buffer slot and checks their
correspondence with actual macro execution.

The original critic, memory schema, asynchronous GAE, start/end filtering,
and PPO formulas are unchanged. Graphs for PPO are rebuilt from recorded
local observations; unavailable current-action/duration fields retain default
values rather than reading live hidden environment state.

NoDependency uses tg_dependency_steps=0. Full TG uses one predecessor message
step, observable edge features, active-edge mean aggregation, and residual
task updates before robot-task scoring. Both methods use the same mask.
