"""Deterministic, NumPy-only construction of an egocentric Overcooked task graph.

The builder consumes an already decoded *observable* state.  Unknown entities
must use ``visible=False`` and ``position=None``; the builder never queries the
environment or fills hidden locations from global state.  This keeps graph
construction independently testable and prevents actor-side state leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Sequence, Tuple

import numpy as np


class TaskType(IntEnum):
    GET_INGREDIENT = 0
    GET_PLATE = 1
    GO_TO_KNIFE = 2
    CHOP = 3
    DELIVER = 4
    WAIT = 5
    EXPLORE = 6
    COLLECT_CHOPPED_INGREDIENT = 7
    ADD_TO_PLATE = 8


class TaskStatus(IntEnum):
    NOT_STARTED = 0
    ACTIVE = 1
    COMPLETED = 2


@dataclass(frozen=True)
class RobotObservation:
    robot_id: int
    position: Optional[Tuple[int, int]]
    visible: bool = True
    held_object_id: Optional[str] = None
    held_object_kind: Optional[str] = None
    held_ingredient_types: Tuple[str, ...] = ()
    ready: Optional[bool] = True
    current_macro_action: Optional[int] = None
    elapsed_duration: int = 0
    current_task_id: Optional[str] = None


@dataclass(frozen=True)
class EntityObservation:
    entity_id: str
    kind: str
    position: Optional[Tuple[int, int]]
    visible: bool = True
    ingredient_type: Optional[str] = None
    chopped: bool = False
    chop_progress: float = 0.0
    holder_id: Optional[str] = None
    on_resource_id: Optional[str] = None
    container_id: Optional[str] = None
    contents: Tuple[str, ...] = ()
    occupied_by: Optional[str] = None


@dataclass(frozen=True)
class TaskGraphMetadata:
    grid_shape: Tuple[int, int]
    recipe_ingredients: Tuple[str, ...]
    macro_action_names: Tuple[str, ...]
    walkable_positions: Optional[Tuple[Tuple[int, int], ...]] = None
    ingredient_vocabulary: Tuple[str, ...] = (
        "tomato",
        "lettuce",
        "onion",
        "A",
        "B",
        "C",
    )
    max_macro_duration: int = 200


@dataclass(frozen=True)
class _Task:
    task_id: str
    task_type: TaskType
    target_id: Optional[str]
    ingredient_id: Optional[str]
    ingredient_type: Optional[str]
    resource_id: Optional[str]
    macro_action: int
    status: TaskStatus
    prerequisite_satisfied: bool
    resource_available: bool
    target_visible: bool
    position: Optional[Tuple[int, int]]
    progress: float
    direction: Optional[str] = None


@dataclass(frozen=True)
class TaskGraphObservation:
    robot_features: np.ndarray
    robot_mask: np.ndarray
    task_features: np.ndarray
    task_mask: np.ndarray
    robot_task_edge_features: np.ndarray
    robot_task_mask: np.ndarray
    task_dependency_edge_index: np.ndarray
    task_dependency_edge_features: np.ndarray
    task_to_macro_action: np.ndarray
    task_ids: Tuple[str, ...]
    task_types: np.ndarray


class TaskGraphBuilder:
    """Build a graph without reading or mutating the environment.

    Robot features: id, x, y, visible, ready, holds-none,
    holds-ingredient, holds-plate, current-action, elapsed-duration, followed
    by a vocabulary one-hot for ingredients held directly or in a plate.

    Task features: nine-way task type, ingredient vocabulary one-hot,
    three-way status, target-visible, prerequisite-satisfied,
    resource-available, x, y, progress, target ordinal, resource ordinal,
    followed by a four-way direction encoding for Explore instances.

    Robot-task edge features (8): Manhattan distance, executable,
    target-visible, object-compatible, resource-available,
    prerequisite-satisfied, claimed-by-other, estimated duration.

    Dependency edge features (3): active, predecessor-completed,
    same-ingredient relation.
    """

    ROBOT_BASE_FEATURE_DIM = 10
    EDGE_FEATURE_DIM = 8
    DEPENDENCY_FEATURE_DIM = 3
    DIRECTION_NAMES = ("right", "down", "left", "up")
    DIRECTION_DELTAS = {
        "right": (0, 1),
        "down": (1, 0),
        "left": (0, -1),
        "up": (-1, 0),
    }

    def __init__(self, metadata: TaskGraphMetadata):
        width, height = metadata.grid_shape
        if width <= 0 or height <= 0:
            raise ValueError("grid_shape must contain positive dimensions")
        if not metadata.recipe_ingredients:
            raise ValueError("recipe_ingredients must not be empty")
        self.metadata = metadata
        self._actions = {name: index for index, name in enumerate(metadata.macro_action_names)}
        self._walkable_positions = (
            None
            if metadata.walkable_positions is None
            else frozenset(metadata.walkable_positions)
        )
        if self._walkable_positions is not None:
            for position in self._walkable_positions:
                if not self._position_in_bounds(position):
                    raise ValueError(f"walkable position is outside grid: {position}")

    @property
    def robot_feature_dim(self) -> int:
        return self.ROBOT_BASE_FEATURE_DIM + len(self.metadata.ingredient_vocabulary)

    @property
    def task_feature_dim(self) -> int:
        return len(TaskType) + len(self.metadata.ingredient_vocabulary) + 3 + 8 + 4

    @property
    def task_type_feature_dim(self) -> int:
        return len(TaskType)

    def build(
        self,
        robots: Sequence[RobotObservation],
        entities: Sequence[EntityObservation],
        *,
        episode_done: bool = False,
    ) -> TaskGraphObservation:
        if not robots:
            raise ValueError("at least one robot observation is required")
        self._validate_robots(robots)
        entity_by_id = self._validate_entities(entities)
        tasks, dependencies = self._build_tasks(entity_by_id, episode_done)

        robot_features = np.vstack([self._robot_features(robot, len(robots)) for robot in robots])
        robot_mask = np.asarray([robot.visible for robot in robots], dtype=np.bool_)
        task_features = np.vstack(
            [self._task_features(task, index, len(tasks)) for index, task in enumerate(tasks)]
        )
        task_mask = np.asarray([task.macro_action >= 0 for task in tasks], dtype=np.bool_)
        task_to_macro_action = np.asarray([task.macro_action for task in tasks], dtype=np.int64)
        task_ids = tuple(task.task_id for task in tasks)
        task_types = np.asarray([int(task.task_type) for task in tasks], dtype=np.int64)

        edge_features = np.zeros(
            (len(robots), len(tasks), self.EDGE_FEATURE_DIM), dtype=np.float32
        )
        edge_mask = np.zeros((len(robots), len(tasks)), dtype=np.bool_)
        for robot_index, robot in enumerate(robots):
            for task_index, task in enumerate(tasks):
                features, executable = self._robot_task_edge(
                    robot, task, robots, entity_by_id, episode_done
                )
                edge_features[robot_index, task_index] = features
                edge_mask[robot_index, task_index] = executable and task_mask[task_index]

        if dependencies:
            dependency_edge_index = np.asarray(dependencies, dtype=np.int64).T
            dependency_features = np.asarray(
                [self._dependency_features(tasks[src], tasks[dst]) for src, dst in dependencies],
                dtype=np.float32,
            )
        else:
            dependency_edge_index = np.empty((2, 0), dtype=np.int64)
            dependency_features = np.empty((0, self.DEPENDENCY_FEATURE_DIM), dtype=np.float32)

        return TaskGraphObservation(
            robot_features=robot_features,
            robot_mask=robot_mask,
            task_features=task_features,
            task_mask=task_mask,
            robot_task_edge_features=edge_features,
            robot_task_mask=edge_mask,
            task_dependency_edge_index=dependency_edge_index,
            task_dependency_edge_features=dependency_features,
            task_to_macro_action=task_to_macro_action,
            task_ids=task_ids,
            task_types=task_types,
        )

    @staticmethod
    def _validate_robots(robots: Sequence[RobotObservation]) -> None:
        ids = set()
        for robot in robots:
            if robot.robot_id in ids:
                raise ValueError(f"duplicate robot_id: {robot.robot_id}")
            if not robot.visible:
                hidden_dynamic_state = bool(
                    robot.position is not None
                    or robot.held_object_id
                    or robot.held_object_kind
                    or robot.held_ingredient_types
                    or robot.ready is not None
                    or robot.current_macro_action is not None
                    or robot.elapsed_duration
                    or robot.current_task_id
                )
                if hidden_dynamic_state:
                    raise ValueError(
                        f"hidden robot {robot.robot_id} exposes dynamic state"
                    )
            ids.add(robot.robot_id)

    def _validate_entities(
        self, entities: Sequence[EntityObservation]
    ) -> dict[str, EntityObservation]:
        result: dict[str, EntityObservation] = {}
        valid_kinds = {"ingredient", "plate", "knife", "delivery"}
        for entity in entities:
            if entity.entity_id in result:
                raise ValueError(f"duplicate entity_id: {entity.entity_id}")
            if entity.kind not in valid_kinds:
                raise ValueError(f"unsupported entity kind: {entity.kind}")
            if not entity.visible:
                hidden_dynamic_state = bool(
                    entity.position is not None
                    or entity.chopped
                    or entity.chop_progress
                    or entity.holder_id
                    or entity.on_resource_id
                    or entity.container_id
                    or entity.contents
                    or entity.occupied_by
                )
                if hidden_dynamic_state:
                    raise ValueError(
                        f"hidden entity {entity.entity_id} exposes dynamic state"
                    )
            result[entity.entity_id] = entity
        return result

    def _build_tasks(
        self, entities: dict[str, EntityObservation], episode_done: bool
    ) -> tuple[list[_Task], list[tuple[int, int]]]:
        ingredients = sorted(
            (
                entity
                for entity in entities.values()
                if entity.kind == "ingredient"
                and entity.ingredient_type in self.metadata.recipe_ingredients
            ),
            key=lambda entity: entity.entity_id,
        )
        plates = sorted(
            (entity for entity in entities.values() if entity.kind == "plate"),
            key=lambda entity: entity.entity_id,
        )
        knives = sorted(
            (entity for entity in entities.values() if entity.kind == "knife"),
            key=lambda entity: entity.entity_id,
        )
        deliveries = sorted(
            (entity for entity in entities.values() if entity.kind == "delivery"),
            key=lambda entity: entity.entity_id,
        )
        if not ingredients or not plates or not knives or not deliveries:
            raise ValueError("ingredients, plates, knives, and delivery entities are required")

        tasks: list[_Task] = []
        dependencies: list[tuple[int, int]] = []
        get_index: dict[str, int] = {}
        go_index: dict[tuple[str, str], int] = {}
        chop_index: dict[tuple[str, str], int] = {}
        plate_index: dict[str, int] = {}

        for ingredient in ingredients:
            completed = self._ingredient_acquired(ingredient)
            task = self._make_task(
                task_id=f"get:{ingredient.entity_id}",
                task_type=TaskType.GET_INGREDIENT,
                target=ingredient,
                ingredient=ingredient,
                resource=None,
                macro_action=self._action(f"get {ingredient.ingredient_type}"),
                status=self._status(completed, True, episode_done),
                prerequisite=True,
                resource_available=True,
            )
            get_index[ingredient.entity_id] = len(tasks)
            tasks.append(task)

        for plate_number, plate in enumerate(plates, start=1):
            completed = bool(plate.holder_id or plate.container_id)
            task = self._make_task(
                task_id=f"get:{plate.entity_id}",
                task_type=TaskType.GET_PLATE,
                target=plate,
                ingredient=None,
                resource=None,
                macro_action=self._action(f"get plate {plate_number}"),
                status=self._status(completed, True, episode_done),
                prerequisite=True,
                resource_available=True,
            )
            plate_index[plate.entity_id] = len(tasks)
            tasks.append(task)

        for ingredient in ingredients:
            acquired = self._ingredient_acquired(ingredient)
            for knife_number, knife in enumerate(knives, start=1):
                on_this_knife = (
                    ingredient.on_resource_id == knife.entity_id
                    or knife.occupied_by == ingredient.entity_id
                )
                available = knife.occupied_by in (None, ingredient.entity_id)
                go_task = self._make_task(
                    task_id=f"go:{ingredient.entity_id}:{knife.entity_id}",
                    task_type=TaskType.GO_TO_KNIFE,
                    target=knife,
                    ingredient=ingredient,
                    resource=knife,
                    macro_action=self._action(f"go to knife {knife_number}"),
                    status=self._status(on_this_knife, acquired, episode_done),
                    prerequisite=acquired,
                    resource_available=available,
                )
                go_key = (ingredient.entity_id, knife.entity_id)
                go_index[go_key] = len(tasks)
                tasks.append(go_task)
                dependencies.append((get_index[ingredient.entity_id], go_index[go_key]))

                chop_task = self._make_task(
                    task_id=f"chop:{ingredient.entity_id}:{knife.entity_id}",
                    task_type=TaskType.CHOP,
                    target=knife,
                    ingredient=ingredient,
                    resource=knife,
                    macro_action=self._action("chop"),
                    status=self._status(ingredient.chopped, on_this_knife, episode_done),
                    prerequisite=on_this_knife,
                    resource_available=on_this_knife,
                    progress=ingredient.chop_progress,
                )
                chop_index[go_key] = len(tasks)
                tasks.append(chop_task)
                dependencies.append((go_index[go_key], chop_index[go_key]))

        collect_index: dict[str, int] = {}
        add_index: dict[tuple[str, str], int] = {}
        for ingredient in ingredients:
            collect_index[ingredient.entity_id] = len(tasks)
            tasks.append(self._make_task(
                task_id=f"collect:{ingredient.entity_id}",
                task_type=TaskType.COLLECT_CHOPPED_INGREDIENT,
                target=ingredient, ingredient=ingredient, resource=None,
                macro_action=self._action(f"get {ingredient.ingredient_type}"),
                status=self._status(
                    bool(ingredient.holder_id or ingredient.container_id),
                    ingredient.chopped, episode_done),
                prerequisite=ingredient.visible and ingredient.chopped,
                resource_available=ingredient.holder_id is None,
            ))
            for knife in knives:
                dependencies.append((
                    chop_index[(ingredient.entity_id, knife.entity_id)],
                    collect_index[ingredient.entity_id]))
            for plate_number, plate in enumerate(plates, start=1):
                key = (ingredient.entity_id, plate.entity_id)
                add_index[key] = len(tasks)
                tasks.append(self._make_task(
                    task_id=f"add:{ingredient.entity_id}:{plate.entity_id}",
                    task_type=TaskType.ADD_TO_PLATE,
                    target=plate, ingredient=ingredient, resource=plate,
                    macro_action=self._action(f"get plate {plate_number}"),
                    status=self._status(
                        ingredient.container_id == plate.entity_id,
                        ingredient.chopped, episode_done),
                    prerequisite=ingredient.visible and ingredient.chopped,
                    resource_available=plate.holder_id is None,
                ))
                dependencies.append((collect_index[ingredient.entity_id], add_index[key]))

        assembled_plate = self._assembled_plate(plates)
        delivery = deliveries[0]
        delivered = episode_done or bool(delivery.contents)
        deliver_task = self._make_task(
            task_id="deliver:recipe",
            task_type=TaskType.DELIVER,
            target=delivery,
            ingredient=None,
            resource=delivery,
            macro_action=self._action_any("deliver", "delivery"),
            status=self._status(delivered, assembled_plate is not None, episode_done),
            prerequisite=assembled_plate is not None,
            resource_available=True,
        )
        deliver_index = len(tasks)
        tasks.append(deliver_task)
        for index in add_index.values():
            dependencies.append((index, deliver_index))
        for index in plate_index.values():
            dependencies.append((index, deliver_index))

        for direction in self.DIRECTION_NAMES:
            tasks.append(
                _Task(
                    task_id=f"explore:{direction}",
                    task_type=TaskType.EXPLORE,
                    target_id=None,
                    ingredient_id=None,
                    ingredient_type=None,
                    resource_id=None,
                    macro_action=self._action(direction),
                    status=TaskStatus.ACTIVE,
                    prerequisite_satisfied=True,
                    resource_available=True,
                    target_visible=True,
                    position=None,
                    progress=0.0,
                    direction=direction,
                )
            )

        tasks.append(
            _Task(
                task_id="wait",
                task_type=TaskType.WAIT,
                target_id=None,
                ingredient_id=None,
                ingredient_type=None,
                resource_id=None,
                macro_action=self._action_any("stay", "wait"),
                status=TaskStatus.ACTIVE,
                prerequisite_satisfied=True,
                resource_available=True,
                target_visible=True,
                position=None,
                progress=0.0,
            )
        )
        return tasks, dependencies

    def _make_task(
        self,
        *,
        task_id: str,
        task_type: TaskType,
        target: EntityObservation,
        ingredient: Optional[EntityObservation],
        resource: Optional[EntityObservation],
        macro_action: int,
        status: TaskStatus,
        prerequisite: bool,
        resource_available: bool,
        progress: float = 0.0,
    ) -> _Task:
        return _Task(
            task_id=task_id,
            task_type=task_type,
            target_id=target.entity_id,
            ingredient_id=ingredient.entity_id if ingredient else None,
            ingredient_type=ingredient.ingredient_type if ingredient else None,
            resource_id=resource.entity_id if resource else None,
            macro_action=macro_action,
            status=status,
            prerequisite_satisfied=prerequisite,
            resource_available=resource_available,
            target_visible=target.visible,
            position=target.position if target.visible else None,
            progress=float(np.clip(progress, 0.0, 1.0)),
        )

    @staticmethod
    def _ingredient_acquired(ingredient: EntityObservation) -> bool:
        return bool(
            ingredient.holder_id
            or ingredient.on_resource_id
            or ingredient.container_id
            or ingredient.chopped
        )

    def _assembled_plate(
        self, plates: Sequence[EntityObservation]
    ) -> Optional[EntityObservation]:
        required = sorted(self.metadata.recipe_ingredients)
        for plate in plates:
            if sorted(plate.contents) == required:
                return plate
        return None

    @staticmethod
    def _status(completed: bool, active: bool, episode_done: bool) -> TaskStatus:
        if completed or episode_done:
            return TaskStatus.COMPLETED
        return TaskStatus.ACTIVE if active else TaskStatus.NOT_STARTED

    def _robot_features(self, robot: RobotObservation, count: int) -> np.ndarray:
        width, height = self.metadata.grid_shape
        x, y = self._normalised_position(robot.position if robot.visible else None)
        current_action = (
            -1.0
            if robot.current_macro_action is None
            else robot.current_macro_action / max(1, len(self.metadata.macro_action_names) - 1)
        )
        held_kind = robot.held_object_kind
        base = np.asarray(
            [
                robot.robot_id / max(1, count - 1),
                x,
                y,
                float(robot.visible),
                -1.0 if robot.ready is None else float(robot.ready),
                float(held_kind is None),
                float(held_kind == "ingredient"),
                float(held_kind == "plate"),
                current_action,
                min(robot.elapsed_duration / max(1, self.metadata.max_macro_duration), 1.0),
            ],
            dtype=np.float32,
        )
        held_ingredients = np.zeros(
            len(self.metadata.ingredient_vocabulary), dtype=np.float32
        )
        for ingredient_type in robot.held_ingredient_types:
            if ingredient_type in self.metadata.ingredient_vocabulary:
                held_ingredients[
                    self.metadata.ingredient_vocabulary.index(ingredient_type)
                ] = 1.0
        return np.concatenate((base, held_ingredients))

    def _task_features(self, task: _Task, index: int, count: int) -> np.ndarray:
        type_one_hot = np.zeros(len(TaskType), dtype=np.float32)
        type_one_hot[int(task.task_type)] = 1.0
        ingredient_one_hot = np.zeros(len(self.metadata.ingredient_vocabulary), dtype=np.float32)
        if task.ingredient_type in self.metadata.ingredient_vocabulary:
            ingredient_one_hot[self.metadata.ingredient_vocabulary.index(task.ingredient_type)] = 1.0
        status_one_hot = np.zeros(3, dtype=np.float32)
        status_one_hot[int(task.status)] = 1.0
        x, y = self._normalised_position(task.position)
        target_ordinal = index / max(1, count - 1)
        resource_ordinal = (
            0.0
            if task.resource_id is None
            else (sum(ord(char) for char in task.resource_id) % 997) / 996.0
        )
        tail = np.asarray(
            [
                float(task.target_visible),
                float(task.prerequisite_satisfied),
                float(task.resource_available),
                x,
                y,
                task.progress,
                target_ordinal,
                resource_ordinal,
            ],
            dtype=np.float32,
        )
        direction_one_hot = np.zeros(len(self.DIRECTION_NAMES), dtype=np.float32)
        if task.direction in self.DIRECTION_NAMES:
            direction_one_hot[self.DIRECTION_NAMES.index(task.direction)] = 1.0
        return np.concatenate(
            (type_one_hot, ingredient_one_hot, status_one_hot, tail, direction_one_hot)
        )

    def _robot_task_edge(
        self,
        robot: RobotObservation,
        task: _Task,
        robots: Sequence[RobotObservation],
        entities: dict[str, EntityObservation],
        episode_done: bool,
    ) -> tuple[np.ndarray, bool]:
        distance = (
            1.0 / max(1, sum(self.metadata.grid_shape) - 2)
            if task.task_type == TaskType.EXPLORE
            else self._distance(robot.position, task.position)
        )
        compatible = self._object_compatible(robot, task, entities)
        claimed = task.task_type != TaskType.EXPLORE and any(
            other.robot_id != robot.robot_id
            and other.visible
            and other.ready is False
            and other.current_task_id == task.task_id
            for other in robots
        )
        executable = bool(
            robot.visible
            and robot.ready is True
            and not episode_done
            and task.status != TaskStatus.COMPLETED
            and task.prerequisite_satisfied
            and task.resource_available
            and task.target_visible
            and compatible
            and not claimed
        )
        if task.task_type == TaskType.CHOP:
            executable = (executable and robot.held_object_kind is None
                          and self._raw_distance(robot.position, task.position) == 1)
        if task.task_type == TaskType.EXPLORE:
            executable = executable and self._explore_destination_is_walkable(robot, task)
        if task.task_type == TaskType.WAIT:
            executable = bool(robot.visible and robot.ready is True and not episode_done)
        duration = distance
        return (
            np.asarray(
                [
                    distance,
                    float(executable),
                    float(task.target_visible),
                    float(compatible),
                    float(task.resource_available),
                    float(task.prerequisite_satisfied),
                    float(claimed),
                    duration,
                ],
                dtype=np.float32,
            ),
            executable,
        )

    def _object_compatible(
        self,
        robot: RobotObservation,
        task: _Task,
        entities: dict[str, EntityObservation],
    ) -> bool:
        held_kind = robot.held_object_kind
        if task.task_type in (TaskType.WAIT, TaskType.EXPLORE):
            return True
        if task.task_type == TaskType.GET_INGREDIENT:
            ingredient = entities[task.ingredient_id]
            return held_kind is None or (held_kind == "plate" and ingredient.chopped)
        if task.task_type == TaskType.GET_PLATE:
            return held_kind is None
        if task.task_type == TaskType.COLLECT_CHOPPED_INGREDIENT:
            ingredient = entities[task.ingredient_id]
            if (not ingredient.visible or not ingredient.chopped
                    or ingredient.container_id is not None
                    or ingredient.holder_id is not None):
                return False
            if held_kind is None:
                return True
            if held_kind == "plate" and robot.held_object_id in entities:
                return ingredient.ingredient_type not in entities[robot.held_object_id].contents
            return False
        if task.task_type == TaskType.ADD_TO_PLATE:
            ingredient = entities[task.ingredient_id]
            plate = entities[task.target_id]
            return (held_kind == "ingredient"
                    and robot.held_object_id == ingredient.entity_id
                    and ingredient.chopped
                    and plate.holder_id is None
                    and ingredient.ingredient_type not in plate.contents)
        if task.task_type == TaskType.GO_TO_KNIFE:
            return (
                held_kind == "ingredient"
                and robot.held_object_id == task.ingredient_id
                and not entities[task.ingredient_id].chopped
            )
        if task.task_type == TaskType.CHOP:
            return True
        if task.task_type == TaskType.DELIVER:
            if held_kind != "plate" or robot.held_object_id not in entities:
                return False
            return sorted(entities[robot.held_object_id].contents) == sorted(
                self.metadata.recipe_ingredients
            )
        return False

    def _explore_destination_is_walkable(
        self, robot: RobotObservation, task: _Task
    ) -> bool:
        if robot.position is None or task.direction not in self.DIRECTION_DELTAS:
            return False
        delta = self.DIRECTION_DELTAS[task.direction]
        destination = (robot.position[0] + delta[0], robot.position[1] + delta[1])
        if not self._position_in_bounds(destination):
            return False
        if self._walkable_positions is None:
            return False
        return destination in self._walkable_positions

    def _position_in_bounds(self, position: Tuple[int, int]) -> bool:
        width, height = self.metadata.grid_shape
        return 0 <= position[0] < width and 0 <= position[1] < height

    def _dependency_features(self, source: _Task, target: _Task) -> np.ndarray:
        return np.asarray(
            [
                float(target.status != TaskStatus.COMPLETED),
                float(source.status == TaskStatus.COMPLETED),
                float(
                    source.ingredient_id is not None
                    and source.ingredient_id == target.ingredient_id
                ),
            ],
            dtype=np.float32,
        )

    def _action(self, name: str) -> int:
        return self._actions.get(name, -1)

    def _action_any(self, *names: str) -> int:
        for name in names:
            if name in self._actions:
                return self._actions[name]
        return -1

    def _normalised_position(
        self, position: Optional[Tuple[int, int]]
    ) -> tuple[float, float]:
        if position is None:
            return 0.0, 0.0
        width, height = self.metadata.grid_shape
        return position[0] / max(1, width - 1), position[1] / max(1, height - 1)

    def _distance(
        self,
        source: Optional[Tuple[int, int]],
        target: Optional[Tuple[int, int]],
    ) -> float:
        raw = self._raw_distance(source, target)
        if raw is None:
            return 1.0
        width, height = self.metadata.grid_shape
        return raw / max(1, width + height - 2)

    @staticmethod
    def _raw_distance(
        source: Optional[Tuple[int, int]],
        target: Optional[Tuple[int, int]],
    ) -> Optional[int]:
        if source is None or target is None:
            return None
        return abs(source[0] - target[0]) + abs(source[1] - target[1])
