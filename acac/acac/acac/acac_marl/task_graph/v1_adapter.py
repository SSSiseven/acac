"""Decode the V1 macro-observation vector without consulting global state."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence, Tuple

import numpy as np

from .builder import EntityObservation, RobotObservation


@dataclass(frozen=True)
class MacroEntitySpec:
    entity_id: str
    kind: str
    ingredient_type: Optional[str] = None
    robot_id: Optional[int] = None


@dataclass(frozen=True)
class V1MacroObservationLayout:
    grid_shape: Tuple[int, int]
    entity_specs: Tuple[MacroEntitySpec, ...]
    task_vector_size: int = 7

    @property
    def observation_size(self) -> int:
        entity_size = sum(
            3 if spec.kind == "ingredient" else 2 for spec in self.entity_specs
        )
        return entity_size + self.task_vector_size


def decode_v1_macro_observation(
    observation: Sequence[float],
    layout: V1MacroObservationLayout,
    *,
    self_robot_id: int,
    self_ready: bool,
    self_current_macro_action: Optional[int] = None,
    self_elapsed_duration: int = 0,
) -> tuple[tuple[RobotObservation, ...], tuple[EntityObservation, ...]]:
    """Decode only information carried by one agent's macro observation.

    V1 writes ``(0, 0[, 0])`` for an entity outside the local observation
    radius.  Static IDs and kinds come from ``layout``; no true location or
    dynamic state is accepted for an invisible entity.  Co-location is used to
    infer holding, plate containment, and knife occupancy, matching the V1
    environment's item movement rules.
    """

    values = np.asarray(observation, dtype=np.float32).reshape(-1)
    if values.size != layout.observation_size:
        raise ValueError(
            f"expected observation size {layout.observation_size}, got {values.size}"
        )
    width, height = layout.grid_shape
    cursor = 0
    entities: dict[str, EntityObservation] = {}
    robot_positions: dict[int, Optional[Tuple[int, int]]] = {}
    robot_visibility: dict[int, bool] = {}

    for spec in layout.entity_specs:
        x_value = float(values[cursor])
        y_value = float(values[cursor + 1])
        cursor += 2
        progress = 0.0
        if spec.kind == "ingredient":
            progress = float(np.clip(values[cursor], 0.0, 1.0))
            cursor += 1
        visible = not (x_value == 0.0 and y_value == 0.0)
        position = None
        if visible:
            position = (
                int(round(x_value * width)),
                int(round(y_value * height)),
            )
        if spec.kind == "robot":
            if spec.robot_id is None:
                raise ValueError(f"robot spec {spec.entity_id} has no robot_id")
            robot_positions[spec.robot_id] = position
            robot_visibility[spec.robot_id] = visible
        else:
            entities[spec.entity_id] = EntityObservation(
                entity_id=spec.entity_id,
                kind=spec.kind,
                position=position,
                visible=visible,
                ingredient_type=spec.ingredient_type,
                chopped=bool(progress >= 1.0) if visible else False,
                chop_progress=progress if visible else 0.0,
            )

    if not robot_visibility.get(self_robot_id, False):
        raise ValueError("the perspective robot must be visible in its own observation")

    def visible_at(kind: str, position: Tuple[int, int]):
        return sorted(
            (
                entity
                for entity in entities.values()
                if entity.kind == kind
                and entity.visible
                and entity.position == position
            ),
            key=lambda entity: entity.entity_id,
        )

    # A food and plate can share a cell only after the plate contains the food.
    for ingredient in list(entities.values()):
        if ingredient.kind != "ingredient" or not ingredient.visible:
            continue
        plates = visible_at("plate", ingredient.position)
        if plates:
            entities[ingredient.entity_id] = replace(
                ingredient, container_id=plates[0].entity_id
            )

    for plate in list(entities.values()):
        if plate.kind != "plate" or not plate.visible:
            continue
        contents = tuple(
            sorted(
                ingredient.ingredient_type
                for ingredient in entities.values()
                if ingredient.kind == "ingredient"
                and ingredient.container_id == plate.entity_id
                and ingredient.ingredient_type is not None
            )
        )
        entities[plate.entity_id] = replace(plate, contents=contents)

    # A knife holds at most one plate or loose ingredient.  Prefer a plate,
    # since ingredients co-located with that plate are its contents.
    for knife in list(entities.values()):
        if knife.kind != "knife" or not knife.visible:
            continue
        plates = visible_at("plate", knife.position)
        loose_ingredients = [
            ingredient
            for ingredient in visible_at("ingredient", knife.position)
            if ingredient.container_id is None
        ]
        occupant = plates[0] if plates else (loose_ingredients[0] if loose_ingredients else None)
        if occupant is not None:
            entities[knife.entity_id] = replace(knife, occupied_by=occupant.entity_id)
            entities[occupant.entity_id] = replace(
                entities[occupant.entity_id], on_resource_id=knife.entity_id
            )

    held_by_robot: dict[int, Optional[str]] = {}
    for robot_id, position in robot_positions.items():
        held_id = None
        if position is not None:
            plates = [
                plate
                for plate in visible_at("plate", position)
                if plate.on_resource_id is None
            ]
            loose_ingredients = [
                ingredient
                for ingredient in visible_at("ingredient", position)
                if ingredient.container_id is None
                and ingredient.on_resource_id is None
            ]
            held = plates[0] if plates else (loose_ingredients[0] if loose_ingredients else None)
            if held is not None:
                held_id = held.entity_id
                entities[held_id] = replace(
                    entities[held_id], holder_id=f"robot:{robot_id}"
                )
        held_by_robot[robot_id] = held_id

    deliveries = [entity for entity in entities.values() if entity.kind == "delivery"]
    for delivery in deliveries:
        if not delivery.visible:
            continue
        delivered_plates = visible_at("plate", delivery.position)
        if delivered_plates:
            entities[delivery.entity_id] = replace(
                delivery, contents=tuple(plate.entity_id for plate in delivered_plates)
            )

    robots = []
    for robot_id in sorted(robot_positions):
        visible = robot_visibility[robot_id]
        if not visible:
            robots.append(
                RobotObservation(
                    robot_id=robot_id,
                    position=None,
                    visible=False,
                    ready=None,
                )
            )
            continue
        held_id = held_by_robot[robot_id]
        held_kind = entities[held_id].kind if held_id else None
        held_types: Tuple[str, ...] = ()
        if held_id and held_kind == "ingredient":
            ingredient_type = entities[held_id].ingredient_type
            held_types = (ingredient_type,) if ingredient_type else ()
        elif held_id and held_kind == "plate":
            held_types = entities[held_id].contents
        is_self = robot_id == self_robot_id
        robots.append(
            RobotObservation(
                robot_id=robot_id,
                position=robot_positions[robot_id],
                visible=True,
                held_object_id=held_id,
                held_object_kind=held_kind,
                held_ingredient_types=held_types,
                ready=self_ready if is_self else None,
                current_macro_action=self_current_macro_action if is_self else None,
                elapsed_duration=self_elapsed_duration if is_self else 0,
            )
        )

    return tuple(robots), tuple(entities[key] for key in sorted(entities))
