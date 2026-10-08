import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))

from acac.acac_marl.task_graph import (  # noqa: E402
    EntityObservation,
    RobotObservation,
    TaskGraphBuilder,
    TaskGraphMetadata,
    TaskStatus,
    TaskType,
)


ACTIONS = (
    "stay",
    "get tomato",
    "get lettuce",
    "get onion",
    "get plate 1",
    "get plate 2",
    "go to knife 1",
    "go to knife 2",
    "deliver",
    "chop",
    "right",
    "down",
    "left",
    "up",
)


WALKABLE_A7 = tuple((x, y) for x in range(1, 6) for y in range(1, 6))


def base_entities():
    return (
        EntityObservation("tomato", "ingredient", (1, 1), ingredient_type="tomato"),
        EntityObservation("lettuce", "ingredient", (1, 2), ingredient_type="lettuce"),
        EntityObservation("onion", "ingredient", (1, 3), ingredient_type="onion"),
        EntityObservation("plate1", "plate", (5, 1)),
        EntityObservation("plate2", "plate", (5, 2)),
        EntityObservation("knife1", "knife", (3, 2)),
        EntityObservation("knife2", "knife", (4, 2)),
        EntityObservation("delivery", "delivery", (6, 3)),
    )


def update_entities(entities, **updates):
    return tuple(replace(entity, **updates.get(entity.entity_id, {})) for entity in entities)


class TaskGraphBuilderScenarios(unittest.TestCase):
    def setUp(self):
        self.builder = TaskGraphBuilder(
            TaskGraphMetadata(
                grid_shape=(7, 7),
                recipe_ingredients=("tomato", "lettuce", "onion"),
                macro_action_names=ACTIONS,
                walkable_positions=WALKABLE_A7,
            )
        )
        self.robot = RobotObservation(0, (3, 3))
        self.entities = base_entities()

    @staticmethod
    def index(graph, task_id):
        return graph.task_ids.index(task_id)

    def status(self, graph, index):
        start = self.builder.task_type_feature_dim + len(
            self.builder.metadata.ingredient_vocabulary
        )
        return TaskStatus(int(np.argmax(graph.task_features[index, start : start + 3])))

    def test_episode_initial_state(self):
        graph = self.builder.build((self.robot,), self.entities)
        self.assertEqual(graph.robot_features.shape, (1, self.builder.robot_feature_dim))
        self.assertEqual(graph.task_features.shape, (32, self.builder.task_feature_dim))
        self.assertEqual(graph.robot_task_edge_features.shape, (1, 32, 8))
        self.assertEqual(graph.task_dependency_edge_index.shape, (2, 32))
        self.assertEqual(graph.task_dependency_edge_features.shape, (32, 3))
        self.assertEqual(graph.task_to_macro_action[self.index(graph, "get:tomato")], 1)
        self.assertEqual(graph.task_to_macro_action[self.index(graph, "go:tomato:knife1")], 6)
        self.assertEqual(graph.task_to_macro_action[self.index(graph, "chop:tomato:knife1")], 9)
        self.assertEqual(graph.task_to_macro_action[self.index(graph, "deliver:recipe")], 8)
        self.assertEqual(graph.task_to_macro_action[self.index(graph, "explore:right")], 10)
        self.assertEqual(graph.task_to_macro_action[self.index(graph, "explore:down")], 11)
        self.assertEqual(graph.task_to_macro_action[self.index(graph, "explore:left")], 12)
        self.assertEqual(graph.task_to_macro_action[self.index(graph, "explore:up")], 13)
        self.assertEqual(graph.task_to_macro_action[self.index(graph, "wait")], 0)
        self.assertTrue(graph.robot_task_mask[0, self.index(graph, "get:tomato")])
        self.assertFalse(graph.robot_task_mask[0, self.index(graph, "go:tomato:knife1")])
        self.assertFalse(graph.robot_task_mask[0, self.index(graph, "chop:tomato:knife1")])
        self.assertFalse(graph.robot_task_mask[0, self.index(graph, "deliver:recipe")])
        self.assertTrue(graph.robot_task_mask[0, self.index(graph, "wait")])

    def test_agent_holding_ingredient_enables_matching_go_to_knife(self):
        entities = update_entities(self.entities, tomato={"holder_id": "robot:0"})
        robot = replace(
            self.robot,
            held_object_id="tomato",
            held_object_kind="ingredient",
            held_ingredient_types=("tomato",),
        )
        graph = self.builder.build((robot,), entities)
        self.assertEqual(self.status(graph, self.index(graph, "get:tomato")), TaskStatus.COMPLETED)
        self.assertTrue(graph.robot_task_mask[0, self.index(graph, "go:tomato:knife1")])
        self.assertTrue(graph.robot_task_mask[0, self.index(graph, "go:tomato:knife2")])
        self.assertFalse(graph.robot_task_mask[0, self.index(graph, "go:lettuce:knife1")])

    def test_chopped_ingredient_marks_chop_task_completed(self):
        entities = update_entities(
            self.entities,
            tomato={
                "position": (3, 2),
                "chopped": True,
                "chop_progress": 1.0,
                "on_resource_id": "knife1",
            },
            knife1={"occupied_by": "tomato"},
        )
        graph = self.builder.build((self.robot,), entities)
        chop = self.index(graph, "chop:tomato:knife1")
        self.assertEqual(self.status(graph, chop), TaskStatus.COMPLETED)
        self.assertFalse(graph.robot_task_mask[0, chop])

    def test_agent_holding_complete_plate_can_deliver(self):
        contents = ("tomato", "lettuce", "onion")
        entities = update_entities(
            self.entities,
            tomato={"chopped": True, "container_id": "plate1"},
            lettuce={"chopped": True, "container_id": "plate1"},
            onion={"chopped": True, "container_id": "plate1"},
            plate1={"position": (3, 3), "holder_id": "robot:0", "contents": contents},
        )
        robot = replace(
            self.robot,
            held_object_id="plate1",
            held_object_kind="plate",
            held_ingredient_types=contents,
        )
        graph = self.builder.build((robot,), entities)
        deliver = self.index(graph, "deliver:recipe")
        self.assertEqual(self.status(graph, deliver), TaskStatus.ACTIVE)
        self.assertTrue(graph.robot_task_mask[0, deliver])

    def test_agent_holding_empty_plate_cannot_deliver(self):
        entities = update_entities(
            self.entities,
            plate1={"position": (3, 3), "holder_id": "robot:0"},
        )
        robot = replace(
            self.robot,
            held_object_id="plate1",
            held_object_kind="plate",
        )
        graph = self.builder.build((robot,), entities)
        self.assertFalse(graph.robot_task_mask[0, self.index(graph, "deliver:recipe")])
        self.assertFalse(graph.robot_task_mask[0, self.index(graph, "get:tomato")])

    def test_occupied_knife_masks_conflicting_go_to_knife(self):
        entities = update_entities(
            self.entities,
            tomato={"holder_id": "robot:0"},
            lettuce={"position": (3, 2), "on_resource_id": "knife1"},
            knife1={"occupied_by": "lettuce"},
        )
        robot = replace(self.robot, held_object_id="tomato", held_object_kind="ingredient")
        graph = self.builder.build((robot,), entities)
        self.assertFalse(graph.robot_task_mask[0, self.index(graph, "go:tomato:knife1")])
        self.assertTrue(graph.robot_task_mask[0, self.index(graph, "go:tomato:knife2")])

    def test_unfinished_predecessor_masks_chop(self):
        graph = self.builder.build((self.robot,), self.entities)
        chop = self.index(graph, "chop:tomato:knife1")
        self.assertEqual(self.status(graph, chop), TaskStatus.NOT_STARTED)
        self.assertFalse(graph.robot_task_mask[0, chop])
        incoming = graph.task_dependency_edge_index[1] == chop
        self.assertEqual(int(incoming.sum()), 1)
        self.assertEqual(graph.task_dependency_edge_features[incoming, 1].item(), 0.0)

    def test_completed_task_is_not_executable(self):
        entities = update_entities(self.entities, tomato={"holder_id": "robot:1"})
        graph = self.builder.build((self.robot,), entities)
        get_tomato = self.index(graph, "get:tomato")
        self.assertEqual(self.status(graph, get_tomato), TaskStatus.COMPLETED)
        self.assertFalse(graph.robot_task_mask[0, get_tomato])

    def test_busy_agent_has_no_new_task_decision(self):
        robot = replace(
            self.robot,
            ready=False,
            current_macro_action=1,
            current_task_id="get:tomato",
            elapsed_duration=4,
        )
        graph = self.builder.build((robot,), self.entities)
        self.assertFalse(graph.robot_task_mask.any())

    def test_wait_is_only_fallback_when_all_other_tasks_are_unavailable(self):
        builder = TaskGraphBuilder(
            replace(self.builder.metadata, walkable_positions=((3, 3),))
        )
        robot = replace(
            self.robot,
            held_object_id="unusable",
            held_object_kind="ingredient",
        )
        graph = builder.build((robot,), self.entities)
        wait = self.index(graph, "wait")
        non_wait = np.delete(graph.robot_task_mask[0], wait)
        self.assertFalse(non_wait.any())
        self.assertTrue(graph.robot_task_mask[0, wait])

    def test_explore_nodes_are_directional_instances_of_one_task_type(self):
        graph = self.builder.build((self.robot,), self.entities)
        direction_start = (
            self.builder.task_type_feature_dim
            + len(self.builder.metadata.ingredient_vocabulary)
            + 3
            + 8
        )
        for direction_index, direction in enumerate(self.builder.DIRECTION_NAMES):
            task_index = self.index(graph, f"explore:{direction}")
            self.assertEqual(graph.task_types[task_index], int(TaskType.EXPLORE))
            self.assertTrue(graph.robot_task_mask[0, task_index])
            expected = np.zeros(4, dtype=np.float32)
            expected[direction_index] = 1.0
            np.testing.assert_array_equal(
                graph.task_features[task_index, direction_start : direction_start + 4],
                expected,
            )

    def test_explore_mask_uses_static_walkable_cells(self):
        robot = replace(self.robot, position=(1, 1))
        graph = self.builder.build((robot,), self.entities)
        self.assertTrue(graph.robot_task_mask[0, self.index(graph, "explore:right")])
        self.assertTrue(graph.robot_task_mask[0, self.index(graph, "explore:down")])
        self.assertFalse(graph.robot_task_mask[0, self.index(graph, "explore:left")])
        self.assertFalse(graph.robot_task_mask[0, self.index(graph, "explore:up")])

    def test_explore_is_masked_when_static_topology_is_not_supplied(self):
        builder = TaskGraphBuilder(
            replace(self.builder.metadata, walkable_positions=None)
        )
        graph = builder.build((self.robot,), self.entities)
        explore = graph.task_types == int(TaskType.EXPLORE)
        self.assertFalse(graph.robot_task_mask[0, explore].any())

    def test_locally_invisible_target_has_no_location_or_executable_edge(self):
        entities = update_entities(
            self.entities,
            tomato={"visible": False, "position": None},
        )
        graph = self.builder.build((self.robot,), entities)
        get_tomato = self.index(graph, "get:tomato")
        self.assertTrue(graph.task_mask[get_tomato])
        self.assertFalse(graph.robot_task_mask[0, get_tomato])
        tail_start = (
            self.builder.task_type_feature_dim
            + len(self.builder.metadata.ingredient_vocabulary)
            + 3
        )
        self.assertEqual(graph.task_features[get_tomato, tail_start], 0.0)
        np.testing.assert_array_equal(
            graph.task_features[get_tomato, tail_start + 3 : tail_start + 5],
            np.zeros(2, dtype=np.float32),
        )

    def test_hidden_entity_dynamic_state_is_rejected(self):
        entities = update_entities(
            self.entities,
            tomato={"visible": False, "position": None, "chopped": True},
        )
        with self.assertRaisesRegex(ValueError, "exposes dynamic state"):
            self.builder.build((self.robot,), entities)


if __name__ == "__main__":
    unittest.main()

