"""Regression coverage for post-chop retrieval and full V1 assembly."""
import copy
import unittest
from dataclasses import replace

import test_task_graph_builder as fixtures
update_entities = fixtures.update_entities
from test_task_graph_v1_adapter import (
    Overcooked_MA_V1, TaskGraphBuilder, TaskGraphMetadata,
    decode_v1_macro_observation, ACTIONS, WALKABLE_A7, LAYOUT,
)


class AssemblySemanticsTests(unittest.TestCase):
    setUp = fixtures.TaskGraphBuilderScenarios.setUp
    index = staticmethod(fixtures.TaskGraphBuilderScenarios.index)
    def test_collect_requires_chopped_unheld_uncontained_visible_food(self):
        entities = update_entities(
            self.entities,
            tomato={"position": (3, 2), "chopped": True,
                    "chop_progress": 1.0, "on_resource_id": "knife1"},
            knife1={"occupied_by": "tomato"},
        )
        graph = self.builder.build((self.robot,), entities)
        index = self.index(graph, "collect:tomato")
        self.assertTrue(graph.robot_task_mask[0, index])
        self.assertEqual(graph.task_to_macro_action[index], 1)
        self.assertFalse(graph.robot_task_mask[0, self.index(graph, "get:tomato")])
        for changes in (
            {"chopped": False, "chop_progress": 0.0},
            {"holder_id": "robot:1"},
            {"container_id": "plate1"},
            {"visible": False, "position": None, "chopped": False,
             "chop_progress": 0.0, "on_resource_id": None},
        ):
            modified = update_entities(entities, tomato=changes)
            graph = self.builder.build((self.robot,), modified)
            self.assertFalse(graph.robot_task_mask[0, index])

    def test_partial_plate_allows_add_and_later_pickup(self):
        entities = update_entities(
            self.entities,
            tomato={"position": self.robot.position, "chopped": True,
                    "chop_progress": 1.0, "holder_id": "robot:0"},
            plate1={"contents": ("lettuce",)},
        )
        robot = replace(self.robot, held_object_kind="ingredient",
                        held_object_id="tomato")
        graph = self.builder.build((robot,), entities)
        add = self.index(graph, "add:tomato:plate1")
        self.assertTrue(graph.robot_task_mask[0, add])
        self.assertEqual(graph.task_to_macro_action[add], 4)
        self.assertFalse(graph.robot_task_mask[0, self.index(graph, "get:plate1")])
        empty = self.builder.build((self.robot,), entities)
        self.assertTrue(empty.robot_task_mask[0, self.index(empty, "get:plate1")])

    def test_add_masks_wrong_food_duplicate_busy_and_occupied_plate(self):
        entities = update_entities(
            self.entities,
            tomato={"chopped": True, "holder_id": "robot:0"},
        )
        robot = replace(self.robot, held_object_kind="ingredient",
                        held_object_id="tomato")
        add_id = "add:tomato:plate1"
        for modified_robot, modified_entities in (
            (replace(robot, held_object_id="lettuce"), entities),
            (replace(robot, ready=False), entities),
            (robot, update_entities(entities, plate1={"contents": ("tomato",)})),
            (robot, update_entities(entities, plate1={"holder_id": "robot:1"})),
            (robot, update_entities(entities, tomato={"chopped": False})),
            (robot, update_entities(entities, plate1={"visible": False, "position": None})),
        ):
            graph = self.builder.build((modified_robot,), modified_entities)
            self.assertFalse(graph.robot_task_mask[0, self.index(graph, add_id)])

    def test_held_plate_can_collect_chopped_food_directly(self):
        robot = replace(self.robot, held_object_kind="plate", held_object_id="plate1")
        entities = update_entities(self.entities, tomato={"chopped": True},
                                   plate1={"holder_id": "robot:0"})
        graph = self.builder.build((robot,), entities)
        self.assertTrue(graph.robot_task_mask[0, self.index(graph, "collect:tomato")])
        entities = update_entities(entities, plate1={"contents": ("tomato",)})
        graph = self.builder.build((robot,), entities)
        self.assertFalse(graph.robot_task_mask[0, self.index(graph, "collect:tomato")])


class OriginalV1AssemblyChainTests(unittest.TestCase):
    def test_full_chain_executes_original_macros_with_feasible_graph_decisions(self):
        reward = {"subtask finished": 10, "correct delivery": 200,
                  "wrong delivery": -5, "step penalty": -0.1}
        env = Overcooked_MA_V1((7, 7), "lettuce-onion-tomato salad", reward,
                              map_type="A", n_agent=3, rand_start=False)
        env.reset()
        # Fixed legal state: all movable items on counters, robots on floor.
        env.map = copy.deepcopy(env.pomap)
        for robot, pos in zip(env.agent, ((1, 1), (4, 4), (4, 5))):
            robot.move(*pos)
            env.map[pos[0]][pos[1]] = 2
        for food, pos, code in zip(
            (env.tomato[0], env.lettuce[0], env.onion[0]),
            ((0, 1), (0, 2), (0, 3)), (3, 4, 8),
        ):
            food.move(*pos)
            env.map[pos[0]][pos[1]] = code
        for plate, pos in zip(env.plate, ((0, 4), (0, 5))):
            plate.move(*pos)
            env.map[pos[0]][pos[1]] = 5
        for knife in env.knife:
            env.map[knife.x][knife.y] = 6
        for delivery in env.delivery:
            env.map[delivery.x][delivery.y] = 7
        builder = TaskGraphBuilder(TaskGraphMetadata(
            grid_shape=(7, 7), recipe_ingredients=("tomato", "lettuce", "onion"),
            macro_action_names=ACTIONS, walkable_positions=WALKABLE_A7))
        done = False

        def decide(task_id, robot_id=0):
            nonlocal done
            env._get_obs()
            robots, entities = decode_v1_macro_observation(
                env._get_macro_obs()[robot_id], LAYOUT,
                self_robot_id=robot_id, self_ready=True)
            graph = builder.build(robots, entities)
            index = graph.task_ids.index(task_id)
            self.assertTrue(graph.robot_task_mask[robot_id, index], task_id)
            macro = int(graph.task_to_macro_action[index])
            for _ in range(40):
                _, _, done, info = env.run([macro if i == robot_id else 0 for i in range(3)])
                if info["mac_done"][robot_id]:
                    return
            self.fail(f"macro did not terminate: {task_id}")

        def move_to(robot_id, target):
            for _ in range(20):
                position = (env.agent[robot_id].x, env.agent[robot_id].y)
                if position == target:
                    return
                if position[0] < target[0]: direction = "down"
                elif position[0] > target[0]: direction = "up"
                elif position[1] < target[1]: direction = "right"
                else: direction = "left"
                decide(f"explore:{direction}", robot_id)
            self.fail("fixture movement did not reach destination")

        decide("get:tomato")
        decide("go:tomato:knife1")
        decide("chop:tomato:knife1")
        decide("collect:tomato")
        move_to(0, (1, 2))
        decide("add:tomato:plate1")
        # Original get-plate macro deposits food then picks up the partial plate.
        self.assertIs(env.agent[0].holding, env.plate[0])
        move_to(0, (3, 2))
        move_to(1, (2, 2))
        for ingredient in ("lettuce", "onion"):
            decide(f"get:{ingredient}", 1)
            move_to(1, (1, 2))
            decide(f"go:{ingredient}:knife2", 1)
            decide(f"chop:{ingredient}:knife2", 1)
            move_to(1, (2, 2))
            decide(f"collect:{ingredient}", 0)
            self.assertIn(ingredient, [food.rawName for food in env.plate[0].containing])
            if ingredient == "lettuce":
                move_to(0, (3, 2))
        decide("deliver:recipe")
        self.assertTrue(done)
        self.assertIs(env.delivery[0].holding, env.plate[0])


if __name__ == "__main__":
    unittest.main()
