import sys
import unittest
from pathlib import Path

import numpy as np

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "acac" / "acac"
sys.path.insert(0, str(SOURCE_ROOT))

from acac.acac_marl.task_graph import (  # noqa: E402
    MacroEntitySpec,
    TaskGraphBuilder,
    TaskGraphMetadata,
    V1MacroObservationLayout,
    decode_v1_macro_observation,
)
from acac.env.overcooked_env.overcooked_MA_V1 import Overcooked_MA_V1  # noqa: E402


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
LAYOUT = V1MacroObservationLayout(
    grid_shape=(7, 7),
    entity_specs=(
        MacroEntitySpec("tomato", "ingredient", "tomato"),
        MacroEntitySpec("lettuce", "ingredient", "lettuce"),
        MacroEntitySpec("onion", "ingredient", "onion"),
        MacroEntitySpec("plate1", "plate"),
        MacroEntitySpec("plate2", "plate"),
        MacroEntitySpec("knife1", "knife"),
        MacroEntitySpec("knife2", "knife"),
        MacroEntitySpec("delivery", "delivery"),
        MacroEntitySpec("robot0", "robot", robot_id=0),
        MacroEntitySpec("robot1", "robot", robot_id=1),
        MacroEntitySpec("robot2", "robot", robot_id=2),
    ),
)


class V1MacroObservationAdapterTests(unittest.TestCase):
    def test_real_ovca7_reset_decodes_without_global_dynamic_state(self):
        reward = {
            "subtask finished": 10,
            "correct delivery": 200,
            "wrong delivery": -5,
            "step penalty": -0.1,
        }
        env = Overcooked_MA_V1(
            (7, 7),
            "lettuce-onion-tomato salad",
            reward,
            map_type="A",
            n_agent=3,
            rand_start=False,
        )
        observations = env.reset()
        self.assertEqual(tuple(env.macroActionName), ACTIONS)
        self.assertEqual([len(obs) for obs in observations], [32, 32, 32])
        builder = TaskGraphBuilder(
            TaskGraphMetadata(
                grid_shape=(7, 7),
                recipe_ingredients=("tomato", "lettuce", "onion"),
                macro_action_names=ACTIONS,
            )
        )
        for robot_id, observation in enumerate(observations):
            robots, entities = decode_v1_macro_observation(
                observation,
                LAYOUT,
                self_robot_id=robot_id,
                self_ready=True,
            )
            self_robot = next(robot for robot in robots if robot.robot_id == robot_id)
            self.assertTrue(self_robot.visible)
            self.assertTrue(self_robot.ready)
            for robot in robots:
                if robot.robot_id != robot_id and robot.visible:
                    self.assertIsNone(robot.ready)
            graph = builder.build(robots, entities)
            self.assertEqual(len(graph.task_ids), 19)
            self.assertEqual(graph.task_to_macro_action[graph.task_ids.index("wait")], 0)

    def test_real_ovcar7_random_reset_uses_same_observation_contract(self):
        import random

        random.seed(0)
        np.random.seed(0)
        reward = {
            "subtask finished": 10,
            "correct delivery": 200,
            "wrong delivery": -5,
            "step penalty": -0.1,
        }
        env = Overcooked_MA_V1(
            (7, 7),
            "lettuce-onion-tomato salad",
            reward,
            map_type="A",
            n_agent=3,
            rand_start=True,
        )
        observations = env.reset()
        builder = TaskGraphBuilder(
            TaskGraphMetadata(
                grid_shape=(7, 7),
                recipe_ingredients=("tomato", "lettuce", "onion"),
                macro_action_names=tuple(env.macroActionName),
            )
        )
        for robot_id, observation in enumerate(observations):
            robots, entities = decode_v1_macro_observation(
                observation,
                LAYOUT,
                self_robot_id=robot_id,
                self_ready=True,
            )
            graph = builder.build(robots, entities)
            self.assertEqual(graph.robot_features.shape[1], builder.robot_feature_dim)
            self.assertEqual(graph.task_features.shape[0], 19)

    def test_colocation_decodes_complete_plate_held_by_self(self):
        observation = np.zeros(LAYOUT.observation_size, dtype=np.float32)
        location = (3 / 7, 3 / 7)
        for offset in (0, 3, 6):
            observation[offset] = location[0]
            observation[offset + 1] = location[1]
            observation[offset + 2] = 1.0
        observation[9:11] = location
        observation[17:19] = (4 / 7, 3 / 7)
        observation[19:21] = location
        observation[-1] = 1.0

        robots, entities = decode_v1_macro_observation(
            observation,
            LAYOUT,
            self_robot_id=0,
            self_ready=True,
        )
        robot = next(item for item in robots if item.robot_id == 0)
        entity = {item.entity_id: item for item in entities}
        self.assertEqual(robot.held_object_id, "plate1")
        self.assertEqual(robot.held_object_kind, "plate")
        self.assertEqual(
            sorted(robot.held_ingredient_types),
            ["lettuce", "onion", "tomato"],
        )
        self.assertEqual(entity["tomato"].container_id, "plate1")

        builder = TaskGraphBuilder(
            TaskGraphMetadata(
                grid_shape=(7, 7),
                recipe_ingredients=("tomato", "lettuce", "onion"),
                macro_action_names=ACTIONS,
            )
        )
        graph = builder.build(robots, entities)
        self.assertTrue(
            graph.robot_task_mask[0, graph.task_ids.index("deliver:recipe")]
        )


if __name__ == "__main__":
    unittest.main()
