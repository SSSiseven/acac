"""Diagnostic-only wrapper: delivery, counters and unchanged transitions."""
import copy
import random
import unittest
import numpy as np
from acac.env import OvercookedMacEnvWrapper
from test_tg_acac import fixture
from acac.acac_marl.diagnostics import PilotMetricsWrapper, contention_counts

class PilotMetricsTests(unittest.TestCase):
    def test_resource_counts_and_empty_denominators(self):
        self.assertEqual(contention_counts(["knife1","knife1",None]),(2,2))
        self.assertEqual(contention_counts(["knife1","plate1",None]),(0,2))
        self.assertEqual(contention_counts([None,None,None]),(0,0))

    def test_no_change_to_observations_reward_done_or_rng(self):
        raw,_=fixture()
        base=OvercookedMacEnvWrapper(raw)
        observed=PilotMetricsWrapper(copy.deepcopy(base),pilot=True)
        np.random.seed(2); random.seed(2)
        initial=base.reset()
        np.random.seed(2); random.seed(2)
        monitored=observed.reset()
        for a,b in zip(initial,monitored):np.testing.assert_array_equal(a,b)
        for actions in [[0,0,0],[10,12,13],[0,0,0]]:
            before_np=np.random.get_state();before_random=random.getstate()
            result=base.step(actions)
            after_np=np.random.get_state();after_random=random.getstate()
            np.random.set_state(before_np);random.setstate(before_random)
            check=observed.step(actions)
            for a,b in zip(result[0],check[0]):np.testing.assert_array_equal(a,b)
            self.assertEqual(result[1:3],check[1:3])
            for key in result[3]:self.assertEqual(result[3][key],check[3][key])
            self.assertEqual(random.getstate(),after_random)
            np.testing.assert_array_equal(np.random.get_state()[1],after_np[1])
        metrics=observed.get_episode_metrics()
        self.assertEqual(metrics["AgentSteps"],9)
        self.assertGreater(metrics["DecisionCount"],0)
        self.assertTrue(0<=metrics["IdleRatio"]<=1)

    def test_actual_delivery_count_and_reset(self):
        raw,_=fixture()
        # Place a correct chopped dish in robot 0's hands next to delivery.
        robot=raw.agent[0];plate=raw.plate[0]
        raw.map[robot.x][robot.y]=0
        robot.move(3,1);raw.map[3][1]=2
        for food in (raw.tomato[0],raw.lettuce[0],raw.onion[0]):
            raw.map[food.x][food.y]=1
            food.chopped=True
            plate.contain(food)
        robot.pickup(plate)
        raw._get_obs()
        raw._createMacroActionItemList()
        wrapper=PilotMetricsWrapper(OvercookedMacEnvWrapper(raw),delivery=True)
        delivery=raw.macroActionName.index("deliver")
        wrong=copy.deepcopy(wrapper)
        wrong.unwrapped.agent[0].holding.containing=[
            wrong.unwrapped.agent[0].holding.containing[0]]
        _,wrong_reward,wrong_done,wrong_info=wrong.step([delivery,0,0])
        self.assertFalse(wrong_done)
        self.assertEqual(wrong_info["shelf_deliveries"],0)
        self.assertLess(wrong_reward[0],0)
        obs,reward,done,info=wrapper.step([delivery,0,0])
        self.assertTrue(done)
        self.assertEqual(info["shelf_deliveries"],1)
        self.assertGreater(reward[0],190)
        wrapper.reset()
        _,_,done,info=wrapper.step([0,0,0])
        self.assertFalse(done)
        self.assertEqual(info["shelf_deliveries"],0)

    def test_empty_plate_resource_lookup(self):
        raw,_=fixture()
        wrapper=PilotMetricsWrapper(OvercookedMacEnvWrapper(raw),pilot=True)
        wrapper.reset()
        self.assertIsNone(wrapper._resource(0,raw.macroActionName.index("get tomato")))
        wrapper.step([raw.macroActionName.index("get tomato"),0,0])

    def test_wait_busy_and_invalid_macro_accounting(self):
        raw,_=fixture()
        wrapper=PilotMetricsWrapper(OvercookedMacEnvWrapper(raw),pilot=True)
        wrapper.reset()
        wrapper.step([raw.macroActionName.index("deliver"),0,0])
        m=wrapper.get_episode_metrics()
        self.assertEqual(m["DecisionCount"],3)
        self.assertGreaterEqual(m["InvalidProposalCount"],1)
        self.assertGreaterEqual(m["IdleAgentSteps"],2)
        busy=[not a.cur_macro_action_done for a in raw.macroAgent]
        before=m["DecisionCount"]
        wrapper.step([-1 if b else 0 for b in busy])
        self.assertEqual(wrapper.get_episode_metrics()["DecisionCount"]-before,
                         sum(not b for b in busy))

if __name__=="__main__":unittest.main()
