"""TG pointer, masking, gradients, task experience, and ready-only history tests."""
import unittest
import numpy as np
import torch
from acac.acac_marl.cores.acac.models import GRUEncoder
from acac.acac_marl.cores.tg_acac.models import (
    TaskGraphActor, make_v1_contract, masked_task_log_probs,
)
from acac.acac_marl.cores.tg_acac.controller import TaskGraphMAC
from acac.acac_marl.cores.tg_acac.envs_runner import TaskGraphRunner
from test_task_graph_v1_adapter import Overcooked_MA_V1


def fixture():
    env = Overcooked_MA_V1((7, 7), "lettuce-onion-tomato salad",
        {"subtask finished":10, "correct delivery":200,
         "wrong delivery":-5, "step penalty":-0.1},
        map_type="A", n_agent=3, rand_start=False)
    observations = env.reset()
    return env, observations


class TGPointerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.env, self.observations = fixture()
        layout, metadata = make_v1_contract(self.env)
        self.encoder = GRUEncoder(34, [32,32], 32)
        self.actor = TaskGraphActor(
            32, 14, [32,32], 32, encoder=self.encoder,
            use_time_emb=True, time_emb_dim=2, max_timestep=200,
            use_ln=True, robot_id=0, layout=layout, metadata=metadata)

    def test_shapes_masks_padding_and_preserved_encoder(self):
        self.assertIs(self.actor.encoder, self.encoder)
        obs = torch.from_numpy(self.observations[0]).float()
        x = torch.stack((obs, torch.zeros_like(obs))).view(1,2,32)
        logits, hidden = self.actor(x, time_emb=torch.tensor([[1,0]]))
        self.assertEqual(tuple(logits.shape), (1,2,32))
        self.assertEqual(tuple(hidden.shape), (1,1,32))
        self.assertTrue(torch.isfinite(logits).all())
        mask = self.actor.graph_tensors(x)[3]
        self.assertTrue((logits.exp()[~mask] == 0).all())
        torch.testing.assert_close(logits.exp().sum(-1), torch.ones(1,2))
        self.assertEqual(int(mask[0,1].sum()), 1)

    def test_sequence_replay_matches_ready_decision_history(self):
        raw = torch.tensor(np.stack([self.observations[0]]*3)).float().view(1,3,32)
        timing = torch.tensor([[1,2,3]])
        all_logits, all_hidden = self.actor(raw, time_emb=timing)
        hidden = None
        sequential = []
        for step in range(3):
            logits, hidden = self.actor(
                raw[:,step:step+1], hidden, time_emb=timing[:,step:step+1])
            sequential.append(logits)
        torch.testing.assert_close(torch.cat(sequential,1),all_logits)
        torch.testing.assert_close(hidden,all_hidden)

    def test_restore_early_nodependency_checkpoint_static_index(self):
        state=self.actor.state_dict()
        state.pop("dependency_edge_index")
        self.actor.load_state_dict(state)
        self.assertEqual(tuple(self.actor.dependency_edge_index.shape),(2,32))

    def test_diagnostic_mass_does_not_change_policy_or_gradients(self):
        x=torch.tensor(self.observations[0]).float().view(1,1,-1)
        timing=torch.ones(1,1,dtype=torch.long)
        baseline,_=self.actor(x,time_emb=timing)
        self.actor.collect_diagnostics=True
        measured,_=self.actor(x,time_emb=timing)
        torch.testing.assert_close(baseline,measured)
        mass=self.actor.last_unmasked_invalid_mass
        self.assertFalse(mass.requires_grad)
        self.assertTrue(((mass>=0)&(mass<=1)).all())
        self.actor.collect_diagnostics=False

    def test_epsilon_uniform_is_over_feasible_tasks(self):
        x = torch.tensor(self.observations[0]).float().view(1,1,-1)
        mask = self.actor.graph_tensors(x)[3]
        logits,_ = self.actor(x, eps=1, time_emb=torch.ones(1,1,dtype=torch.long))
        expected = mask.float()/mask.sum(-1,keepdim=True)
        torch.testing.assert_close(logits.exp(),expected)
        with self.assertRaisesRegex(ValueError, "empty"):
            masked_task_log_probs(torch.zeros(1,3),torch.zeros(1,3,dtype=torch.bool))

    def test_valid_task_likelihood_has_finite_nonzero_gradients(self):
        x = torch.tensor(self.observations[0]).float().view(1,1,-1)
        logits,_ = self.actor(x, time_emb=torch.ones(1,1,dtype=torch.long))
        loss = -logits[..., self.actor.task_ids.index("explore:right")].mean()
        loss.backward()
        for module in (self.encoder, self.actor.task_encoder,
                       self.actor.relation_encoder, self.actor.query):
            grads=[p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads),0)

    def test_busy_controller_preserves_hidden_and_last_task(self):
        mac = TaskGraphMAC(
            self.env, a_mlp_layer_size=[32,32], a_rnn_layer_size=32,
            c_mlp_layer_size=[32,32], c_rnn_layer_size=32,
            init_critic=False, share_encoder=False)
        obs=torch.tensor(np.stack(self.observations)).float().view(1,3,1,32)
        hidden=torch.zeros(1,3,32)
        valid=[[torch.tensor(True),torch.tensor(True),torch.tensor(True)]]
        available=[[torch.ones(1,14) for _ in range(3)]]
        actions,hidden=mac.select_action(None,hidden,valid,available,
                                        obs_history=obs)
        tasks=mac.selected_tasks[0].copy()
        for i in range(3):
            self.assertEqual(actions[0][i],
                int(mac.agents[i].actor_net.task_to_macro_action[tasks[i]]))
        before=hidden.clone()
        valid=[[torch.tensor(False),torch.tensor(True),torch.tensor(False)]]
        actions,hidden=mac.select_action(None,hidden,valid,available,
                                        obs_history=obs)
        self.assertEqual(actions[0][0],-1)
        self.assertEqual(actions[0][2],-1)
        torch.testing.assert_close(hidden[:,0],before[:,0])
        torch.testing.assert_close(hidden[:,2],before[:,2])
        self.assertEqual(mac.selected_tasks[0][0],tasks[0])
        self.assertEqual(mac.selected_tasks[0][2],tasks[2])

    def test_dependency_messages_change_output_and_receive_gradients(self):
        layout, metadata = make_v1_contract(self.env)
        actor = TaskGraphActor(
            32,14,[32,32],32,use_time_emb=True,time_emb_dim=2,
            use_ln=True,robot_id=0,layout=layout,metadata=metadata,
            dependency_steps=1)
        actor.load_state_dict(self.actor.state_dict(),strict=False)
        x=torch.tensor(self.observations[0]).float().view(1,1,-1)
        # Held raw food enables GoToKnife, which has an incoming acquisition edge.
        x[0,0,0:2] = x[0,0,19:21]
        tasks,robots,edges,mask,dependencies=actor.graph_tensors(x)
        keys=actor.task_encoder(tasks)
        disabled=actor.propagate_dependencies(keys,torch.zeros_like(dependencies))
        torch.testing.assert_close(disabled,keys)
        enabled=actor.propagate_dependencies(keys,dependencies)
        self.assertFalse(torch.allclose(enabled,keys))
        logits,_=actor(x,time_emb=torch.ones(1,1,dtype=torch.long))
        baseline,_=self.actor(x,time_emb=torch.ones(1,1,dtype=torch.long))
        self.assertFalse(torch.allclose(logits[...,mask[0,0]],
                                        baseline[...,mask[0,0]]))
        (-logits[...,actor.task_ids.index("explore:right")].mean()).backward()
        gradients=[p.grad for p in actor.dependency_message.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all()
                            for g in gradients))
        self.assertGreater(sum(float(g.abs().sum()) for g in gradients),0)

    def test_runner_records_task_index_and_checks_macro_mapping(self):
        mac = TaskGraphMAC(
            self.env,a_mlp_layer_size=[32,32],a_rnn_layer_size=32,
            init_critic=False,share_encoder=False)
        mac.selected_tasks=[[a.actor_net.task_ids.index("explore:right")
                             for a in mac.agents]]
        runner=object.__new__(TaskGraphRunner)
        runner.controller=mac
        runner.n_agent=3
        runner.obs_last_action=False
        runner.avail_actions=[[torch.ones(1,14) for _ in range(3)]]
        macros=[10,10,10]
        exp=(self.observations,[True]*3,macros,[0]*3,0,
             self.observations,[[1]*14]*3,False,[True]*3,True)
        transition=runner._exp_to_tensor(0,exp,0)
        for i in range(3):
            self.assertEqual(int(transition[3][i]),mac.selected_tasks[0][i])
        wrong=list(exp); wrong[2]=[0,10,10]
        with self.assertRaisesRegex(RuntimeError,"disagrees"):
            runner._exp_to_tensor(0,tuple(wrong),0)


if __name__ == "__main__":
    unittest.main()
