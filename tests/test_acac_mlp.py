"""Parameter matching without graph observations or encoder expansion."""
import unittest
from unittest.mock import patch
import torch
from test_tg_acac import fixture
from acac.acac_marl.cores.acac.controller import MAC
from acac.acac_marl.cores.acac_mlp.controller import MatchedMLPMAC, trainable_count
from acac.acac_marl.cores.tg_acac.controller import TaskGraphMAC

class MatchedMLPTests(unittest.TestCase):
    def setUp(self):
        self.env,self.obs=fixture()
        self.options=dict(init_critic=False,share_encoder=False,
            a_mlp_layer_size=[32,32],a_rnn_layer_size=32,
            time_emb_actor=True,time_emb_dim=2,max_timestep=200,use_actor_ln=True)

    def test_matching_and_unchanged_encoder(self):
        torch.manual_seed(7)
        original=MAC(self.env,**self.options)
        torch.manual_seed(7)
        matched=MatchedMLPMAC(self.env,**self.options)
        target=TaskGraphMAC(self.env,dependency_steps=1,**self.options)
        for base,mlp,tg in zip(original.agents,matched.agents,target.agents):
            self.assertEqual(mlp.actor_net.fc3.out_features,126)
            self.assertLess(abs(trainable_count(mlp.actor_net)/
                trainable_count(tg.actor_net)-1),0.01)
            for key,value in base.actor_net.encoder.state_dict().items():
                torch.testing.assert_close(value,mlp.actor_net.encoder.state_dict()[key])
            self.assertEqual(tuple(mlp.actor_net.fc4.weight.shape),(14,126))
            for key,value in mlp.actor_net.state_dict().items():
                torch.testing.assert_close(value,mlp.actor_tgt_net.state_dict()[key])

    def test_forward_gradients_and_no_graph_access(self):
        mac=MatchedMLPMAC(self.env,**self.options)
        actor=mac.agents[0].actor_net
        x=torch.tensor(self.obs[0]).float().view(1,1,32)
        with patch("acac.acac_marl.task_graph.builder.TaskGraphBuilder.build",
                   side_effect=AssertionError("MLP forward must not build graphs")):
            logits,_=actor(x,time_emb=torch.ones(1,1,dtype=torch.long),eps=0.1)
            self.assertEqual(tuple(logits.shape),(1,1,14))
            torch.testing.assert_close(logits.exp().sum(-1),torch.ones(1,1))
            (-logits[...,0].mean()).backward()
        for module in (actor.encoder,actor.fc3,actor.fc4):
            grads=[p.grad for p in module.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads),0)

    def test_reject_unmatched_manual_width(self):
        with self.assertRaisesRegex(ValueError,"mismatch"):
            MatchedMLPMAC(self.env,mlp_head_width=32,**self.options)

if __name__=="__main__":
    unittest.main()
