"""Enlarge only the original post-GRU macro-action head."""
import copy
import torch
from torch import nn
from acac.acac_marl.cores.acac.controller import MAC
from acac.acac_marl.cores.acac.models import Linear
from acac.acac_marl.cores.tg_acac.models import TaskGraphActor, make_v1_contract

def trainable_count(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

class MatchedMLPMAC(MAC):
    def __init__(self,*args,tg_embed_dim=32,mlp_head_width=0,
                 mlp_match_tolerance=0.01,**kwargs):
        self.tg_embed_dim=tg_embed_dim
        self.mlp_head_width=mlp_head_width
        self.mlp_match_tolerance=mlp_match_tolerance
        if mlp_head_width < 0 or not 0 <= mlp_match_tolerance < 1:
            raise ValueError("Invalid MLP matching configuration")
        self.match_report=[]
        super().__init__(*args,**kwargs)

    def _build_agent(self):
        super()._build_agent()
        layout,metadata=make_v1_contract(self.env)
        for idx,agent in enumerate(self.agents):
            actor=agent.actor_net
            # Static architecture reference only. Never use graph features in
            # the MLP policy. Preserve RNG state while counting the reference.
            with torch.random.fork_rng(devices=[]):
                reference=TaskGraphActor(self._get_actor_input_shape(idx),
                    self.env.n_action[idx],self.a_mlp_layer_size,self.a_rnn_layer_size,
                    encoder=actor.encoder,use_time_emb=self.time_emb_actor,
                    time_emb_alg=self.time_emb_alg,time_emb_dim=self.time_emb_dim,
                    max_timestep=self.max_timestep,use_ln=self.use_actor_ln,
                    robot_id=idx,layout=layout,metadata=metadata,
                    tg_embed_dim=self.tg_embed_dim,dependency_steps=1)
                target=trainable_count(reference)
            old_head=trainable_count(actor.fc3)+trainable_count(actor.fc4)
            if self.use_actor_ln:
                old_head+=trainable_count(actor.ln)
            unchanged=trainable_count(actor)-old_head
            actions=self.env.n_action[idx]
            slope=self.a_rnn_layer_size+1+actions+2*int(self.use_actor_ln)
            optimal=(target-unchanged-actions)/slope
            candidates={max(1,int(optimal)),max(1,int(optimal)+1)}
            width=self.mlp_head_width or min(candidates,
                key=lambda w:abs(unchanged+slope*w+actions-target))
            predicted=unchanged+slope*width+actions
            mismatch=abs(predicted-target)/target
            if mismatch > self.mlp_match_tolerance:
                raise ValueError(f"MLP parameter mismatch {mismatch:.6f} exceeds "
                                 f"tolerance {self.mlp_match_tolerance}")
            actor.fc3=Linear(self.a_rnn_layer_size,width).to(self.device)
            if self.use_actor_ln:
                actor.ln=nn.LayerNorm(width).to(self.device)
            actor.fc4=Linear(width,actions,act_fn="linear").to(self.device)
            agent.actor_tgt_net=copy.deepcopy(actor)
            if self.share_encoder:
                agent.encoder_tgt=agent.actor_tgt_net.encoder
            self.match_report.append(dict(agent=idx,width=width,target=target,
                actual=trainable_count(actor),relative_error=mismatch))
        print("ACAC-MLP matching:",self.match_report)
