"""Sample task indices, map to unchanged macros, retain ready-only GRU updates."""
import copy
import torch
from torch.distributions import Categorical
from acac.acac_marl.cores.acac.controller import MAC
from .models import TaskGraphActor, make_v1_contract


class TaskGraphMAC(MAC):
    def __init__(self, *args, tg_embed_dim=32, dependency_steps=0, **kwargs):
        self.tg_embed_dim = tg_embed_dim
        self.dependency_steps = dependency_steps
        self.selected_tasks = None
        super().__init__(*args, **kwargs)

    def _build_agent(self):
        if self.obs_last_action:
            raise ValueError("Stage 3 task-index buffer requires obs_last_action=False")
        layout, metadata = make_v1_contract(self.env)
        super()._build_agent()
        for idx, agent in enumerate(self.agents):
            encoder = agent.actor_net.encoder
            agent.actor_net = TaskGraphActor(
                self._get_actor_input_shape(idx), self.env.n_action[idx],
                self.a_mlp_layer_size, self.a_rnn_layer_size, encoder=encoder,
                use_time_emb=self.time_emb_actor, time_emb_alg=self.time_emb_alg,
                time_emb_dim=self.time_emb_dim, max_timestep=self.max_timestep,
                use_ln=self.use_actor_ln, robot_id=idx, layout=layout,
                metadata=metadata, tg_embed_dim=self.tg_embed_dim,
                dependency_steps=self.dependency_steps).to(self.device)
            agent.actor_tgt_net = copy.deepcopy(agent.actor_net)
            if self.share_encoder:
                agent.encoder = agent.actor_net.encoder
                agent.encoder_tgt = agent.actor_tgt_net.encoder

    def select_action(self, obses, h_states, valids, avail_actions, eps=0.0,
                      test_mode=False, using_tgt_net=False, obs_history=None,
                      attn_mask=None, time_seq=None):
        n_envs = obs_history.shape[0]
        if self.selected_tasks is None or len(self.selected_tasks) != n_envs:
            self.selected_tasks = [[-1]*self.n_agent for _ in range(n_envs)]
        actions = [[-1]*self.n_agent for _ in range(n_envs)]
        with torch.no_grad():
            for idx, agent in enumerate(self.agents):
                ready = [i for i in range(n_envs) if bool(valids[i][idx])]
                if not ready:
                    continue
                actor = agent.actor_tgt_net if using_tgt_net else agent.actor_net
                timing = None
                if self.time_emb_actor:
                    timing = time_seq[ready, idx, -1:].long().to(self.device)
                logits, next_h = actor(
                    obs_history[ready, idx, -1:].to(self.device),
                    h_states[ready, idx].unsqueeze(0), eps=eps,
                    test_mode=test_mode, time_emb=timing)
                # Account for an external macro availability mask if one exists.
                for row, env_idx in enumerate(ready):
                    available = avail_actions[env_idx][idx].reshape(-1).to(self.device)
                    logits[row] = logits[row].masked_fill(
                        available[actor.task_to_macro_action] == 0, -1e9)
                choices = Categorical(logits=logits).sample().reshape(-1)
                for row, env_idx in enumerate(ready):
                    task = int(choices[row])
                    self.selected_tasks[env_idx][idx] = task
                    actions[env_idx][idx] = int(actor.task_to_macro_action[task])
                    h_states[env_idx, idx] = next_h[0, row]
        return actions, h_states
