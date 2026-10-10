"""Task pointer over graphs rebuilt solely from recorded local observations."""
import math
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from acac.acac_marl.cores.acac.models import AgentCentricGRUActor, Linear
from acac.acac_marl.task_graph import (
    TaskGraphBuilder, TaskGraphMetadata, MacroEntitySpec,
    V1MacroObservationLayout, decode_v1_macro_observation,
)


def make_v1_contract(env):
    raw = env.unwrapped
    if type(raw).__name__ != "Overcooked_MA_V1" or raw.mapType != "A":
        raise ValueError("TG Stage 3 currently supports V1 map A (ovcA7/ovcAR7) only")
    # Fixed slot order is the audited V1 _createMacroActionItemList contract.
    specs = [MacroEntitySpec(name, "ingredient", name)
             for name in ("tomato", "lettuce", "onion")]
    specs += [MacroEntitySpec(f"plate{i}", "plate") for i in (1, 2)]
    specs += [MacroEntitySpec(f"knife{i}", "knife") for i in (1, 2)]
    specs += [MacroEntitySpec("delivery", "delivery")]
    specs += [MacroEntitySpec(f"robot{i}", "robot", robot_id=i)
              for i in range(raw.n_agent)]
    shape = (raw.xlen, raw.ylen)
    layout = V1MacroObservationLayout(shape, tuple(specs))
    recipe = tuple(name for name in ("tomato", "lettuce", "onion") if name in raw.task)
    walkable = tuple((x, y) for x in range(1, shape[0]-1)
                     for y in range(1, shape[1]-1))
    metadata = TaskGraphMetadata(shape, recipe, tuple(raw.macroActionName),
                                walkable_positions=walkable)
    return layout, metadata


def masked_task_log_probs(scores, mask, eps=0.0, test_mode=False):
    if not mask.any(-1).all():
        raise ValueError("task mask has an empty decision row")
    if not 0 <= eps <= 1:
        raise ValueError("epsilon must be in [0, 1]")
    # Finite sentinel prevents inf*0 in the unchanged PPO padding/entropy code.
    logits = scores.masked_fill(~mask, -1e9)
    log_probs = torch.log_softmax(logits, -1)
    if not test_mode and eps:
        uniform = mask.to(scores.dtype) / mask.sum(-1, keepdim=True)
        probs = (1-eps) * log_probs.exp() + eps * uniform
        log_probs = probs.clamp_min(torch.finfo(scores.dtype).tiny).log()
    return log_probs.masked_fill(~mask, -1e9)


class TaskGraphActor(AgentCentricGRUActor):
    def __init__(self, *args, robot_id, layout, metadata,
                 tg_embed_dim=32, dependency_steps=0, **kwargs):
        super().__init__(*args, **kwargs)
        del self.fc3, self.fc4
        if self.use_ln:
            del self.ln
        self.robot_id, self.layout = robot_id, layout
        self.builder = TaskGraphBuilder(metadata)
        self.dependency_steps = dependency_steps
        if dependency_steps not in (0, 1):
            raise ValueError("Stage 3 dependency_steps must be 0 or 1")
        hidden = self.encoder.gru.hidden_size
        self.query = Linear(hidden, tg_embed_dim)
        self.query_norm = nn.LayerNorm(tg_embed_dim) if self.use_ln else nn.Identity()
        self.task_encoder = nn.Sequential(
            Linear(self.builder.task_feature_dim, tg_embed_dim), nn.LeakyReLU(),
            Linear(tg_embed_dim, tg_embed_dim))
        self.relation_encoder = nn.Sequential(
            Linear(self.builder.robot_feature_dim + self.builder.EDGE_FEATURE_DIM,
                   tg_embed_dim), nn.LeakyReLU())
        if dependency_steps:
            self.dependency_message = nn.Sequential(
                Linear(tg_embed_dim + self.builder.DEPENDENCY_FEATURE_DIM,
                       tg_embed_dim), nn.LeakyReLU(),
                Linear(tg_embed_dim, tg_embed_dim))
        self.scale = math.sqrt(tg_embed_dim)
        self.collect_diagnostics = False
        self.last_unmasked_invalid_mass = None
        self._graph_cache = OrderedDict()
        # Shape template for PPO's zero padding; padding has only Wait feasible.
        dummy = np.zeros(layout.observation_size, dtype=np.float32)
        offset = 19 + 2 * robot_id
        dummy[offset:offset+2] = (1 / metadata.grid_shape[0],
                                1 / metadata.grid_shape[1])
        robots, entities = decode_v1_macro_observation(
            dummy, layout, self_robot_id=robot_id, self_ready=True)
        self.template = self.builder.build(robots, entities)
        self.task_ids = self.template.task_ids
        self.register_buffer("task_to_macro_action",
                             torch.tensor(self.template.task_to_macro_action.copy()))
        self.register_buffer("dependency_edge_index",
            torch.tensor(self.template.task_dependency_edge_index.copy()))
        self.output_dim = len(self.task_ids)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        # Early NoDependency smoke checkpoints predate the static edge buffer.
        # The edge index is derived from the same immutable task vocabulary.
        key = prefix + "dependency_edge_index"
        if key not in state_dict:
            state_dict[key] = self.dependency_edge_index.clone()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def graph_for(self, observation):
        key = observation.tobytes()
        if key in self._graph_cache:
            self._graph_cache.move_to_end(key)
            return self._graph_cache[key]
        if not observation.any():
            graph = self.template
            mask = np.zeros(len(self.task_ids), dtype=np.bool_)
            mask[self.task_ids.index("wait")] = True
        else:
            robots, entities = decode_v1_macro_observation(
                observation, self.layout, self_robot_id=self.robot_id, self_ready=True)
            graph = self.builder.build(robots, entities)
            mask = graph.robot_task_mask[self.robot_id]
        result = (graph.task_features, graph.robot_features[self.robot_id],
                  graph.robot_task_edge_features[self.robot_id], mask,
                  graph.task_dependency_edge_features)
        self._graph_cache[key] = result
        if len(self._graph_cache) > 1024:
            self._graph_cache.popitem(last=False)
        return result

    def graph_tensors(self, raw):
        batch, steps = raw.shape[:2]
        arrays = [self.graph_for(row) for row in raw.detach().cpu().numpy().reshape(
            -1, self.layout.observation_size)]
        tasks, robots, edges, masks, dependencies = zip(*arrays)
        convert = lambda rows, dtype: torch.as_tensor(
            np.stack(rows), dtype=dtype, device=raw.device)
        tasks = convert(tasks, raw.dtype).reshape(batch, steps, self.output_dim, -1)
        robots = convert(robots, raw.dtype).reshape(batch, steps, 1, -1)
        edges = convert(edges, raw.dtype).reshape(batch, steps, self.output_dim, -1)
        masks = convert(masks, torch.bool).reshape(batch, steps, self.output_dim)
        dependencies = convert(dependencies, raw.dtype).reshape(
            batch, steps, -1, self.builder.DEPENDENCY_FEATURE_DIM)
        return tasks, robots, edges, masks, dependencies

    def propagate_dependencies(self, keys, edge_features):
        if not self.dependency_steps:
            return keys
        source, target = self.dependency_edge_index
        active = edge_features[..., :1]
        messages = self.dependency_message(torch.cat(
            (keys.index_select(-2, source), edge_features), -1)) * active
        incoming = torch.zeros_like(keys).index_add(-2, target, messages)
        degree = keys.new_zeros(*keys.shape[:-1], 1).index_add(
            -2, target, active)
        return keys + incoming / degree.clamp_min(1)

    def forward(self, x, h=None, eps=0.0, test_mode=False, time_emb=None):
        tasks, robots, edges, mask, dependencies = self.graph_tensors(x)
        encoder_input = x
        if time_emb is not None:
            encoder_input = torch.cat((x, self.time_embedding(time_emb)), -1)
        history, next_h = self.encoder(encoder_input, h)
        query = self.query_norm(torch.nn.functional.leaky_relu(self.query(history)))
        keys = self.propagate_dependencies(self.task_encoder(tasks), dependencies)
        relations = self.relation_encoder(torch.cat(
            (robots.expand(-1, -1, self.output_dim, -1), edges), -1))
        scores = (query.unsqueeze(-2) * (keys + relations)).sum(-1) / self.scale
        if self.collect_diagnostics:
            self.last_unmasked_invalid_mass = (
                scores.detach().softmax(-1) * (~mask)).sum(-1).cpu()
        return masked_task_log_probs(scores, mask, eps, test_mode), next_h
