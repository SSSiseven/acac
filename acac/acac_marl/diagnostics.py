"""Opt-in, evaluator-side metrics. Never alter environment observations/actions."""
from collections import Counter
import numpy as np
from gym import Wrapper
from acac.acac_marl.cores.tg_acac.models import make_v1_contract
from acac.acac_marl.task_graph import TaskGraphBuilder, decode_v1_macro_observation

def contention_counts(targets):
    counts=Counter(t for t in targets if t is not None)
    return sum(n for n in counts.values() if n>1),sum(counts.values())

class PilotMetricsWrapper(Wrapper):
    def __init__(self,env,pilot=False,delivery=False):
        super().__init__(env)
        self.pilot_metrics_enabled=bool(pilot)
        self.delivery_metrics_enabled=bool(delivery or pilot)
        raw=env.unwrapped
        if type(raw).__name__!="Overcooked_MA_V1":
            raise ValueError("Delivery diagnostics audited for Overcooked-MA-v1 only")
        if pilot:
            self.layout,metadata=make_v1_contract(env)
            self.builder=TaskGraphBuilder(metadata)
        self.last_observations=None
        self._clear()

    def _clear(self):
        self.counts=dict(DecisionCount=0,InvalidProposalCount=0,
            AgentSteps=0,IdleAgentSteps=0,ResourceAgentSteps=0,
            ConflictAgentSteps=0,ResourceConflictSteps=0)

    def reset(self,**kwargs):
        obs=self.env.reset(**kwargs)
        self.last_observations=obs
        self._clear()
        return obs

    def _resource(self,robot_id,macro):
        raw=self.env.unwrapped
        if macro is None or macro<0:return None
        name=raw.macroActionName[macro]
        if name.startswith("go to knife"):
            return "knife"+name[-1]
        if name.startswith("get plate"):
            return "plate"+name[-1]
        if name=="chop":
            robot=raw.agent[robot_id]
            # Same primitive direction order as original Chop: right/down/left/up.
            for dx,dy in ((0,1),(1,0),(0,-1),(-1,0)):
                for i,knife in enumerate(raw.knife):
                    if (knife.x,knife.y)==(robot.x+dx,robot.y+dy):
                        held=knife.holding
                        if held is not None and hasattr(held,"chopped") and not held.chopped:
                            return f"knife{i+1}"
        # Collection can use a knife or a plate carrying that food. This is
        # centralized diagnostic state, never provided to a policy.
        if name.startswith("get ") and name[4:] in ("tomato","lettuce","onion"):
            food=getattr(raw,name[4:])[0]
            for i,knife in enumerate(raw.knife):
                if knife.holding is food:return f"knife{i+1}"
            for i,plate in enumerate(raw.plate):
                if food in (plate.containing or ()):return f"plate{i+1}"
        return None

    def step(self,actions):
        raw=self.env.unwrapped
        ready=[a.cur_macro_action_done for a in raw.macroAgent]
        active=[int(actions[i]) if ready[i] else a.cur_macro_action
                for i,a in enumerate(raw.macroAgent)]
        if self.pilot_metrics_enabled:
            obs=self.last_observations
            if obs is None:obs=raw._get_macro_obs()
            for i,is_ready in enumerate(ready):
                if not is_ready:continue
                self.counts["DecisionCount"]+=1
                robots,entities=decode_v1_macro_observation(
                    obs[i],self.layout,self_robot_id=i,self_ready=True)
                graph=self.builder.build(robots,entities)
                feasible=graph.robot_task_mask[i]
                possible=(graph.task_to_macro_action==active[i]) & feasible
                self.counts["InvalidProposalCount"]+=int(not possible.any())
            targets=[self._resource(i,a) for i,a in enumerate(active)]
            conflicted,total=contention_counts(targets)
            self.counts["ConflictAgentSteps"]+=conflicted
            self.counts["ResourceAgentSteps"]+=total
            self.counts["ResourceConflictSteps"]+=int(conflicted>0)
            self.counts["AgentSteps"]+=raw.n_agent
            self.counts["IdleAgentSteps"]+=sum(a==0 for a in active)
        obs,rewards,done,info=self.env.step(actions)
        info=dict(info)
        if self.delivery_metrics_enabled:
            # Audited V1: done is set only after a correct dish is delivered.
            # Also inspect delivery state so future unrelated terminations
            # cannot silently be mistaken for a successful delivery.
            delivered=raw.delivery[0].holding
            names=[] if delivered is None else [f.rawName for f in (delivered.containing or ())]
            recipe="-".join(n for n in ("lettuce","onion","tomato") if n in names)+" salad"
            info["shelf_deliveries"]=int(bool(done) and recipe==raw.task)
        self.last_observations=obs
        return obs,rewards,done,info

    def get_episode_metrics(self):
        counts=dict(self.counts)
        def ratio(n,d):return counts[n]/counts[d] if counts[d] else 0.0
        counts.update(
            InvalidTaskProposalRate=ratio("InvalidProposalCount","DecisionCount"),
            IdleRatio=ratio("IdleAgentSteps","AgentSteps"),
            ResourceConflictRate=ratio("ConflictAgentSteps","ResourceAgentSteps"))
        return counts
