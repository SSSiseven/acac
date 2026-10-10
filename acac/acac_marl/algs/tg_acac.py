"""TG policy with the original ACAC learner and asynchronous advantage estimator."""
from .acac import AgentCentricActorCritic
from acac.acac_marl.cores.tg_acac.controller import TaskGraphMAC
from acac.acac_marl.cores.tg_acac.envs_runner import TaskGraphRunner


class TaskGraphACAC(AgentCentricActorCritic):
    def __init__(self, env, *args, tg_embed_dim=32, tg_dependency_steps=0, **kwargs):
        label = kwargs.get("alg_name", "tg_acac_no_dependency")
        if label == "tg_acac_no_dependency" and tg_dependency_steps != 0:
            raise ValueError("NoDependency requires tg_dependency_steps=0")
        if label == "tg_acac" and tg_dependency_steps != 1:
            raise ValueError("Full TG-ACAC requires tg_dependency_steps=1")
        kwargs["_controller_class"] = TaskGraphMAC
        kwargs["_runner_class"] = TaskGraphRunner
        kwargs["_controller_options"] = dict(
            tg_embed_dim=tg_embed_dim, dependency_steps=tg_dependency_steps,
            record_pilot_metrics=kwargs.get('record_pilot_metrics',False))
        super().__init__(env, *args, **kwargs)
