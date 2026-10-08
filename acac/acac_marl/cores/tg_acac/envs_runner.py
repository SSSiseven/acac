"""Use task indices in actor experience; retain original worker macro execution."""
import torch
from acac.acac_marl.cores.acac.envs_runner import EnvsRunner


class TaskGraphRunner(EnvsRunner):
    def _exp_to_tensor(self, env_idx, exp, eps):
        transition = list(super()._exp_to_tensor(env_idx, exp, eps))
        tasks = self.controller.selected_tasks[env_idx]
        for idx, task in enumerate(tasks):
            if task < 0:
                raise RuntimeError("experience has no selected task")
            macro = int(self.controller.agents[idx].actor_net.task_to_macro_action[task])
            if macro != int(exp[2][idx]):
                raise RuntimeError("stored task disagrees with executed macro")
        transition[3] = [torch.tensor(task).view(1, -1) for task in tasks]
        return tuple(transition)
