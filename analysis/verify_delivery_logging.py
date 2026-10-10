"""Replay existing baseline weights without training or modifying old outputs."""
import json
from pathlib import Path
import random
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis.profile_stage4 import build
from acac.acac_marl.diagnostics import PilotMetricsWrapper
from acac.acac_marl.cores.acac.envs_runner import EnvsRunner

def evaluate(enabled):
    random.seed(0);np.random.seed(0);torch.manual_seed(0)
    cfg,env,mac,learner=build("acac","ovcA7","cpu")
    checkpoint=ROOT/"data/performance/acac_ovcA7_baseline/ovcA7/ckpt"
    for i,agent in enumerate(mac.agents):
        state=torch.load(checkpoint/f"0_agent_{i}_1.tar",map_location="cpu")
        agent.actor_net.load_state_dict(state["actor_net_state_dict"])
    if enabled:env=PilotMetricsWrapper(env,pilot=True)
    runner=EnvsRunner(env,2,mac,learner.memory,200,0.99,0,parallel=True)
    try:
        runner.run(n_epis=2,test_mode=True)
        metrics=runner.get_diagnostics()
        return {k:v.tolist() for k,v in metrics.items() if k.startswith("Eval/")}
    finally:
        runner.close();env.close()

def main():
    torch.set_num_threads(1)
    before=evaluate(False);after=evaluate(True)
    for key in ("Eval/Return","Eval/EpiLen","Eval/DiscReturn"):
        np.testing.assert_array_equal(before[key],after[key])
    assert sum(after["Eval/NumDelivery"])>0
    assert sum(before["Eval/NumDelivery"])==0
    output=ROOT/"data/pilot_preflight/delivery_replay.json"
    output.write_text(json.dumps(dict(before=before,after=after,
        same_returns_lengths_and_discounted_returns=True),indent=2))
    print("delivery",before["Eval/NumDelivery"],"->",after["Eval/NumDelivery"])
    print("unchanged returns",after["Eval/Return"])

if __name__=="__main__":main()
