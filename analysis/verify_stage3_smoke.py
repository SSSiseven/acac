"""Verify finite smoke metrics/checkpoints without modifying experiment CSVs."""
import argparse
import csv
import json
import math
from pathlib import Path
import torch


def verify(exp_name):
    root=Path(__file__).resolve().parents[1]
    runs=sorted((root/"data/log"/exp_name/"ovcA7").glob("*/progress.csv"))
    if len(runs) != 1:
        raise ValueError(f"expected one run for {exp_name}, found {len(runs)}")
    with runs[0].open(newline="") as handle:
        rows=list(csv.DictReader(handle))
    if not rows:
        raise ValueError("no progress rows")
    for row in rows:
        for key,value in row.items():
            if value is None or not math.isfinite(float(value)):
                raise ValueError(f"nonfinite/missing CSV value {key}: {value}")
    variant=json.loads(runs[0].with_name("variant.json").read_text())
    ckpt=root/"data/performance"/exp_name/"ovcA7/ckpt"
    generic=torch.load(ckpt/"0_genric_1.tar",map_location="cpu")
    if generic["epi_count"] != int(float(rows[-1]["Episodes"])):
        raise ValueError("checkpoint and CSV episode counts differ")
    actor_checkpoints = [torch.load(ckpt/f"0_agent_{i}_1.tar",map_location="cpu")
                         for i in range(3)]
    def finite_optimizer(state):
        if isinstance(state, torch.Tensor):
            return bool(torch.isfinite(state).all())
        if isinstance(state, dict):
            return all(finite_optimizer(value) for value in state.values())
        if isinstance(state, (tuple, list)):
            return all(finite_optimizer(value) for value in state)
        return True
    optimizers=[generic["joint_critic_optimizer_state_dict"]] + [
        checkpoint["actor_optimizer_state_dict"] for checkpoint in actor_checkpoints]
    if not all(finite_optimizer(state) for state in optimizers):
        raise ValueError("nonfinite optimizer state")
    for label,state in [("critic",generic["joint_critic_net_state_dict"])] + [
        (f"actor{i}",checkpoint["actor_net_state_dict"])
        for i,checkpoint in enumerate(actor_checkpoints)]:
        for key,tensor in state.items():
            if not torch.isfinite(tensor).all():
                raise ValueError(f"nonfinite checkpoint {label}.{key}")
    if variant["alg_name"].startswith("tg_acac"):
        import sys
        sys.path.insert(0,str(root))
        from acac.env.overcooked_env.overcooked_MA_V1 import Overcooked_MA_V1
        from acac.acac_marl.cores.tg_acac.controller import TaskGraphMAC
        env=Overcooked_MA_V1((7,7),"lettuce-onion-tomato salad",
            {"subtask finished":10,"correct delivery":200,
             "wrong delivery":-5,"step penalty":-0.1},
            map_type="A",n_agent=3,rand_start=False)
        observations=env.reset()
        mac=TaskGraphMAC(env,init_critic=False,share_encoder=False,
            a_mlp_layer_size=variant["a_mlp_layer_size"],
            a_rnn_layer_size=variant["a_rnn_layer_size"],
            time_emb_actor=variant["time_emb_actor"],
            time_emb_alg=variant["time_emb_alg"],time_emb_dim=variant["time_emb_dim"],
            use_actor_ln=variant["use_actor_ln"],
            tg_embed_dim=variant["tg_embed_dim"],
            dependency_steps=variant["tg_dependency_steps"])
        with torch.no_grad():
            for i,checkpoint in enumerate(actor_checkpoints):
                actor=mac.agents[i].actor_net
                actor.load_state_dict(checkpoint["actor_net_state_dict"])
                x=torch.tensor(observations[i]).float().view(1,1,-1)
                timing=torch.ones(1,1,dtype=torch.long) if actor.use_time_emb else None
                logits,_=actor(x,test_mode=True,time_emb=timing)
                if not torch.isfinite(logits).all():
                    raise ValueError("nonfinite restored actor output")
    return dict(
        experiment=exp_name,seed=variant["seed"],
        algorithm=variant["alg_name"],
        dependency_steps=variant.get("tg_dependency_steps",0),
        progress=str(runs[0].relative_to(root)),
        episodes=generic["epi_count"],timesteps=float(rows[-1]["Timesteps"]),
        training_evaluation_seconds=float(rows[-1]["TotalTime"]),
        final_eval_return=float(rows[-1]["Eval/ReturnAverage"]),
        checkpoint_mib=sum(p.stat().st_size for p in ckpt.glob("*.tar"))/2**20,
        finite_metrics_and_checkpoints=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("experiments",nargs="+")
    args=parser.parse_args()
    summaries=[verify(name) for name in args.experiments]
    path=Path(__file__).resolve().parents[1]/"data/stage3/verified_smokes.json"
    path.write_text(json.dumps(summaries,indent=2),encoding="utf-8")
    print(json.dumps(summaries,indent=2))
