"""Reproducible four-method profiling; source experiment CSVs are read-only."""
import argparse
import csv
import inspect
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from acac.env import OvercookedMacEnvWrapper
import gym
from acac.scripts.params.utils import get_env_params
from acac.acac_marl.cores.acac.controller import MAC
from acac.acac_marl.cores.acac_mlp.controller import MatchedMLPMAC
from acac.acac_marl.cores.tg_acac.controller import TaskGraphMAC
from acac.acac_marl.cores.acac.learner_acac import Learner_ACAC
from acac.acac_marl.cores.acac.memory import Memory_epi
METHODS=["acac","acac_mlp","tg_acac_no_dependency","tg_acac"]

def config(method,env_name):
    result={}
    for path in [ROOT/"acac/scripts/params/default.json",
                 ROOT/f"acac/scripts/params/alg/{method}.json",
                 ROOT/f"acac/scripts/params/env/{env_name}.json"]:
        result.update(json.loads(path.read_text()))
    return result

def build(method,env_name,device):
    cfg=config(method,env_name)
    env=OvercookedMacEnvWrapper(gym.make(cfg["env_id"],
        **get_env_params(argparse.Namespace(**cfg))))
    options={k:cfg[k] for k in inspect.signature(MAC.__init__).parameters
             if k in cfg}
    options.update(device=device,init_critic=False,max_timestep=cfg["env_terminate_step"])
    if method=="acac":
        mac=MAC(env,**options)
    elif method=="acac_mlp":
        mac=MatchedMLPMAC(env,tg_embed_dim=cfg["tg_embed_dim"],
            mlp_head_width=cfg["mlp_head_width"],
            mlp_match_tolerance=cfg["mlp_match_tolerance"],**options)
    else:
        mac=TaskGraphMAC(env,tg_embed_dim=cfg["tg_embed_dim"],
            dependency_steps=cfg["tg_dependency_steps"],**options)
    memory=Memory_epi(env.obs_size,env.n_action,cfg["obs_last_action"],size=4)
    options={k:cfg[k] for k in inspect.signature(Learner_ACAC.__init__).parameters
             if k in cfg}
    options["device"]=device
    learner=Learner_ACAC(env,mac,memory,**options)
    return cfg,env,mac,learner

def count(modules,trainable=True):
    unique={id(p):p for module in modules for p in module.parameters()
            if not trainable or p.requires_grad}
    return sum(p.numel() for p in unique.values())

def common_observations(env,size):
    # Fixed random-macro rollout from the original environment, reused by
    # all methods. Benchmark requests one ready decision for each robot.
    rng=np.random.RandomState(17)
    pool=[]
    np.random.seed(17)
    random.seed(17)
    obs=env.reset()
    for step in range(size):
        pool.append(torch.tensor(np.stack(obs)).float().view(1,env.n_agent,1,-1))
        obs,_,done,_=env.step([int(rng.randint(n)) for n in env.n_action])
        if done or (step+1)%200==0:
            obs=env.reset()
    return pool

def latency(mac,pool,device,cold,repeats):
    hidden=torch.zeros(1,mac.n_agent,mac.a_rnn_layer_size,device=device)
    avail=[[torch.ones(1,n) for n in mac.env.n_action]]
    ready=[[torch.tensor(True) for _ in mac.agents]]
    timing=torch.ones(1,mac.n_agent,1,dtype=torch.long)
    def synchronize():
        if device=="cuda":
            torch.cuda.synchronize()
    def call(obs):
        mac.select_action(None,hidden.clone(),ready,avail,
            obs_history=obs,time_seq=timing,eps=0.05,test_mode=False)
    for obs in pool:
        call(obs)
    synchronize()
    samples=[]
    for _ in range(repeats):
        for obs in pool:
            if cold:
                for agent in mac.agents:
                    cache=getattr(agent.actor_net,"_graph_cache",None)
                    if cache is not None:
                        cache.clear()
            synchronize()
            start=time.perf_counter()
            call(obs)
            synchronize()
            samples.append((time.perf_counter()-start)*1000)
    return dict(mean_ms=float(np.mean(samples)),median_ms=float(np.median(samples)),
                p95_ms=float(np.percentile(samples,95)),samples=len(samples))

def train_benchmark(method,env_name,output,prefix):
    name=f"{prefix}_{method}_timing16"
    existing=list((ROOT/f"data/log/{name}/{env_name}").glob("*/progress.csv"))
    if existing:
        raise RuntimeError(f"Refusing ambiguous/reused experiment name: {name}")
    command=[sys.executable,str(ROOT/"acac/scripts/acac_main.py"),
        "--alg_name",method,"--exp_name",name,"--env_name",env_name,"--seed","0",
        "--total_epi","16","--n_env","4","--train_freq","4",
        "--eval_freq","8","--eval_num_epi","2","--device","cpu"]
    print("Training",method,flush=True)
    started=time.perf_counter()
    with (output/f"{name}.log").open("w",encoding="utf-8") as log:
        process=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS="1",MKL_NUM_THREADS="1"))
    wall=time.perf_counter()-started
    if process.returncode:
        raise RuntimeError(f"Training failed; see {name}.log")
    files=list((ROOT/f"data/log/{name}/{env_name}").glob("*/progress.csv"))
    if len(files)!=1:
        raise ValueError("Expected exactly one CSV")
    with files[0].open(newline="") as handle:
        rows=list(csv.DictReader(handle))
    if not rows or not all(v is not None and math.isfinite(float(v))
                          for row in rows for v in row.values()):
        raise ValueError("Missing/nonfinite training metric")
    cfg=json.loads(files[0].with_name("variant.json").read_text())
    expected={"alg_name":method,"seed":0,"total_epi":16,"n_env":4,
        "train_freq":4,"eval_freq":8,"eval_num_epi":2,"device":"cpu"}
    assert all(cfg[k]==v for k,v in expected.items())
    ckpt=ROOT/f"data/performance/{name}/{env_name}/ckpt"
    generic=torch.load(ckpt/"0_genric_1.tar",map_location="cpu")
    assert generic["epi_count"]==int(float(rows[-1]["Episodes"]))
    def finite(value):
        if isinstance(value,torch.Tensor):
            return bool(torch.isfinite(value).all())
        if isinstance(value,dict):
            return all(finite(v) for v in value.values())
        if isinstance(value,(tuple,list)):
            return all(finite(v) for v in value)
        return True
    assert finite(generic)
    checkpoints=[torch.load(ckpt/f"0_agent_{i}_1.tar",map_location="cpu") for i in range(3)]
    assert all(finite(c) for c in checkpoints)
    _,env,mac,learner=build(method,env_name,"cpu")
    for agent,state in zip(mac.agents,checkpoints):
        agent.actor_net.load_state_dict(state["actor_net_state_dict"])
        agent.actor_tgt_net.load_state_dict(state["actor_net_state_dict"])
    learner.joint_critic_net.load_state_dict(generic["joint_critic_net_state_dict"])
    pool=common_observations(env,4)
    latency(mac,pool,"cpu",False,1)
    env.close()
    sample=sum(float(r["SamplingTime"]) for r in rows)
    update=sum(float(r["TrainingTime"]) for r in rows)
    steps=int(float(rows[-1]["Timesteps"]))
    return dict(command=command,progress_csv=str(files[0].relative_to(ROOT)),
        config=cfg,steps=steps,episodes=int(float(rows[-1]["Episodes"])),
        sampling_seconds=sample,ppo_seconds=update,wall_seconds=wall,
        train_steps_per_second=steps/(sample+update),
        end_to_end_steps_per_second=steps/wall,
        checkpoint_mib=sum(p.stat().st_size for p in ckpt.glob("*"))/1024**2,
        finite_metrics_checkpoints=True,checkpoint_restore=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--env-name",default="ovcAR7")
    parser.add_argument("--output-dir",default="data/stage4")
    parser.add_argument("--run-prefix",default="stage4")
    parser.add_argument("--train",action="store_true")
    parser.add_argument("--training-only",action="store_true",
        help="Reuse existing inference evidence, execute only training calibration")
    parser.add_argument("--repeats",type=int,default=5)
    args=parser.parse_args()
    torch.set_num_threads(1)
    output=ROOT/args.output_dir
    output.mkdir(parents=True,exist_ok=True)
    report=dict(environment=args.env_name,seed=0,torch=torch.__version__,
        python=sys.version,platform=platform.platform(),cpu=platform.processor(),
        cuda_available=torch.cuda.is_available(),threads=1,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        protocol="Online networks only; unique trainable parameters, exclude target copies. "
        "Inference includes local graph build/cache lookup, tensor transfers, GRU, head, "
        "mask and categorical sampling for three ready robots, batch=1, synchronized "
        "CUDA timing; common 64-observation original random-macro rollout, 5 blocks. "
        "Cold cache clears excluded from timing; warm cache primed with full pool. "
        "Training CPU only: identical 3200-step target and original budget accounting; "
        "throughput actual env steps/(sampling+PPO); wall includes setup, evaluation, saves.",
        repeats=args.repeats,observation_rollout_seed=17,inference_time_embedding_index=1,methods={})
    if args.training_only:
        report=json.loads((output/"profile.json").read_text())
        if report["environment"] != args.env_name:
            raise ValueError("Existing inference environment differs")
    for method in ([] if args.training_only else METHODS):
        torch.manual_seed(0)
        cfg,env,mac,learner=build(method,args.env_name,"cpu")
        actors=[a.actor_net for a in mac.agents]
        critic=learner.joint_critic_net
        record=dict(actor_trainable=count(actors),critic_trainable=count([critic]),
            total_trainable=count(actors+[critic]),total_including_frozen=count(actors+[critic],False),
            per_actor_trainable=[count([a]) for a in actors],config=cfg,latency={})
        if method=="acac_mlp":
            record["matching"]=mac.match_report
        pool=common_observations(env,64)
        inputs=np.stack([x.numpy() for x in pool])
        digest=hashlib.sha256(inputs.tobytes()).hexdigest()
        if "observation_sha256" in report:
            assert report["observation_sha256"]==digest
        else:
            report["observation_sha256"]=digest
            np.save(output/"common_observations.npy",inputs)
        for device in ["cpu"]+(["cuda"] if torch.cuda.is_available() else []):
            for a in mac.agents:
                a.actor_net.to(device)
            mac.device=device
            for a in mac.agents:
                a.actor_net.eval()
            record["latency"][device]=dict(
                warm=latency(mac,pool,device,False,args.repeats),
                cold=latency(mac,pool,device,True,args.repeats))
        if method.startswith("tg"):
            a=actors[0]
            record["graph_dimensions"]=dict(robot=a.builder.robot_feature_dim,
                task=a.builder.task_feature_dim,relation=a.builder.EDGE_FEATURE_DIM,
                dependency=a.builder.DEPENDENCY_FEATURE_DIM)
        env.close()
        report["methods"][method]=record
        print(method,record["total_trainable"],record["latency"],flush=True)
    critic_counts={r["critic_trainable"] for r in report["methods"].values()}
    assert len(critic_counts)==1
    matched=report["methods"]["acac_mlp"]["actor_trainable"]
    target=report["methods"]["tg_acac"]["actor_trainable"]
    assert abs(matched/target-1)<0.01
    # Save inference evidence before training so failures remain inspectable.
    path=output/"profile.json"
    path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    if args.train:
        for method in METHODS:
            report["methods"][method]["training"]=train_benchmark(method,args.env_name,output,args.run_prefix)
            path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    rows=[]
    for method,r in report["methods"].items():
        row=dict(method=method,actor=r["actor_trainable"],critic=r["critic_trainable"],
            total=r["total_trainable"],cpu_warm_ms=r["latency"]["cpu"]["warm"]["median_ms"],
            cpu_cold_ms=r["latency"]["cpu"]["cold"]["median_ms"])
        if "cuda" in r["latency"]:
            row["cuda_warm_ms"]=r["latency"]["cuda"]["warm"]["median_ms"]
            row["cuda_cold_ms"]=r["latency"]["cuda"]["cold"]["median_ms"]
        if "training" in r:
            row["train_steps_per_second"]=r["training"]["train_steps_per_second"]
            row["wall_steps_per_second"]=r["training"]["end_to_end_steps_per_second"]
        rows.append(row)
    with (output/"summary.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved",path,flush=True)

if __name__=="__main__":
    main()
