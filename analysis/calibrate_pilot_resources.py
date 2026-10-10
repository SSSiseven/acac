"""Short CPU/CUDA calibration with pilot diagnostics and measured resource use."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import psutil
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis.profile_stage4 import METHODS,build

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prefix",default="pilotcal_261009")
    parser.add_argument("--episodes",type=int,default=32)
    args=parser.parse_args()
    output=ROOT/"data/pilot_preflight"
    output.mkdir(exist_ok=True)
    report=dict(episodes_budget=args.episodes,steps_target=args.episodes*200,
        cuda_available=torch.cuda.is_available(),gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        devices=["cpu"]+(["cuda"] if torch.cuda.is_available() else []),threads=1,results={})
    report["disk_free_gib"]=shutil.disk_usage(ROOT).free/1024**3
    for device in report["devices"]:
        for method in METHODS:
            name=f"{args.prefix}_{method}_{device}"
            if (ROOT/f"data/log/{name}").exists():
                raise ValueError(f"Use a fresh prefix; experiment already exists: {name}")
            cmd=[sys.executable,str(ROOT/"acac/scripts/acac_main.py"),
                "--alg_name",method,"--env_name","ovcAR7","--exp_name",name,
                "--seed","0","--total_epi",str(args.episodes),"--n_env","8",
                "--train_freq","8","--eval_freq",str(args.episodes),
                "--eval_num_epi","20","--device",device,
                "--record_pilot_metrics","true","--record_delivery_metrics","true",
                "--profile_resources","true"]
            print("Starting",name,flush=True)
            started=time.perf_counter();peak_rss=0
            with (output/f"{name}.log").open("w",encoding="utf-8") as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    env=dict(os.environ,OMP_NUM_THREADS="1",MKL_NUM_THREADS="1"))
                while proc.poll() is None:
                    try:
                        parent=psutil.Process(proc.pid)
                        processes=[parent]+parent.children(recursive=True)
                        rss=sum(p.memory_info().rss for p in processes if p.is_running())
                        peak_rss=max(peak_rss,rss)
                    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                    if time.perf_counter()-started>1200:
                        for child in psutil.Process(proc.pid).children(recursive=True):
                            child.kill()
                        proc.kill()
                        raise RuntimeError("Calibration exceeded 20-minute limit")
                    time.sleep(0.5)
            wall=time.perf_counter()-started
            if proc.returncode:
                report["failure"]=dict(command=cmd,returncode=proc.returncode,log=f"{name}.log")
                (output/"calibration.json").write_text(json.dumps(report,indent=2))
                raise RuntimeError(f"Calibration failed: {name}")
            files=list((ROOT/f"data/log/{name}/ovcAR7").glob("*/progress.csv"))
            assert len(files)==1
            with files[0].open(newline="") as handle:rows=list(csv.DictReader(handle))
            assert rows and all(v is not None and math.isfinite(float(v))
                               for r in rows for v in r.values())
            needed=["InvalidTaskProposalRate","ResourceConflictRate","IdleRatio",
                    "DecisionCount","InvalidProposalCount","AgentSteps",
                    "IdleAgentSteps","ResourceAgentSteps","ConflictAgentSteps"]
            for prefix in ("Train","Eval"):
                for metric in needed:
                    assert f"{prefix}/{metric}Average" in rows[0]
                for metric in needed[:3]:
                    assert all(0<=float(r[f"{prefix}/{metric}Average"])<=1 for r in rows)
            _,env,mac,learner=build(method,"ovcAR7","cpu")
            ckpt=ROOT/f"data/performance/{name}/ovcAR7/ckpt"
            generic=torch.load(ckpt/"0_genric_1.tar",map_location="cpu")
            assert generic["epi_count"]==int(float(rows[-1]["Episodes"]))
            def finite(value):
                if isinstance(value,torch.Tensor):return bool(torch.isfinite(value).all())
                if isinstance(value,dict):return all(finite(v) for v in value.values())
                if isinstance(value,(tuple,list)):return all(finite(v) for v in value)
                return True
            assert finite(generic)
            learner.joint_critic_net.load_state_dict(generic["joint_critic_net_state_dict"])
            for i,agent in enumerate(mac.agents):
                saved=torch.load(ckpt/f"0_agent_{i}_1.tar",map_location="cpu")
                assert finite(saved)
                agent.actor_net.load_state_dict(saved["actor_net_state_dict"])
            env.close()
            duration=sum(float(r["SamplingTime"])+float(r["TrainingTime"]) for r in rows)
            steps=int(float(rows[-1]["Timesteps"]))
            resources=json.loads((ROOT/f"data/resource_profiles/{name}/ovcAR7/seed_0.json").read_text())
            directories=[ROOT/f"data/{folder}/{name}" for folder in
                         ("performance","policy_nns","mylog","log","resource_profiles")]
            size=sum(f.stat().st_size for directory in directories
                     for f in directory.rglob("*") if f.is_file())
            metrics={k:float(rows[-1][k]) for k in rows[-1]
                     if any(m in k for m in needed+["NumDelivery"])}
            result=dict(command=cmd,experiment=name,progress_csv=str(files[0].relative_to(ROOT)),
                steps=steps,episodes=int(float(rows[-1]["Episodes"])),
                training_seconds=duration,wall_seconds=wall,
                train_steps_per_second=steps/duration,wall_steps_per_second=steps/wall,
                peak_process_tree_rss_mib=peak_rss/1024**2,output_mib=size/1024**2,
                resources=resources,metrics=metrics,finite_metrics_and_checkpoints=True)
            report["results"][f"{method}/{device}"]=result
            (output/"calibration.json").write_text(json.dumps(report,indent=2)+"\n")
            print("Finished",name,"steps/s",round(result["train_steps_per_second"],2),
                  "wall",round(wall,2),"resources",resources,flush=True)

if __name__=="__main__":main()
