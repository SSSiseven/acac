"""Read-only pilot metrics aggregation with an explicit common AUC interval."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
METHODS=["acac","acac_mlp","tg_acac_no_dependency","tg_acac"]

def curve_auc(x,y,start,end):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    if len(x)<2 or not np.all(np.diff(x)>0) or not np.isfinite(y).all():
        raise ValueError("Need a finite increasing curve with at least two points")
    if not x[0]<=start<end<=x[-1]:
        raise ValueError("AUC cannot extrapolate beyond recorded evaluations")
    points=np.concatenate(([start],x[(x>start)&(x<end)],[end]))
    values=np.interp(points,x,y)
    auc=float(np.trapz(values,points))
    return auc,auc/(end-start)

def pooled_rate(rows,numerator,denominator,prefix="Eval"):
    # Weight interval means by the actual number of completed eval episodes.
    n=sum(float(r[f"{prefix}/{numerator}Average"]) *
          float(r.get(f"{prefix}/EpisodeCountAverage",1)) for r in rows)
    d=sum(float(r[f"{prefix}/{denominator}Average"]) *
          float(r.get(f"{prefix}/EpisodeCountAverage",1)) for r in rows)
    return n/d if d else 0.0

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--log-root",type=Path,required=True,
        help="Logging root containing the pilot experiment directories")
    parser.add_argument("--experiment-prefix",default="pilot_ovcAR7_")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"data/pilot_analysis")
    parser.add_argument("--budget-steps",type=int,default=10000000)
    args=parser.parse_args()
    selected={};configs=[]
    for path in args.log_root.rglob("progress.csv"):
        if not path.relative_to(args.log_root).parts[0].startswith(args.experiment_prefix):continue
        cfg=json.loads(path.with_name("variant.json").read_text())
        key=(cfg["alg_name"],cfg["seed"])
        if key[0] not in METHODS or key[1] not in (0,1,2):continue
        if key in selected:raise ValueError(f"Duplicate method/seed: {key}")
        with path.open(newline="") as handle:rows=list(csv.DictReader(handle))
        if len(rows)<2:raise ValueError(f"Incomplete run: {path}")
        if any(not math.isfinite(float(v)) for r in rows for v in r.values()):
            raise ValueError(f"Nonfinite data: {path}")
        if float(rows[-1]["Timesteps"])<args.budget_steps:
            raise ValueError(f"Budget not reached: {path}")
        expected={"env_name":"ovcAR7","n_env":8,"train_freq":8,"eval_freq":200,
            "eval_num_epi":20,"total_epi":50000,"record_pilot_metrics":True}
        if any(cfg.get(k)!=v for k,v in expected.items()):
            raise ValueError(f"Unexpected pilot configuration: {path}")
        configs.append(cfg)
        selected[key]=(path,rows)
    if len(selected)!=12:raise ValueError(f"Need all 4 methods x 3 seeds; got {len(selected)}")
    for field in ("device","gamma","step_penalty","GAE_lambda","n_train_repeat",
                  "clip_ratio","a_lr","c_lr","share_encoder","time_emb_actor"):
        if len({str(c[field]) for c in configs})!=1:
            raise ValueError(f"Fairness mismatch: {field}")
    start=max(float(rows[0]["Timesteps"]) for _,rows in selected.values())
    end=args.budget_steps
    results=[]
    for (method,seed),(path,rows) in selected.items():
        auc,normalized=curve_auc([float(r["Timesteps"]) for r in rows],
            [float(r["Eval/ReturnAverage"]) for r in rows],start,end)
        record=dict(method=method,seed=seed,source=str(path),
            auc_start=start,auc_end=end,auc=auc,normalized_auc=normalized,
            final_return=float(rows[-1]["Eval/ReturnAverage"]),
            final_delivery=float(rows[-1]["Eval/NumDeliveryAverage"]),
            final_episode_length=float(rows[-1]["Eval/EpiLenAverage"]),
            actual_steps=float(rows[-1]["Timesteps"]),
            invalid_proposal_rate=pooled_rate(rows,"InvalidProposalCount","DecisionCount"),
            conflict_rate=pooled_rate(rows,"ConflictAgentSteps","ResourceAgentSteps"),
            idle_ratio=pooled_rate(rows,"IdleAgentSteps","AgentSteps"),
            inference_ms=float(np.mean([float(r["Eval/InferenceTimeMsAverage"]) for r in rows])))
        if "Eval/PreMaskInvalidTaskMassAverage" in rows[0]:
            record["premask_invalid_mass"]=pooled_rate(rows,"PreMaskInvalidTaskMassSum","PreMaskDecisionCount")
        results.append(record)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/"per_seed.json").write_text(json.dumps(results,indent=2))
    summaries={}
    for method in METHODS:
        values=[r for r in results if r["method"]==method]
        summaries[method]={k:dict(mean=float(np.mean([r[k] for r in values])),
            std=float(np.std([r[k] for r in values],ddof=1)))
            for k in ("final_return","normalized_auc","final_delivery",
                      "final_episode_length","invalid_proposal_rate","conflict_rate","idle_ratio")}
    (args.output_dir/"summary.json").write_text(json.dumps(summaries,indent=2))
    print("Saved",args.output_dir)

if __name__=="__main__":main()
