"""Paired-seed statistical audit for joint VA-RTO live experiments.

Never equate nominal challenge opportunities with realized communication/CPU cost.
Run after completing a campaign; outputs are derived solely from its raw CSV.
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
from statistics import mean, stdev
from scipy import stats

def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

def evaluate(folder):
    rows=csv_rows(folder/"summary.csv")
    stages=csv_rows(folder/"verify.csv")
    paired={}
    for row in rows:
        seed=int(row["seed"])
        method=row["method"]
        if method in paired.setdefault(seed,{}):
            raise AssertionError(f"duplicate {seed}/{method}")
        paired[seed][method]=row
    if not paired:raise AssertionError("empty experiment")
    delta=[];cost=[];differences=[]
    for seed,pair in sorted(paired.items()):
        if set(pair)!={"joint","random_verify","no_verify"}:
            raise AssertionError(f"missing method at seed {seed}")
        a,b=pair["joint"],pair["random_verify"]
        if int(a["jobs"])!=int(b["jobs"]):
            raise AssertionError(f"unpaired workload seed {seed}")
        offered_equal=True # maximum budget, as set by a single shared JointConfig
        reservations_equal=all(math.isclose(float(a[f]),float(b[f]),abs_tol=1e-8,rel_tol=0)
            for f in ("rb_reserved","cpu_reserved"))
        attempts_equal=int(a["verifies"])==int(b["verifies"])
        actual_equal=all(math.isclose(float(a[f]),float(b[f]),abs_tol=1e-8,rel_tol=0)
            for f in ("rb_active","cpu_executed"))
        d=float(a["rate"])-float(b["rate"])
        delta.append(d)
        cost.append((attempts_equal,reservations_equal,actual_equal))
        differences.append({"seed":seed,"joint":float(a["rate"]),"random":float(b["rate"]),
            "difference":d,"attempts_equal":attempts_equal,"reserved_equal":reservations_equal,
            "actual_resource_equal":actual_equal})
    for row in stages:
        if row["status"]=="done":
            for key in ("up","cpu","down","rb","cpu_gc"):
                if float(row[key])<=0:
                    raise AssertionError(f"native verification stage missing {row['task_id']}/{key}")
    n=len(delta);avg=mean(delta)
    if n>1:
        half=float(stats.t.ppf(.975,n-1)*stdev(delta)/(n**.5))
        ci=[avg-half,avg+half]
        p=float(stats.ttest_1samp(delta,0).pvalue)
    else:
        ci=[None,None];p=None
    aggregate={}
    for method in ("joint","random_verify","no_verify"):
        selected=[p[method] for p in paired.values()]
        total=sum(int(z["jobs"]) for z in selected)
        correct=sum(int(z["correct"]) for z in selected)
        aggregate[method]={"jobs":total,"correct":correct,
            "pooled_correct_rate":correct/max(1,total),
            "mean_seed_correct_rate":mean(float(z["rate"]) for z in selected),
            "verification_attempts":sum(int(z["verifies"]) for z in selected),
            "verification_completions":sum(int(z["verify_success"]) for z in selected),
            "rb_reserved_tti":sum(float(z["rb_reserved"]) for z in selected),
            "rb_active_tti":sum(float(z["rb_active"]) for z in selected),
            "cpu_reserved_gc":sum(float(z["cpu_reserved"]) for z in selected),
            "cpu_actual_gc":sum(float(z["cpu_executed"]) for z in selected),
            "risk_infeasible_batches":sum(int(z["risk_infeasible"]) for z in selected)}
    out={"status":"independent seeded simulation only if code/config were frozen before running",
         "seed_count":n,"seeds":sorted(paired),"paired_delta_mean":avg,
         "paired_95pct_t_CI":ci,"paired_p_two_sided_unadjusted":p,
         "pair_wins":sum(v>0 for v in delta),"ties":sum(v==0 for v in delta),
         "losses":sum(v<0 for v in delta),
         "equal_attempts_pairs":sum(x[0] for x in cost),
         "equal_reserved_cost_pairs":sum(x[1] for x in cost),
         "equal_realized_cost_pairs":sum(x[2] for x in cost),
         "budget_interpretation":"same per-epoch maximum number/tokens; realized costs may differ",
         "native_verify_stages_completed":sum(z["status"]=="done" for z in stages),
         "aggregate":aggregate,"pairs":differences}
    (folder/"independent_statistics.json").write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({k:v for k,v in out.items() if k!="pairs"},indent=2))
    return out

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("folder",type=Path)
    evaluate(a.parse_args().folder)
