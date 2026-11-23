"""Paired statistical audit; each random seed is one independent unit."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from math import isclose
from scipy import stats

def load(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def analyze(folder):
    rows = load(folder / "summary.csv")
    audits = load(folder / "verification_stage_audit.csv")
    grouped = {}
    for r in rows:
        grouped.setdefault(int(r["seed"]), {})[r["method"]] = r
    diffs, actual_equal, details = [], [], []
    for seed, pair in sorted(grouped.items()):
        if set(pair) != {"va_rto", "random_verify"}:
            raise AssertionError(f"missing paired policy for seed {seed}")
        a, b = pair["va_rto"], pair["random_verify"]
        for field in ("jobs", "verifications_issued", "verify_rb_reserved_tti", "verify_cpu_reserved_gc"):
            if not isclose(float(a[field]), float(b[field]), abs_tol=1e-8, rel_tol=0):
                raise AssertionError(f"unequal offered budget: {field}, seed {seed}")
        same_actual = all(isclose(float(a[f]), float(b[f]), abs_tol=1e-8, rel_tol=0)
                          for f in ("verify_rb_active_tti", "verify_cpu_executed_gc"))
        actual_equal.append(same_actual)
        d = float(a["rate"]) - float(b["rate"])
        diffs.append(d)
        details.append(dict(seed=seed, va_rto=float(a["rate"]),
                            random_verify=float(b["rate"]), delta=d,
                            actual_cost_equal=same_actual))
    if len(diffs) < 2:
        raise ValueError("need at least 2 paired seeds")
    for a in audits:
        if a["status"] == "done":
            for field in ("upload_tti", "compute_tti", "return_tti", "rb_active_tti", "cpu_executed_gc"):
                if float(a[field]) <= 0:
                    raise AssertionError(f"{a['task_id']} did not execute {field} in AirFogSim")
    n=len(diffs)
    delta=mean(diffs)
    ci=stats.t.interval(0.95, n-1, loc=delta, scale=stdev(diffs)/(n**0.5))
    result={
       "status":"exploratory development-seed analysis; not held-out confirmation",
       "seed_count":n,
       "total_tasks_per_method":sum(int(x["jobs"]) for x in rows if x["method"]=="va_rto"),
       "mean_success_rate":{m:mean(float(x["rate"]) for x in rows if x["method"]==m)
                            for m in ("va_rto","random_verify")},
       "successful_tasks":{m:sum(int(x["correct_on_time"]) for x in rows if x["method"]==m)
                           for m in ("va_rto","random_verify")},
       "paired_success_delta":delta,
       "paired_95pct_t_CI":list(map(float,ci)),
       "paired_t_p_two_sided":float(stats.ttest_1samp(diffs,0).pvalue),
       "wins":sum(x>0 for x in diffs),
       "ties":sum(x==0 for x in diffs),
       "losses":sum(x<0 for x in diffs),
       "equal_reserved_budget_all_seeds":True,
       "equal_actual_verification_cost_seeds":sum(actual_equal),
       "completed_challenges_audited":sum(a["status"]=="done" for a in audits),
       "paired":details,
    }
    (folder/"paired_stats.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("folder",type=Path)
    data=analyze(p.parse_args().folder)
    print(json.dumps({k:v for k,v in data.items() if k!="paired"}, indent=2))
