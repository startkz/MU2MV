# Live AirFogSim validation: resource accounting and matched verification budget

## Objective and scope

This is an executable **integration experiment** against AirFogSim at the fixed
source revision `76c0edb4ac6c7b5e09cc146db9928c1bc64bbb60`, not an offline
trace replay. The live simulator's own TaskManager, communication resource block
allocator, CPU computation callback, return-task routing and deadline checks
perform the work. It uses AirFogSim's `traffic_mode=real` with a deterministic
synthetic vehicle trajectory CSV and a SUMO-compatible network file. This is not
a running SUMO process, a real captured mobility trace, nor hardware validation.

The comparison isolates **which vehicle to verify** rather than confounding it
with unequal offered verification resources:

- Same seeded vehicle trajectories and normal-task arrivals across methods.
- Same verification start slots (one admission every 20 TTIs when eligible).
- Fixed request size, return size, CPU cycles, and challenge deadline.
- A verification reserves five real AirFogSim RBs for 15 TTIs plus 1 GHz of
  the target vehicle's CPU allocation for the same interval.
- Unused reserved RBs/CPU are not lent to ordinary tasks, even if a challenge
  completes early. This makes *reserved* RB×TTI and Gcycle budgets equal.
- Actual on-air RB×TTI, CPU executed, response success, deadline violations,
  and task utility can differ and are reported separately. An *equal budget*
  must never be mislabeled *equal packets transmitted*.

The active verification and ordinary jobs are both AirFogSim `Task` objects;
validation is only counted after a return response is marked completed by the
simulator. Hidden service correctness is injected using deterministic keyed
Bernoulli outcomes **after** the simulator completes the radio/compute stages.
Failure to receive the verification is not recorded as a negative service
correctness result; no hidden truth enters the scheduler.

The framework is intended to catch accidental free verification overhead. It
does **not** establish an end-to-end globally optimal policy or implement the
INFOCOM DDP full multi-task dynamic program in the live mode. Normal-job target
selection in this specific integration case uses a risk-filtered one-step
heuristic with a CVaR check; the standalone `core.py` contains the multi-task
loss-distribution DDP reference. These distinct algorithms must not be merged
in method labels or used for an exact DDP baseline claim.

## Execute

In GitHub Actions, the root `.github/workflows/tccn-va-rto.yml` installs
CPU-only dependencies, checks out the fixed upstream revision, executes
contract tests and runs both methods through the live AirFogSim engine.

For a prepared local environment:

```bash
cd research/tccn_va_rto
PYTHONPATH="$PWD:$HOME/AirFogSim" python -m tccn_va_rto.airfogsim_live \
  --airfogsim-root "$HOME/AirFogSim" --steps 80 --seeds 3 \
  --out results/airfogsim_live_paired
```

### Audit data

* `summary.csv`: per-seed paired service success, verification attempts,
  response completions, reserved/used RB×TTI and CPU cycles.
* `task_records.csv`: task IDs, kind, target, AirFogSim final state and
  end-to-end completion status.
* `radio_cpu_tti.csv`: each TTI's RB allocation, reservations and CPU flags.
* `verification_requests.csv`: admitted challenge targets and time slots.
* `manifest.json`: simulator identity, full scenario parameters and evidence
  classification.

All paper-facing results need additional seeds, channel-model parameter checks,
external mobility scenarios, nontrivial risk-budget sensitivity and independent
confirmation. Results from these small engineering cases are not paper claims.
## 20-seed native run and executable statistics

Run `python -m tccn_va_rto.analyze_live results/airfogsim_live_paired` after the upstream simulation to generate `paired_stats.json`. The paired unit of inference is one complete seed scenario; *never* treat individual tasks or TTIs as independent samples.

A completed validation task is accepted in the audit only if it records real uplink RB activity, positive CPU execution, and downlink RB activity. `verification_stage_audit.csv` and `paired_stats.json` are captured in the workflow artifact. Source-based, external service-correctness generation is explicitly distinct from real-world service-integrity ground truth. This native experiment compares the **verification target selector** with a random selector and is not a complete online DDP/POMDP proof. The 20 seeds are development evidence, not an independent held-out evaluation.
