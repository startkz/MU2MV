# VA-RTO: verification-aware risk-constrained two-timescale UAV–VFCN offloading

**Status: executable research prototype; not a TCCN experimental result.** This is a clean `research/tccn-va-rto` branch for the rejected MU2MV manuscript, intentionally excluding independent FL poisoning, reputation-based pricing, and two-layer economic games.

## What is actually implemented

- `tccn_va_rto/core.py`: seeded stochastic UAV–vehicle scenario (reflecting road mobility, 3D distance, contact-window first-exit prediction, Shannon-inspired TDMA airtime, real task queues), hidden vehicle correctness probabilities, possibly inflated reported reputations, authenticated challenge-response observations with communication/CPU/energy overhead, Bayesian Beta posteriors, and execution outcomes hidden from the scheduler unless a challenge yields feedback.
- `DistributionalDP`: small-batch resource-constrained **loss-PMF propagation + CVaR**; FSD frontier pruning only among identical resource-usage states; explicit risk-infeasible fallback. Independent per-task failures are an assumption, not a theorem about the radio environment.
- `va_rto`: a slow verification-budget controller updated every `slow_period` slots; fast stage evaluates verification versus no verification using a one-step information-gain proxy, then optimizes the task batch after observing the verification result. One challenge per slot, no more than `max_verifications_per_epoch` per slow epoch.
- Fairly evaluated candidate methods on identical generated workloads and keyed exogenous randomness: `ddp`, `neutral`, `random_verify`, `va_rto`, and `oracle` (uses hidden truth only as a **diagnostic upper-bound reference**, not a deployable approach).
- `tccn_va_rto/airfogsim_trace.py`: **trace integration** with the upstream AirFogSim APIs (`getVehicleTrafficInfos`, `getUAVTrafficInfos`, `getAllToOffloadTaskInfos`). Can capture AirFogSim/SUMO task and position streams and replay them with the separately injected hidden-service process. It is **not** a live policy plugin and has not been end-to-end executed with SUMO in this environment.

## References / boundaries

1. Z. Wei et al., *AirFogSim: A Light-Weight and Modular Simulator for UAV-Integrated Vehicular Fog Computing*, IEEE TMC, 2026. https://github.com/ZhiweiWei-NAMI/AirFogSim ; https://arxiv.org/abs/2409.02518
2. X. Lin et al., *Task Offloading across Unreliable Edge Networks via Distributional Dynamic Programming*, IEEE INFOCOM **2026 Workshops (ISAC-FutureG)**, DOI:10.1109/INFOCOM59046.2026.11571468. The authors' loss PMF/CVaR/FSD method is the methodological baseline; this code is an independent implementation and not their exact published benchmark or code.

## Local commands (Python >=3.10; no third-party packages for reference simulator)

```bash
python -m unittest discover -s tccn_va_rto -t . -v
python -m tccn_va_rto.core --steps 16 --seeds 3 --out results/smoke
python -m tccn_va_rto.core --steps 64 --seeds 20 --out results/development
```

The command writes `summary.csv` (one row per independent scenario/method), `task_records.csv` (per-task trace), and `manifest.json` (config and evidence caveats). Do not report development-seed averages as final statistically validated improvements. CPU/energy coefficients are engineering placeholders, not calibrated device measurements.

### Capture and replay AirFogSim/SUMO (requires upstream environment and its dependency versions)

From a Python process where `AirFogSim` and `SUMO` are installed:

```python
from pathlib import Path
from tccn_va_rto.airfogsim_trace import record_airfogsim
record_airfogsim(Path('examples/config.yaml'), Path('artifacts/airfogsim_trace.jsonl'), max_time=60)
```

Replay the same trace for each method:

```bash
python -m tccn_va_rto.core --steps 60 --seeds 3 --airfogsim-trace artifacts/airfogsim_trace.jsonl --out results/trace_smoke
```

The replay assumes AirFogSim `task_size` is MB (converted to Mbits) and `task_cpu` is giga-cycles, which must be checked against the specific upstream version/configuration. Offloading task service integrity remains injected separately; AirFogSim's default tasks do not provide a real adversarial correctness ground truth. Replay also uses a simplified radio success model; it does **not** preserve complete AirFogSim PHY/MAC interference and queued channel events. A fully coupled scheduler and compatible verification-task execution hooks are additional engineering work before using this as the final paper testbed.

## Science-critical modeling assumptions

- Each batch task must be dropped, executed locally or assigned to exactly one remote VFCN. Explicit drops contribute deterministic lost reward, avoiding the zero-offload objective loophole.
- UAV/vehicle one-slot resource claims and TDMA airtime are conserved. Upload + queue wait + compute + return must fit the predicted deadline; the entire transaction must fit a continuous contact window.
- The slow stage controls challenge budget; fast decisions are made after the challenge response. Challenge airtime is charged before scheduling, and CPU energy/airtime are reported separately.
- Offline task correctness is hidden unless verified. The node's reported history may be misleading; policy cannot inspect the hidden reliability (except the clearly named oracle).
- The small-batch DDP state explicitly tracks the task-loss distribution. FSD dominance is valid only under stated independent-loss and identical future-resource assumptions. The prototype does not implement the INFOCOM paper's general capacity-rounding approximation.
- A fallback when CVaR constraint cannot be satisfied is labeled `risk_infeasible_slots`; **no unconditional deadline or reliability guarantee** is claimed.
- The verification lookahead includes a heuristic continuation-value proxy, not an optimal POMDP solver or proven performance bound.

## Remaining work before TCCN-grade experiments

1. Integrate live AirFogSim scheduler and verified challenge workloads with actual radio/CPU reservation, realistic RSU/UAV return paths, and disappearance/reentry of vehicles.
2. Reproduce the published INFOCOM DDP configurations from the same task and failure model; compare with an independent exact MIP/oracle for small cases, and include matching contemporary risk-aware baselines.
3. Replace independent link failures with measured/trace-driven temporal fading, calibration error and correlated outages; expand to multi-UAV interference.
4. Profile scalability and decision wall-time. Report paired seeds, confidence intervals, ablations, calibration and sensitivity to verification budget, priors, mobility and risk thresholds.
5. Freeze one version and use unseen seeds/maps. Only then move verified statistics into the paper.

## Live AirFogSim verification experiments

The `tccn_va_rto/airfogsim_live.py` entrypoint and [LIVE_AIRFOGSIM_VALIDATION.md](LIVE_AIRFOGSIM_VALIDATION.md) describe the **native AirFogSim TaskManager/RB/CPU** integration and the paired VA-RTO-vs-random verification-cost audit. The live experiment has the fixed upstream version and its own GitHub Actions job; do not conflate it with standalone DDP or trace replay results.