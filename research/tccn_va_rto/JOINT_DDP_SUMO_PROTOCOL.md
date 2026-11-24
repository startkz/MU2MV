# VA-RTO Stage 2: Joint Verification Budget and Distributional DP

**Status:** Executable implementation awaiting validation. This protocol is not a claim of experimental advantage.

## Scientific question
Can uncertainty/contact-aware verification **target, quota, and timing** choices improve deadline-compliant correct completion against random verification, when both share the same native AirFogSim task engine and the same finite-batch loss-PMF/CVaR scheduler?

## Implemented algorithm
- Slow epoch: Every 32 slots, allocate two disjoint validation reservations (12 TTIs each) to VFCNs from Beta posterior variance, reputation disagreement, and estimated contact duration. Optional timing shift is bounded by the epoch.
- Fast slot: For a batch of tasks, enumerate explicit drop, local execution, and eligible vehicle offload options. Model losses as finite distributions; update their PMF by convolution; use CVaR for a batch-level risk constraint. Prune by first-order stochastic dominance only within identical RB and CPU occupancy states. If no risk-feasible option exists, explicitly record risk infeasible and execute a labeled minimum-CVaR fallback.
- Execution: Both validation and ordinary workloads are injected as upstream AirFogSim Task objects. All link transmissions, per-slot RB choices, CPU progress, deadline tests and return transfers are performed by the AirFogSim engine. The service-correctness Bernoulli is hidden and observed only after complete task return.
- Verification resources: Fixed envelopes impose an opportunity cost, but issued/returned challenges and actually used RB×TTI/Gcycles are separate reported outcomes.

## Sources
- AirFogSim upstream pinned commit: 76c0edb4ac6c7b5e09cc146db9928c1bc64bbb60; IEEE TMC 2026 DOI: 10.1109/TMC.2025.3641373.
- Algorithmic reference: Task Offloading across Unreliable Edge Networks via Distributional Dynamic Programming, IEEE INFOCOM 2026 **Workshops**, DOI: 10.1109/INFOCOM59046.2026.11571468. This code independently implements finite batch PMF and CVaR, not the entire published DDP approximate solver.

## Scenario distinction
1. Deterministic trace with explicit vehicle departures and keyed wireless RB suppression (burst outage + independent outage). Physical task progression uses AirFogSim.
2. SUMO simulator generates speed/position/heading trajectories with TraCI. These are replayed into AirFogSim's traffic_mode=real with native AirFogSim RB/CPU execution. This is **SUMO-simulated traffic**, not field-collected GPS, and **not** live coupled AirFogSim/SUMO co-simulation.
3. Same scenario seed and task arrivals are reused for joint, random verification with identical DDP, and no-verification DDP. The first two receive equal **maximum token/envelope** budgets; realized physical overhead may differ under outages.

## Evidence file contract
- summary.csv: tasks, correct on-time jobs, verification issued/completed, actual and reserved RB/CPU, risk-infeasible decisions.
- tasks.csv, checks.csv and verify.csv: task, targeted challenge and completed upload/CPU/return stages.
- dp.csv: per-batch predicted CVaR/feasibility/solver wall time.
- epochs.csv: verification quota and timing records.
- radio.csv: per-slot reserved and allocated resources, link blackout impact.
- paired.json: within-seed rate differences and flags for equality of attempts, reservations and actual overhead.
- sumo.csv, SUMO network and trajectory files: SUMO provenance and input records.

## Scientific limitations
- FSD pruning is limited to identical resource states. The finite-action DP is not proven to be globally optimal for full AirFogSim physics or correlated outage.
- Model estimated CVaR is not a real-world deadline probability certificate; assumptions may be invalid if link failures are correlated or contact dynamics are misspecified.
- Task-result correctness is simulated after physically executed tasks; no real attestation/Byzantine defense is implied.
- If DDP requested RBs cannot be allocated because of old pending tasks, physical allocation must be measured and model-optimality claims narrowed.
- Initial CI experiments are development studies. True held-out confirmation requires frozen code/config and genuinely untouched seeds or maps.
