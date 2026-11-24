# Independent 20-seed confirmation (frozen code)

**Frozen research commit**: \`17d02c2300723193aca14c82e6b1b0a4a8b10c4b\`.
**Pinned AirFogSim source**: \`76c0edb4ac6c7b5e09cc146db9928c1bc64bbb60\`.
**Independent seeds**: 200–219 (20 matched runs per scenario); **not the development seeds 0–2**.
**Raw logs and artifacts**: https://github.com/startkz/MU2MV/actions/runs/45299776739

## Correct and on-time task completion
| Scenario | Joint verification + finite PMF-CVaR DP | Random verification + same DP | No-verification DP | Joint − Random paired 95% CI | p (two-sided) |
|---|---:|---:|---:|---|---:|
| Vehicle departure + keyed transmission blackout | 827/1320 = 62.65% | 843/1320 = 63.86% | 1009/1320 = 76.44% | −1.21 pp [−3.39,+0.97] | 0.258 |
| SUMO trajectories, fixed timing | 724/1320 = 54.85% | 710/1320 = 53.79% | 735/1320 = 55.68% | +1.06 pp [−1.28,+3.40] | 0.354 |
| SUMO trajectories, adaptive timing | 717/1320 = 54.32% | 690/1320 = 52.27% | 735/1320 = 55.68% | +2.05 pp [−0.21,+4.30] | 0.073 |

Confidence intervals use the **20 seed-paired differences**, not task-level pseudoreplication. No comparison establishes superiority at the conventional 0.05 level. No-verification DP has the highest completion rate in all three scenarios.

## Verification overhead audit
| Scenario | Joint attempts/completions | Random attempts/completions | Joint measured RB×TTI and CPU Gcycles | Random measured RB×TTI and CPU Gcycles | Pairs with identical realized cost |
|---|---|---|---|---|---|
| Departures/outages | 160/156 | 160/152 | 1585 / 56.00 | 1565 / 55.00 | 13/20 |
| SUMO fixed | 112/101 | 112/102 | 1090 / 38.00 | 1085 / 37.70 | 14/20 |
| SUMO adaptive | 102/71 | 123/101 | 880 / 33.05 | 1140 / 40.45 | 1/20 |

The offered **per-epoch maximum budgets** are the same; adaptive scheduling changes realized attempts and costs. Thus adaptive timing **is not** an equal-*actual-cost* controlled trial. Relative to the same Joint fixed-timing controller, Joint adaptive timing reduced actual verification RB×TTI by 19.3% and CPU usage by 13.0%, while its paired completion-rate difference was −0.53 pp (95% CI [−1.92,+0.86] pp; p=0.433).

## Scientific follow-up (do not turn preliminary curves into TCCN claims)
1. **End-of-horizon censoring**: The terminal snapshot retains unresolved tasks; record their later deadline outcomes by draining the task system after workload arrivals end. Pending is not synonymous with confirmed deadline violation.
2. **Resource blocking**: In the departure/outage experiment, Joint and Random verification left 243/244 ordinary tasks in explicit fail state, versus 11 without verification. Workload failures concentrate on remotely offloaded tasks. Test work-conserving release of reserved RB/CPU as soon as a challenge finishes; account for reservation opportunity costs separately.
3. **Risk calibration**: All reported DDP batches were model-CVaR-feasible even though physical execution failed frequently. The loss-PMF model requires calibration against full end-to-end deadlines, queueing, mobile contact and outages; a model CVaR threshold is not a physical reliability guarantee.
4. **Experimental provenance**: SUMO v1.20 generates small synthetic grid routes and TraCI trajectories, later replayed in AirFogSim with native radio RB and CPU task lifecycle. This is **not field GPS and not online SUMO-AirFogSim co-simulation**. Blackout is injected by suppressing schedulable RBs, not modifying the raw Rayleigh propagation model.
5. **Methodology**: Finite-action, small-batch loss-PMF+CVaR DP is inspired by INFOCOM 2026 Workshops DDP, but is **not a reproduction of all published approximation details**. Vehicle correctness remains an injected hidden Bernoulli variable, not real hardware attestation.

**Conclusion:** Native execution and three independent scenarios are confirmed; Joint VA-RTO superiority is **not confirmed**. Do not tune on held-out seeds 200–219 and later reuse them as unseen validation. Subsequent algorithm/measurement repair needs a new evaluation split/map.

Full detailed analysis and raw ZIPs are attached to this ChatGPT conversation.
