# TCCN new-paper research line (results pending)

**Working title:** Verification-Aware Risk-Constrained Two-Timescale Offloading in Dynamic UAV–Vehicular Fog Networks

**Core question:** With partially observed compute integrity, misleading historical reputation and finite UAV/vehicle contact windows, when and where should an UAV spend scarce challenge verification bandwidth/CPU budget before assigning deadline-sensitive tasks?

**Novelty hypothesis (must be checked):** Unlike a known-link-probability DDP baseline, the controller must acquire information about latent service reliability through costly verification; verification and offloading are coupled because the former changes the posterior, consumes communication/CPU slots, and alters the feasibility region within a moving contact window.

**Model and constraints:** Task value, arrival, return size, CPU cycles and deadline; UAV and vehicle positions/contact time; communication rates and shared resource budgets; latent per-vehicle service probability, a biased prior, a verifiable challenge test. Maximize expected on-time-correct reward minus delay/energy and verification cost subject to resource and CVaR or joint success constraints. Report infeasible slots rather than claim false guarantees.

**Algorithm:** Slow epoch-level budget controller at slots 0,L,2L; fast joint verification and distributional-DP scheduling. Use Beta posterior for simple observation model; evaluate both possible challenge results and link-inconclusive result before choosing. After costly challenge, re-run risk-constrained batch scheduling using updated posterior and reduced resource budget. Explicitly name the discounted future information term a proxy pending a rigorous proof.

**Closest work:** AirFogSim (TMC 2026) is the simulation platform, not a novel offloading algorithm. Lin et al. (INFOCOM 2026 Workshops) is a risk-constrained, known-link-probability DDP baseline, not a partially observed compute-trust and active-verification solver. Need full comparison against other TCCN/UAV risk and trust approaches before claiming novelty.

**Paper outline:** I. Introduction and gap; II. Related Work; III. UAV–VFCN system, verification threat model and objective; IV. Verification-aware belief dynamics and feasibility; V. Two-timescale risk-aware controller; VI. Conditional theoretical analysis (only proved statements); VII. AirFogSim/SUMO experiments, baseline parity, task integrity, tail risk, cost, mobility and statistics; VIII. Limitations and Conclusion.

**Research hypotheses to test (no results asserted):** H1: selective challenge feedback improves on-time-correct completion under manipulated reputation at matched budget; H2: risk-constrained PMF scheduling reduces tail loss versus expectation-only policy; H3: contact-window modeling affects deadline success under high speed; H4: gains exceed incurred verification network/energy overhead and remain under calibration drift.

**Non-claims:** No robust-FL results, game equilibrium, general DDP optimality at arbitrary scale, unconditional CVaR guarantees under misspecification, or hardware validated UAV energy savings.