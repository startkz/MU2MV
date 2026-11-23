# VA-RTO: verification-aware risk-constrained two-timescale offloading

**Development specification — 2026-11-23. No paper-grade performance or theoretical guarantees are claimed.**

## Research problem and difference from references

- **AirFogSim (Wei et al., IEEE TMC, 2026)** supplies a UAV/vehicle mobility and computing simulation interface. The current branch supplies an independent, standard-library-only reference simulator and an AirFogSim task/position **trace bridge**, not an integrated upstream online scheduler. Upstream code: https://github.com/ZhiweiWei-NAMI/AirFogSim .
- **Distributional dynamic programming (Lin et al., IEEE INFOCOM 2026 Workshops)** motivates an explicitly propagated probability mass function (PMF) for aggregate task-loss and CVaR constrained optimization. The present DDP is an independent small-batch implementation, not a numerical reproduction of the authors' benchmarks. DOI: 10.1109/INFOCOM59046.2026.11571468 .
- **Our research hypothesis:** verification is a *decision* that modifies future information, current bandwidth/CPU feasibility, and execution reliability estimation; not a passive post-task reputation update. This hypothesis requires a strong independent comparison and a formal conditional analysis before publication.

## Formal service and observation model

Let $u$ be a UAV and $j$ a VFCN. Task $i$ arriving at slot $t$ has input $b_i$ [Mbit], returned result $b_i^r$ [Mbit], cycles $c_i$ [Gcycle], deadline $d_i$ [s] relative to arrival, and loss weight $w_i$. UAV local execution is an explicit action $j=-1$ and dropping is an explicit action $j=-2$; each task has exactly one assignment. A VFCN is selected only if its predicted contact-window first exit covers the entire upload, compute, and return transaction. For a remote option, the nominal completion time from its task arrival is

$$T_{ij} = t-a_i + \max\{b_i/r_{uj}(t),\, [B_j(t)-t]_+\} + c_i/f_j + b_i^r/r_{uj}(t),$$

where $B_j(t)$ is the time that node $j$ becomes idle. Current prototype assumes **one assigned task per compute node per scheduling slot**, a single continuously connected V2U route, predicted channel rates without interference, and nonpreemptive service. The communication first-exit constraint is $T_{ij}-(t-a_i)\le W_{uj}(t)$; nominal deadline feasibility is $T_{ij}\le d_i$. This is an **approximate one-contact execution model**, not a full dynamic flow-scheduling proof. Link loss and execution variability can still violate an otherwise nominally feasible deadline.

Each vehicle has an unobservable correctness probability $\theta_j\in[0,1]$. Its self-reported reputation $r_j$ may be falsely elevated. Before challenge feedback, the scheduler stores a belief $\theta_j\mid\mathcal H_t\sim\mathrm{Beta}(\alpha_{jt},\beta_{jt})$ initialized from a fallible reported score. A verifiable challenge to node $j$ yields (i) an authenticated binary result $z_{jt}$ when both query and response succeed, or (ii) no useful observation after communication failure. If observed, $(\alpha,\beta)$ increments the appropriate success/failure count. The controller never observes $\theta_j$ except the clearly identified diagnostic `oracle` baseline. **This Bayesian update assumes conditionally independent, representative challenges; it does not cover an adaptive adversary that treats challenges differently.**

A challenge requires $a_j^{\mathrm{ver}}$ seconds of shared TDMA airtime, $c_j^{\mathrm{ver}}$ Gcycle of VFCN processing, and metered validation energy. Hence the transmission constraint per slot is

$$\sum_i a_{ij_i}^{\mathrm{task}} + \sum_j y_{jt} a_j^{\mathrm{ver}}\le A_t,$$

where $y_{jt}\in\{0,1\}$ is the verification action; and a slow-time-scale allowance restricts $\sum_{t\in\mathcal E_k,j}y_{jt}\le B_k$. The current implementation allows **at most one challenge per fast slot**. VFCN compute time spent on challenge execution changes $B_j(t)$; it is not merely a cost in the reward.

## Distributional risk control (fast-stage baseline)

For a task selection $x$ with conditional correctness-and-link probability $p_i$, define independent Bernoulli loss $X_i=w_i\mathbf 1\{\text{task }i\text{ does not complete correctly/on time}\}$. Dropped tasks have $p_i=0$ and therefore lose $w_i$ with probability one. The aggregate loss $L=\sum_i X_i$ has a discrete PMF propagated by convolution, and

$$\operatorname{CVaR}_\beta(L)=\min_{\eta\in\mathbb R}\left(\eta +\frac{\mathbb E[(L-\eta)_+]}{1-\beta}\right).$$

The solver selects a feasible batch assignment maximizing estimated task utility minus latency/energy while ensuring $\operatorname{CVaR}_\beta(L)\le C_{\rm risk}$, if such a selection exists. Frontier pruning uses **first-order stochastic dominance only among identical consumption states** (TDMA units and compute-node selection mask). When no candidate satisfies the risk budget, the algorithm returns the minimum-CVaR candidate and sets `risk_feasible=0`; the result is never described as meeting the risk guarantee. The risk model is only calibrated under the assumed independent Bernoulli failure model; **CVaR on aggregate lost value is not equivalent to a per-task chance constraint**.

## Actually implemented two-time-scale controller

Let fast slots be $t=0,1,\dots$ and slow epochs start only at $t=kL$. At each slow epoch, for each VFCN the controller calculates:

$$s_j=\operatorname{Var}(\theta_j\mid\mathcal H_t) \bigl(1+2|\mathbb E[\theta_j]-r_j|\bigr)\bigl(0.25+0.75\,o_j/L\bigr),$$

where $o_j$ is the number of predicted slots in the next $L$ slots with at least a 1s contact window. It then apportions at most $B_k$ validation quotas among visible vehicles, with a diminishing allocation preference based on quota already assigned. These quotas are updated **only every $L$ fast slots**; they are *not* optimal solutions to a slow-horizon Bellman equation. This is a transparent, testable budget heuristic.

Fast stage, for each task batch $\mathcal T_t$:

```
if t mod L == 0:
    QUOTA = allocate_epoch_verifications(posterior, reports, predicted_contact, budget)
B = risk_aware_distributional_DP(tasks, current_belief, full_radio_budget)
for each currently contact-feasible j with QUOTA[j] > 0:
    for signal in {challenge_pass, challenge_fail, no_response}:
        update belief hypothetically (if response)
        solve risk_aware_distributional_DP(tasks, updated_belief,
                                           radio_budget - challenge_airtime,
                                           CPU_availability - challenge_CPU)
    gain[j] = expected_postchallenge_utility - B.utility
              - challenge_cost + estimated_future_information_value
if max(gain) > 0:
    j = argmax(gain)
    perform challenge, charge radio/CPU/energy, consume QUOTA[j]
    update posterior if a usable response arrives
    dispatch using a freshly solved risk_aware_distributional_DP
else:
    dispatch B
record actual task success, deadline violation, verification resource use,
       infeasible-risk events and the per-task decision trace
```

The continuation-value term uses an explicit variance-reduction proxy. It is **not** the value function of an optimal partially observable Markov decision process (POMDP); no global joint optimality or theoretical regret guarantee is claimed. This precise limitation is important for an eventual TCCN manuscript.

## Research validation contract

Claims are hypotheses until validated against matched-budget baselines. Metrics include correct-and-on-time task fraction, weighted task reward, deadline violation rate, CVaR on *realized* loss, nominal risk infeasibility, challenge CPU/TDMA cost, energy, decision wall-clock time, prior-misrepresentation severity, contact-window sensitivity, and dependence on slow-period $L$. All quantities must be obtained from actual executed simulations; plots may not be generated from manually specified favorable values.

A development run (20 paired stochastic seeds × 40 slots) was executed only in the standalone reference engine. In that exploratory run, VA-RTO is **not** better than a random-verification baseline; verification budgets are not yet usage matched. This run is for diagnosis, not publication. Before paper claims: compare against same mean number of challenges or same resource budget; trace-replay actual upstream AirFogSim output; integrate full live scheduling/radio models; independent unseen scenario seeds; report confidence intervals and meaningful risk calibration; and compare with exact small-scale search/oracle and the actual INFOCOM benchmark configurations.