# MU2MV Final Experiment Reproduction

This folder contains the final experiment code for the TMC submission version of
MU2MV. It complements the original infrastructure-data processing scripts in the
repository and regenerates the manuscript-level results.

## What It Reproduces

- Fig. 3-7: UAV/VFCN payoff, payment, and learning-rate convergence curves.
- Fig. 8-12: reputation allocation, resource, delay, SNR, and energy comparisons.
- Fig. 13: robust FL accuracy and effective malicious aggregation weight under
  20% malicious participants.
- Task-priority comparison and the ablation table used in the experiment section.

The final baselines are:

- MU2MV
- AoP-aware
- DGTT
- Greedy

MADDPG and Q-learning are intentionally not used in the final version.

## Run

```bash
python experiments/run_final_experiments.py
```

Outputs are written to:

- `results/figures/*.eps`, `*.pdf`, `*.png`
- `results/csv/*.csv`
- `results/qa/figure_qa_notes.md`

To also copy the generated EPS files into the parent LaTeX paper folder:

```bash
python experiments/run_final_experiments.py --sync-paper-figures
```

## Experimental Configuration

The script writes `results/csv/experimental_configuration.csv` with the current
manuscript configuration: 5 UAVs, 100 VFCNs, 500 m by 3000 m road area, VFCN
speed 18-54 km/h, task sizes 1-10 Mbits, 5000 training iterations, 20 PSO
particles, 50 PSO iterations per slot, and an inspection-image FL task using a
compact CNN with 1.25e6 trainable parameters.

## Robustness Scope

The robust-FL experiment covers label flipping with update perturbation,
adaptive update scaling, and collusive biased updates under 20% malicious
participants. The code follows the paper's bounded claim: trust weighting reduces
detectable malicious influence when the server-side validation signal remains
clean, but it is not a guarantee against validation-set poisoning or fully
stealthy attackers.

