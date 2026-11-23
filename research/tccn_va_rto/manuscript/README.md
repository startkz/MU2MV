# Research-stage IEEE TCCN manuscript

`main.tex` is a compilable IEEEtran **methodology draft**, not a submission-ready article. It documents the actual VA-RTO reference model, period-separated controller, conditional FSD lemma, explicit research assumptions, and evaluation protocol. It deliberately makes no empirical superiority claim.

Compile with TeX Live:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The compiled four-page working PDF has been locally rendered and inspected; original training/traffic/verification experiments still require native AirFogSim/SUMO validation and held-out scenarios. The LaTeX bibliography is preliminary and must be expanded with fully verified metadata and competitive baselines before journal submission.
