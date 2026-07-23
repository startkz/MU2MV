# MU2MV

This repository contains the experimental code and infrastructure-data utilities
used by the final TMC submission version of:

**MU2MV: Robust Trust-Aware Multi-Timescale Task Offloading in UAV-Assisted Vehicular Fog Networks**

## Final manuscript experiments

The current paper experiments are maintained in `experiments/`.

```bash
python experiments/run_final_experiments.py
```

The script regenerates the final manuscript figures and CSV source data:

- payoff, payment, and convergence curves for MU2MV, AoP-aware, DGTT, and Greedy;
- reputation-aware task allocation, resource, delay, SNR, energy, and priority comparisons;
- the robust-FL experiment under 20% malicious participants;
- the ablation table and experimental-configuration table.

Outputs are written to `results/figures`, `results/csv`, and `results/qa`.
To copy the generated EPS figures into the parent LaTeX paper folder, run:

```bash
python experiments/run_final_experiments.py --sync-paper-figures
```

The final baselines are AoP-aware, DGTT, and Greedy. Older MADDPG and Q-learning
comparison labels are intentionally removed from the final experiment package.

## Legacy infrastructure data utilities

The repository also preserves the original code to:
- extract critical infrastructure assets from OSM data
- estimate the amount of infrastructure for given grid cells
- calculate the Critical Infrastructure Spatial Index (CISI)

It also provides a Jupyter Notebook and additional code to reproduce the figures and supplementary material in Nirandjan et al. (2022). These utilities support the real infrastructure data used to drive the UAV inspection scenario, but they are separate from the final MU2MV offloading and robust-FL experiments.

## Data requirements
- All critical infrastructure data is based on OpenStreetMap (OSM), which can be freely downloaded. The planet file used in Nirandjan et al. (2022) is downloaded at January 8, 2021. However, the latest release of planet.osm.pbf file can be used to run the code.

The following datasets are used for the technical validation:
- Gridded global population distribution data is available from the WorldPop data portal (https://www.worldpop.org). In Nirandjan et al. (2022), global population distribution data of 2020 is used. 
- Gridded global GDP data is available from https://datadryad.org/stash/dataset/doi:10.5061/dryad.dk1j0. In Nirandjan et al. (2022), we use the GDP_PPP_30arcsec file for 2015.

## Python requirements

Recommended option is to use a [miniconda](https://conda.io/miniconda.html)
environment to work in for this project, relying on conda to handle some of the
trickier library dependencies.

```bash

# Add conda-forge channel for extra packages
conda config --add channels conda-forge

# Create a conda environment for the project and install packages
conda env create -f environment.yml
conda activate CoInf

```

**Requirements:** [NumPy](http://www.numpy.org/), [pandas](https://pandas.pydata.org/), [geopandas](http://geopandas.org/), [matplotlib](https://matplotlib.org/), [pygeos](https://pypi.org/project/pygeos/)

For the final MU2MV manuscript experiments only, the minimal requirements are:

```bash
pip install -r requirements-mu2mv.txt
```

## Global CISI visualizer
The CISI and the sub-score per system is visualized at a global scale via https://cisi-index.appspot.com/. This is only done for a resolution of 0.25 degrees for speed performance considerations. Have a look and discover where critical infrastructure is located!  

### License
Copyright (C) 2021 Sadhana Nirandjan & Elco Koks. All versions released under the [MIT license](LICENSE).
