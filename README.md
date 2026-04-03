# Champion AHP: Incomplete Pairwise Comparison Methods

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17404548.svg)](https://doi.org/10.5281/zenodo.17404548)

Python implementation of incomplete pairwise comparison matrix (IPCM) construction methods for the Analytic Hierarchy Process (AHP), including the novel **Champion AHP (C-AHP)** and **Champion-Closure AHP (CC-AHP)** methods.

## Overview

This repository accompanies the paper:

> **Sokantika, S. & Ratnapinda, P.** (2025). *The Role of Ranking Information in AHP with Limited Pairwise Comparisons.*

The code implements five IPCM construction strategies:

| Method | Comparisons | Prior Knowledge | Description |
|--------|-------------|-----------------|-------------|
| **Star** | n-1 | None | Random hub selection |
| **Cycle** | n | None | Circular permutation structure |
| **AHP Express** | n-1 | Ranking | Hub at best alternative |
| **Champion AHP (C-AHP)** | n-1 | None (local) | Winner-stays-on tournament |
| **Champion-Closure AHP (CC-AHP)** | ≤n | None (local) | Tournament with closure edge |

## Installation

```bash
git clone https://github.com/[username]/champion-ahp.git
cd champion-ahp
pip install -r requirements.txt
```

## Quick Start

```python
import numpy as np
from src import (
    generate_consistent_pcm, add_noise,
    create_champion_closure_ahp_ipcm,
    llsm_weights_incomplete
)

# Generate a noisy PCM with 5 alternatives
pcm, true_weights = generate_consistent_pcm(n=5)
noisy_pcm = add_noise(pcm, sigma=1.0)

# Create incomplete PCM using CC-AHP
ipcm = create_champion_closure_ahp_ipcm(noisy_pcm)

# Estimate weights via LLSM
estimated_weights = llsm_weights_incomplete(ipcm)
print("Estimated weights:", estimated_weights)
```

## Repository Structure

```
champion-ahp/
├── src/
│   ├── __init__.py          # Package exports
│   ├── ipcm_methods.py       # IPCM construction algorithms
│   ├── pcm_utils.py          # PCM generation, noise, LLSM
│   ├── analysis.py           # Statistical tests
│   └── simulation.py         # Monte Carlo simulation runner
├── notebooks/
│   └── simulation_analysis.ipynb
├── data/                     # Simulation results (see Zenodo)
├── requirements.txt
└── README.md
```

## Data Availability

Full simulation data (200,000 matrices across 20 conditions) is archived on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17404548.svg)](https://doi.org/10.5281/zenodo.17404548)

The Zenodo repository includes:
- All simulated pairwise comparison matrices (10,000 per condition)
- True weight vectors for each simulation
- Complete and incomplete PCMs for all methods
- Estimated weight vectors from LLSM
- Raw performance metrics (Euclidean distance, Kendall's tau)

**For exact replication of paper results, use the pre-generated Zenodo data.**

## Running New Simulations

To generate new simulation data (results may vary due to random seed):

```python
from src.simulation import run_full_experiment

# Run experiment (adjust n_samples for quick testing)
results = run_full_experiment(
    n_values=[4, 5, 6, 7],
    sigma_values=[0.5, 1.0, 1.5, 2.0, 2.5],
    n_samples=1000,  # Use 10000 for full replication
    seed=42
)

results.to_pickle('results.pkl')
```

## Dependencies

- Python ≥ 3.8
- NumPy
- Pandas
- SciPy
- NetworkX
- scikit-posthocs
- tqdm
- matplotlib
- seaborn

## Citation

```bibtex
@article{sokantika2025champion,
  title={The Role of Ranking Information in AHP with Limited Pairwise Comparisons},
  author={Sokantika, Saronsad and Ratnapinda, Parot},
  journal={},
  year={2025}
}
```

## License

MIT License
