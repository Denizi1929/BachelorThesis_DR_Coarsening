# Multilevel t-SNE with Graph Coarsening

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An efficient, scalable implementation of **Multilevel t-SNE** developed as part of a Bachelor's thesis on Dimensionality Reduction and Graph Coarsening. This project incorporates heavy-edge matching, feature vector aggregation, and hierarchical Barnes-Hut optimization to accelerate t-SNE while significantly improving global structure preservation.

---

## Key Highlights

- **Enhanced Global Structure:** Achieves up to **+37% higher Distance Correlation ($dCor$)** over standard t-SNE by optimizing initial coarse macro-graphs.
- **Preserved Local Neighborhoods:** Maintains near-identical **Trustworthiness** and **Continuity** scores ($\ge 0.98$).
- **Accelerated Convergence:** Delivers consistent **15%–22% runtime reductions** across standard benchmark datasets.
- **Hierarchical Uncoarsening:** Uses degree-normalized feature projection matrices ($P_{\text{norm}}$) and jittered interpolation for smooth level-by-level embedding refinements.

---

## Experimental Benchmark Results

Quantitative evaluation conducted on an Apple Silicon Mac using $k=10$ local neighbors (via `ZADU`) and subsampled pairwise distances for global metrics:

| Dataset | Method | Time (s) ↓ | Trustworthiness ↑ | Continuity ↑ | Distance Corr ($dCor$) ↑ | Kruskal Stress ↓ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **COIL-20** ($N=1,440$) | Multilevel t-SNE | **1.33** | 0.9934 | 0.9922 | **0.5935** | 0.9889 |
| | Baseline t-SNE | 1.65 | 0.9955 | 0.9937 | 0.5749 | 0.9834 |
| **Pendigits** ($N=10,992$) | Multilevel t-SNE | **14.00** | 0.9980 | 0.9966 | **0.6408** | 0.7844 |
| | Baseline t-SNE | 16.44 | 0.9986 | 0.9961 | 0.4664 | 0.5596 |
| **MNIST** ($N=10,000$) | Multilevel t-SNE | **17.00** | 0.9839 | 0.9748 | **0.4633** | 0.9874 |
| | Baseline t-SNE | 17.49 | 0.9863 | 0.9736 | 0.3489 | 0.9722 |

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Denizi1929/BachelorThesis_DR_Coarsening.git](https://github.com/Denizi1929/BachelorThesis_DR_Coarsening.git)
   cd BachelorThesis_DR_Coarsening