<div align="center">

# Transition State Sampling

**Robust saddle-point search from noisy starting geometries**

**Paper (PDF): [`contributions.pdf`](contributions.pdf)** — important theory document and precursor to our two in-progress submissions.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: Research](https://img.shields.io/badge/license-research-green.svg)](#license)

*100% transition state convergence from molecules displaced by 2.0 Å of random noise*

[Overview](#overview) &#8226; [Results](#results) &#8226; [Algorithms](#algorithms) &#8226; [Architecture](#architecture) &#8226; [Usage](#usage) &#8226; [Theory](#theory)

</div>

---

## Overview

Finding transition states — the index-1 saddle points governing chemical reaction rates — is one of the hardest problems in computational chemistry. Standard methods fail when the starting geometry is far from the true saddle point, a common scenario in generative molecular design, coarse interpolation, and noisy molecular dynamics.

This toolkit solves that problem. We implement, benchmark, and extend a suite of saddle-point search algorithms with a focus on **robustness to heavily displaced starting geometries**, achieving convergence rates that were previously considered impossible.

**Two contributions:**

1. **Robust TS search from noise:** Novel escape mechanisms (v₂ kicking, adaptive timestep, mode tracking) that achieve **100% convergence from 2.0 Å noise** — a regime where Sella, iHiSD, and plain GAD all fail.

2. **Adjoint sampling for reaction generation:** Diffusion-based generative models sampling molecular conformations from Boltzmann distributions for data-driven reaction pathway discovery.

**Backends:** [HIP](https://github.com/burgerandreas/hip) (Equiformer ML potential, GPU) and [SCINE/Sparrow](https://scine.ethz.ch/download/) (semi-empirical DFTB0/PM6/AM1, CPU).

---

## Results

### Convergence from Noisy Starting Geometries

| Method | Noise | Convergence | Notes |
|:-------|:-----:|:-----------:|:------|
| **v₂ kicking + mode tracking + adaptive dt** | **2.0 Å** | **100%** | Our best method |
| Adaptive dt control (path-based) | 2.0 Å | 92% | No kicking needed |
| State-based dt control | 2.0 Å | 80% | No path history needed |
| Multi-Mode GAD + HPO (SCINE) | 1.0 Å | 100% | 500 Optuna trials |
| Multi-Mode GAD + HPO (HIP) | 1.0 Å | 93.3% | ML potential |
| Sella trust-region + HPO (SCINE) | 1.0 Å | 66.7% | Best of 176 trials |
| Sella trust-region + HPO (HIP) | 1.0 Å | 53.3% | Best of 181 trials |
| Plain GAD-Euler | 1.0 Å | 13% | No escape mechanism |

### From Clean Starting Geometries (Transition1x dataset)

| Method | Convergence | Avg Steps | Avg Time |
|:-------|:-----------:|:---------:|:--------:|
| Direct Eigenvalue Descent | 99.7% | 1.8 | 3.1 s |
| Eigenvalue Product Descent | 92.7% | 2.3 | 3.2 s |
| GAD-Euler | 91.3% | 25.1 | 50 s |
| GAD-RK45 | 91.3% | 7.3 | 64 s |

### Backend Comparison (Multi-Mode GAD, 1.0 Å noise)

| | SCINE (CPU) | HIP (GPU) |
|:--|:-----------:|:---------:|
| Global TS rate | **94.1%** | 74.9% |
| Trials with ≥80% success | **497/500** | 45/102 |
| Mean wall time / sample | **2.9 s** | 96.9 s |

SCINE's analytical Hessians outperform HIP's ML-predicted Hessians for eigenvalue-based saddle characterization, while being 33x faster.

---

## Algorithms

### Core: Gentlest Ascent Dynamics (GAD)

Inverts the force component along the lowest Hessian eigenvector, creating a vector field whose stable fixed points are index-1 saddle points:

$$\mathbf{F}_{\text{GAD}} = -\nabla E + 2(\nabla E \cdot \mathbf{v}_1)\,\mathbf{v}_1$$

Implemented with **Euler** and **RK45** integration, **continuous mode tracking** (maximum-overlap eigenvector selection across steps), and **Eckart-projected mass-weighted Hessians** (7 projection variants tested, all equivalent).

### Novel: v₂ Escape Mechanism

When GAD stalls at a high-index saddle (detected via displacement plateau monitoring), we perturb along the **second vibrational eigenvector**:

1. Detect plateau: mean displacement below threshold over sliding window, stable Morse index > 1
2. Compute v₂ from Eckart-projected Hessian
3. Try both ±δ·v₂, select direction with lower energy
4. Reset adaptive timestep with boost factor
5. Resume GAD dynamics

This achieves **100% convergence on 2.0 Å noise** — the first method to do so.

### Also Implemented

| Algorithm | Type | Key Idea |
|:----------|:-----|:---------|
| **Eigenvector Following** | Newton-like | $\Delta x = H_{\text{abs}}^{-1} \cdot F_{\text{GAD}}$ — quadratic convergence |
| **Eigenvalue Product Descent** | Autograd | Minimize $\lambda_0 \cdot \lambda_1$ via backprop through eigendecomposition |
| **Direct Eigenvalue Descent** | Autograd | Adaptive loss based on Morse index (sign enforcer) |
| **Sella** | Trust-region | Industry-standard ASE saddle optimizer |
| **iHiSD** | Crossover | Interpolates gradient flow → k-HiSD via θ ∈ [0,1] |
| **Recursive HiSD** | Index descent | Systematic n → n-1 → ... → 1 saddle descent |
| **k-HiSD** | Reflection | Generalized GAD with $R_k = I - 2\sum v_i v_i^T$ |

---

## Architecture

```
src/
├── core_algos/                          # Algorithm implementations
│   ├── gad.py                           #   GAD: Euler, RK45, mode tracking
│   ├── eigenproduct.py                  #   Eigenvalue product descent (autograd)
│   └── signenforcer.py                  #   Direct eigenvalue descent
├── dependencies/                        # Shared infrastructure
│   ├── differentiable_projection.py     #   Eckart projection (7 variants)
│   ├── hessian.py                       #   Vibrational frequency analysis
│   ├── calculators.py                   #   HIP / SCINE adapter layer
│   └── common_utils.py                  #   Geometry validation, convergence
├── runners/                             # Integration drivers
│   ├── gad_euler_core.py                #   GAD-Euler with convergence logic
│   ├── gad_rk45_core.py                 #   Adaptive RK45 integration
│   └── eigenvalue_descent_core.py       #   Eigenvalue descent driver
├── noisy/                               # Noisy-start experiments
│   ├── multi_mode_eckartmw.py           #   Production: v₂ kicking + mode tracking
│   └── v2_tests/                        #   Experiment suite
│       ├── kick_experiments/            #     8 perturbation strategies
│       ├── baselines/                   #     iHiSD, recursive HiSD, k-HiSD, GD
│       ├── runners/                     #     Parallel experiment runners
│       ├── scripts/slurm_templates/     #     17 SLURM HPC configurations
│       ├── logging/                     #     Extended metrics & diagnostics
│       └── analysis/                    #     Singularity & stall analysis
└── experiments/                         # Hyperparameter optimization
    ├── Sella/                           #   Sella HPO (Optuna/TPE)
    └── 2025/                            #   Multi-Mode Eckart-MW HPO
```

All algorithms interact through a unified `predict_fn(coords, atomic_nums, do_hessian, require_grad)` interface that wraps both HIP and SCINE backends.

---

## Key Findings

- **v₂ kicking works as brute-force unsticking**, not principled index reduction. Mean Morse index barely changes (5.39 → 5.00) after kicks — but it breaks the dt → 0 deadlock that traps GAD at high-index saddles.

- **Adaptive k = Morse index is fundamentally wrong.** k-HiSD theory proves index-k saddles are *stable* fixed points of k-HiSD — using k = current index *stabilizes* the saddle you're trying to escape (0% success rate).

- **Only 26% of stalls are near eigenvalue singularities.** Most are at genuine high-index saddles where the GAD direction vanishes, not at eigenvalue crossings.

- **Fixed perturbation magnitude beats adaptive scaling** in 99.1% of successful trials — simpler is better.

- **All 7 Eckart projection variants produce identical results** — the projection implementation is never the bottleneck.

---

## Usage

### Quick Start

```python
from src.noisy.multi_mode_eckartmw import run_multi_mode_eckartmw
from src.dependencies.calculators import make_scine_predict_fn

predict_fn = make_scine_predict_fn(functional="DFTB0")

result = run_multi_mode_eckartmw(
    predict_fn=predict_fn,
    coords=initial_coords,          # (N, 3) tensor
    atomic_nums=atomic_nums,         # (N,) tensor
    scine_elements=elements,         # list of element symbols
    n_steps=10000,
    dt=0.003, dt_max=0.07,
    escape_delta=0.27,
    escape_disp_threshold=5.66e-4,
    max_atom_disp=0.35,
)
```

### HPC (SLURM)

```bash
# v₂ kicking with Eckart projection (default production config)
sbatch src/noisy/v2_tests/scripts/slurm_templates/kick_v2_run.slurm

# Override parameters via environment
N_STEPS=15000 MAX_SAMPLES=50 START_FROM=midpoint_rt_noise2.0A \
    sbatch src/noisy/v2_tests/scripts/slurm_templates/kick_v2_run.slurm
```

### Installation

```bash
git clone https://github.com/memo-ozdincer/transition-state-sampling.git
cd transition-state-sampling
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install scine-sparrow scine-utilities  # recommended backend
```

---

## Theory

The core pipeline at each step:

1. **Eckart projection** — Remove 6 translation/rotation modes: $\tilde{H} = P_{\text{vib}}\, M^{-1/2} H\, M^{-1/2}\, P_{\text{vib}}$
2. **Eigendecomposition** — Extract vibrational modes, track lowest via maximum overlap with previous step
3. **GAD step** — $\mathbf{x}_{n+1} = \mathbf{x}_n + \Delta t \cdot \mathbf{F}_{\text{GAD}}$
4. **Plateau detection** — Sliding window over displacement; trigger escape when stalled at Morse index > 1
5. **v₂ escape** — Perturb along second vibrational eigenvector
6. **Adaptive dt** — Grow/shrink/reset timestep based on displacement history and saddle order improvement

**Convergence criterion:** $\lambda_0 \cdot \lambda_1 < 0$ (exactly one negative vibrational eigenvalue).

Full derivations in [`supporting/ts_tools_report.tex`](supporting/ts_tools_report.tex).

---

## References

- E, W. and Zhou, X. "The Gentlest Ascent Dynamics." *Nonlinearity* 24(6):1831, 2011.
- Yin, J. et al. "High-Index Optimization-Based Shrinking Dimer Method." *SIAM J. Sci. Comput.* 41(6), 2019.
- Levitt, A. and Ortner, C. "Convergence and Cycling in Walker-type Saddle Search." *SIAM J. Numer. Anal.* 55(5), 2017.
- [HIP](https://github.com/burgerandreas/hip) — Equiformer ML interatomic potential
- [SCINE](https://scine.ethz.ch/download/) — Semi-empirical quantum chemistry

---

<div align="center">

*Matter Lab — University of Toronto*

</div>
