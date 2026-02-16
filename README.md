# ts-tools

**Robust Transition State Search from Noisy Starting Geometries**

A comprehensive toolkit for locating index-1 saddle points (transition states) on molecular potential energy surfaces, with emphasis on robustness to heavily displaced starting geometries. Implements and benchmarks GAD, eigenvector following, eigenvalue descent, Sella, iHiSD, recursive HiSD, and novel escape strategies that achieve **100% convergence from 2.0 Å noise**.

---

## Overview

Transition states (TSs) are first-order saddle points on the Born–Oppenheimer potential energy surface that govern reaction kinetics. Finding them reliably from noisy or coarse initial guesses—as commonly produced by generative models, interpolation schemes, or molecular dynamics—remains a central challenge in computational chemistry.

**ts-tools** addresses this through two contributions:

1. **Robust saddle point search:** Systematic study of algorithms and escape mechanisms achieving near-perfect TS convergence from geometries perturbed by up to 2.0 Å of random noise—a regime where conventional methods fail catastrophically.

2. **Adjoint sampling for reaction generation:** Integration of diffusion-based generative models that sample molecular conformations from Boltzmann distributions defined by energy functions.

### Headline Results

| Method | Noise Level | Convergence |
|--------|-------------|-------------|
| v₂ kicking + mode tracking + adaptive dt | **2.0 Å** | **100%** |
| Adaptive dt control (path-based) | 2.0 Å | 92% |
| State-based dt control | 2.0 Å | 80% |
| Multi-Mode GAD (SCINE, HPO best) | 1.0 Å | 100% |
| Multi-Mode GAD (HIP, HPO best) | 1.0 Å | 93.3% |
| Sella (SCINE, HPO best) | 1.0 Å | 66.7% |
| Sella (HIP, HPO best) | 1.0 Å | 53.3% |

---

## Algorithms

### Core Saddle Point Search

**Gentlest Ascent Dynamics (GAD)**
Inverts the force component along the lowest Hessian eigenvector, creating a vector field whose stable fixed points are index-1 saddle points:

$$\mathbf{F}_\text{GAD} = -\nabla E + 2(\nabla E \cdot \mathbf{v}_1)\mathbf{v}_1$$

Implemented with Euler and RK45 integration, continuous mode tracking, and Eckart-projected mass-weighted Hessians.

**Eigenvector Following (Newton GAD)**
Newton step on the GAD surface using the absolute-value Hessian: $\Delta\mathbf{x} = H_\text{abs}^{-1} \cdot F_\text{GAD}$. Provides quadratic convergence near saddle points.

**Eigenvalue Product Descent**
Directly minimizes $\mathcal{L} = \lambda_0 \cdot \lambda_1$ via autograd through the eigendecomposition. Requires a differentiable calculator (HIP). Achieves 92.7% from clean starts in 2.3 steps.

**Direct Eigenvalue Descent (Sign Enforcer)**
Adaptive loss based on current Morse index: pushes eigenvalues toward exactly one negative. Achieves 99.7% from clean starts in 1.8 steps.

### Escape Strategies for High-Index Saddle Points

**v₂ Kicking** — The primary innovation. When GAD stalls at a high-index saddle (detected via displacement plateau), perturbs geometry along the second vibrational eigenvector. Tries both ±v₂ directions and selects lower energy. Combined with mode tracking and adaptive dt, achieves **100% convergence on 2.0 Å noise**.

**Adaptive Timestep Control** — Two variants:
- *Path-based:* Adapts dt using displacement history with grow/shrink/boost/reset logic and saddle-order tracking.
- *State-based:* Adapts dt using only local quantities (eigenvalues, gradient norm, spectral gap).

### Baseline Algorithms

| Algorithm | Description |
|-----------|-------------|
| **Sella** | Trust-region saddle optimizer (ASE ecosystem) |
| **iHiSD** | Improved HiSD with crossover parameter θ ∈ [0,1] |
| **Recursive HiSD** | Systematic index descent: n → n-1 → ... → 1 |
| **k-HiSD** | Generalized GAD with reflection operator |
| **Gradient Descent** | Sanity-check baseline (finds minima) |
| **Newton-Raphson** | Minimization with pseudoinverse Hessian |

---

## Eckart Projection

All algorithms operate on projected Hessians with translation/rotation modes removed. Seven projection variants were tested—all produce statistically identical results:

| Variant | Description |
|---------|-------------|
| `eckart_full` | Standard P·H·P in 3N space (default) |
| `reduced_basis` | QR complement → full-rank (3N-6)×(3N-6) Hessian |
| `+ purify` | Sum-rule purification for translational invariance |
| `+ frame_tracking` | Kabsch alignment to reference frame |
| `+ project_grad_v` | Project gradient and guide vector into vibrational subspace |

---

## Computational Backends

### HIP — Machine Learning Interatomic Potential
[github.com/burgerandreas/hip](https://github.com/burgerandreas/hip)

Equiformer-based ML potential providing GPU-accelerated, differentiable energy, force, and Hessian predictions. Required for eigenvalue descent methods (autograd through eigendecomposition).

### SCINE/Sparrow — Semi-Empirical Calculator
[scine.ethz.ch/download](https://scine.ethz.ch/download/)

CPU-only semi-empirical quantum chemistry (DFTB0, PM6, AM1) with analytical Hessians. Consistently outperforms HIP for saddle point characterization: 94.1% vs 74.9% global TS rate, ~33× faster wall time.

Both backends are wrapped into a unified `predict_fn(coords, atomic_nums, do_hessian, require_grad)` interface.

---

## Repository Structure

```
ts-tools/
├── src/
│   ├── core_algos/                 # Algorithm implementations
│   │   ├── gad.py                  # GAD: Euler, RK45, mode tracking
│   │   ├── eigenproduct.py         # Eigenvalue product descent
│   │   └── signenforcer.py         # Direct eigenvalue descent
│   ├── dependencies/               # Shared infrastructure
│   │   ├── differentiable_projection.py  # Eckart projection (7 variants)
│   │   ├── hessian.py              # Hessian analysis, vibrational frequencies
│   │   ├── calculators.py          # HIP and SCINE adapters
│   │   └── common_utils.py         # Geometry validation, convergence checks
│   ├── runners/                    # Integration drivers
│   │   ├── gad_euler_core.py       # GAD-Euler with convergence logic
│   │   ├── gad_rk45_core.py        # GAD-RK45 adaptive integration
│   │   └── eigenvalue_descent_core.py
│   ├── noisy/                      # Noisy-start experiments
│   │   ├── multi_mode_eckartmw.py  # Production algorithm
│   │   ├── v2_tests/               # Latest experiment generation
│   │   │   ├── kick_experiments/   # 8 kick strategies
│   │   │   ├── baselines/          # iHiSD, recursive HiSD, k-HiSD
│   │   │   ├── runners/            # Parallel experiment runners
│   │   │   ├── scripts/            # SLURM templates (17 configs)
│   │   │   ├── logging/            # Extended metrics and diagnostics
│   │   │   └── analysis/           # Singularity and stall analysis
│   │   └── scine_*_parallel.py     # Parallel SCINE wrappers
│   └── experiments/                # HPO and comparison studies
│       ├── Sella/                  # Sella integration and HPO
│       └── 2025/                   # Multi-mode Eckart-MW HPO
├── documentation/
│   └── for_robots/                 # Machine-readable codebase guide
├── supporting/                     # LaTeX reports and presentations
└── hpo_results_full/               # HPO analysis and figures
```

---

## Key Findings

1. **v₂ kicking is the most effective escape mechanism** — but it works as brute-force "unsticking" rather than principled index reduction. Mean Morse index changes only from 5.39 → 5.00 after kicks.

2. **Adaptive k = Morse index is fundamentally wrong** for escaping high-index saddles. k-HiSD theory proves this *stabilizes* the current saddle (0% escape rate).

3. **Only 26% of stalls occur near eigenvalue singularities.** The majority are at genuine high-index saddles where GAD direction becomes weak.

4. **Fixed perturbation magnitude outperforms adaptive scaling** in 99.1% of successful trials.

5. **SCINE outperforms HIP across all metrics** — better convergence (94.1% vs 74.9%), 33× faster, despite being CPU-only. Analytical Hessians are more reliable than ML-predicted ones.

6. **All 7 projection variants produce identical results** — Eckart projection implementation is not a bottleneck.

7. **Multi-Mode GAD dramatically outperforms Sella** on noisy starts: 94.1% vs 47.0% global TS rate.

---

## Installation

```bash
git clone https://github.com/<org>/ts-tools.git
cd ts-tools
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Calculator Setup

**SCINE/Sparrow** (recommended):
```bash
pip install scine-sparrow scine-utilities
```

**HIP** (for differentiable eigenvalue descent):
```bash
# Follow instructions at https://github.com/burgerandreas/hip
```

---

## Usage

### Running GAD with v₂ Kicking (Production)

```python
from src.noisy.multi_mode_eckartmw import run_multi_mode_eckartmw
from src.dependencies.calculators import make_scine_predict_fn

predict_fn = make_scine_predict_fn(functional="DFTB0")

result = run_multi_mode_eckartmw(
    predict_fn=predict_fn,
    coords=initial_coords,
    atomic_nums=atomic_nums,
    scine_elements=elements,
    n_steps=10000,
    dt=0.003,
    dt_max=0.07,
    escape_delta=0.27,
    escape_disp_threshold=5.66e-4,
    max_atom_disp=0.35,
    min_interatomic_dist=0.5,
)
```

### SLURM Submission (HPC)

```bash
# Default: v₂ kicking with Eckart projection
sbatch src/noisy/v2_tests/scripts/slurm_templates/gad_plain_run.slurm

# With environment overrides
N_STEPS=15000 MAX_SAMPLES=50 START_FROM=midpoint_rt_noise2.0A \
    sbatch src/noisy/v2_tests/scripts/slurm_templates/kick_v2_run.slurm

# Projection experiments
PROJECTION_MODE=reduced_basis PURIFY_HESSIAN=true \
    sbatch src/noisy/v2_tests/scripts/slurm_templates/gad_plain_run.slurm
```

### Running HPO

```bash
python src/experiments/2025/multi_mode_eckartmw.py \
    --h5-path data/transition1x.h5 \
    --n-trials 500 \
    --max-samples 15 \
    --scine-functional DFTB0
```

---

## Data

Experiments use the **Transition1x** dataset (HDF5 format) containing reactant and transition state geometries for small organic molecules. Starting geometries are constructed as:

$$\mathbf{x}_0 = \frac{\mathbf{x}_\text{reactant} + \mathbf{x}_\text{TS}}{2} + \sigma \cdot \boldsymbol{\xi}, \quad \boldsymbol{\xi} \sim \mathcal{N}(0, I)$$

where σ controls noise level (typically 1.0 or 2.0 Å).

---

## Theory

The core algorithm solves GAD dynamics on Eckart-projected, mass-weighted Hessians with mode tracking:

1. **Hessian projection:** Remove 6 translation/rotation modes via Eckart B-matrix projection
2. **Mass-weighting:** $H_\text{mw} = M^{-1/2} H M^{-1/2}$
3. **Eigendecomposition:** Extract vibrational modes, track lowest via maximum overlap
4. **GAD step:** $\mathbf{x}_{n+1} = \mathbf{x}_n + \Delta t \cdot \mathbf{F}_\text{GAD}$
5. **Plateau detection:** Monitor displacement window; trigger escape when stalled
6. **v₂ escape:** Perturb along second vibrational eigenvector to exit high-index saddle
7. **Adaptive dt:** Grow/shrink timestep based on displacement history and saddle order

Convergence criterion: $\lambda_0 \cdot \lambda_1 < 0$ (exactly one negative vibrational eigenvalue).

Full mathematical details, derivations, and proofs are in [`supporting/ts_tools_report.tex`](supporting/ts_tools_report.tex).

---

## References

- E, W. and Zhou, X. "The Gentlest Ascent Dynamics." *Nonlinearity* 24(6):1831, 2011.
- Yin, J., Zhang, L., and Zhang, P. "High-Index Optimization-Based Shrinking Dimer Method." *SIAM J. Sci. Comput.* 41(6), 2019.
- Levitt, A. and Ortner, C. "Convergence and Cycling in Walker-type Saddle Search Algorithms." *SIAM J. Numer. Anal.* 55(5), 2017.
- HIP: [github.com/burgerandreas/hip](https://github.com/burgerandreas/hip)
- SCINE: [scine.ethz.ch/download](https://scine.ethz.ch/download/)

---

## License

Research code — Aspuru-Guzik Group, University of Toronto.
