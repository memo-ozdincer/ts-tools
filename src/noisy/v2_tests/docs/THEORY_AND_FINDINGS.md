# Theory and Findings: GAD, k-HiSD, and Transition State Finding

## Executive Summary

**Key Finding**: The "adaptive k-HiSD where k = Morse index" implementation was fundamentally wrong. k-HiSD with k reflections makes index-k saddles **stable**, not unstable. Using k = Morse index stabilizes the current saddle instead of escaping it.

**Why v₂ kicking works**: It's a brute-force "unsticking" mechanism that prevents dt → 0, allowing continued exploration until the TS is eventually found. It does NOT work by systematically reducing the Morse index.

---

## 1. Background: The Transition State Finding Problem

### 1.1 What We're Trying to Do
Find index-1 saddle points (transition states) on a potential energy surface (PES). Starting from a noisy midpoint geometry, we want to converge to the TS.

### 1.2 The GAD Algorithm
**Gentlest Ascent Dynamics (GAD)** follows the direction:

```
ẋ = -∇E + 2(v₁ᵀ∇E)v₁
```

where v₁ is the eigenvector of the lowest eigenvalue of the Hessian.

This is equivalent to:
- **Ascending** along v₁ (the softest mode)
- **Descending** along all other modes

**Goal**: Converge to index-1 saddle points where λ₁ < 0 < λ₂.

### 1.3 The Problem: GAD Gets Stuck
In practice, GAD often:
1. Gets stuck at high-index saddles (index > 1)
2. Has dt → 0 (timestep collapses)
3. GAD norm → 0 (direction vanishes)
4. Never reaches index-1 TS

---

## 2. Theoretical Framework

### 2.1 Levitt-Ortner Paper: Singularities and Cycling

**Key Results**:

1. **Singularity Set**: S = {x : λ₁(x) = λ₂(x)}
   - GAD is undefined on S (v₁ not uniquely defined)
   - Near S, GAD can exhibit quasi-periodic orbits

2. **Attractive Singularities** (Theorem 3.4):
   - Some points in S are attractive for GAD
   - GAD trajectories can get trapped in O(ε) annuli around these points
   - Results in oscillation without convergence

3. **Convergence Conditions** (Theorems 2.2-2.3):
   - GAD converges if started in an "index-1 region" containing the TS
   - Cannot be globally convergent due to singularities

### 2.2 iHiSD Paper: High-index Saddle Dynamics

**k-HiSD Dynamics**:
```
ẋ = -R∇E    where R = I - 2∑ᵢ₌₁ᵏ vᵢvᵢᵀ
```

**Critical Theorem (Theorem 3.2)**:
> **Index-k saddles are stable fixed points of k-HiSD.**

This means:
- **1-HiSD (k=1)**: Stable at index-1 saddles → This is GAD!
- **2-HiSD (k=2)**: Stable at index-2 saddles
- **k-HiSD**: Stable at index-k saddles

**Implication**: To FIND index-k saddles, use k-HiSD. To ESCAPE index-k saddles, you need DIFFERENT dynamics.

### 2.3 iHiSD Crossover Dynamics

The iHiSD paper proposes crossover dynamics:
```
ẋ = β(-αR + (1-α)I)∇E
α̇ = ε(α)
```

- **α = 0**: Gradient descent (finds minima, escapes saddles)
- **α = 1**: k-HiSD (finds index-k saddles)

**The Crossover**: Start with α ≈ 0 to escape high-index regions, then increase α toward 1 to converge to the target saddle.

---

## 3. Our Diagnostic Findings

### 3.1 GAD + v₂ Kicking Diagnostics

From running 20 samples with comprehensive logging:

```
Total stall events: 23
Dominant Morse index at stall: 5
Mean eigenvalue gap at stall: 9.87e-01
Fraction of stalls near singularity (gap < 0.01): 26.1%
```

**Key Observations**:
1. **Stalls occur at high-index saddles (index-5)**, not near singularities
2. **v₂ kicking does NOT reduce Morse index**: Mean index 5.39 → 5.00 per kick
3. **v₂ kicking DOES "unstick" the dynamics**: Prevents dt → 0
4. **85% convergence rate** despite not reducing index directly

### 3.2 Adaptive k-HiSD Diagnostics (My Wrong Implementation)

From running 20 samples with k = Morse index:

```
Total stall events: 153
Dominant Morse index at stall: 7
Fraction of stalls near singularity (gap < 0.01): 0.0%
Convergence rate: 0%
```

**Why It Failed**:
1. Using k = Morse index makes the **current saddle stable**
2. At index-5, using 5-HiSD stabilizes index-5 saddles
3. The system converges TO high-index saddles instead of escaping them
4. Index actually INCREASED to 7 (found even higher-index saddles)

---

## 4. Why v₂ Kicking Actually Works

### 4.1 The Mechanism

v₂ kicking is NOT doing what we thought:

| What We Thought | What Actually Happens |
|-----------------|----------------------|
| Reduce Morse index via v₂ perturbation | Brute-force "unsticking" |
| Systematic descent: index-5 → 4 → 3 → 2 → 1 | Random exploration until TS found |
| Equivalent to 2-HiSD step | Just prevents dt collapse |

### 4.2 Why It Prevents dt Collapse

When GAD is stuck at a high-index saddle:
1. The GAD direction ẋ = -∇E + 2(v₁ᵀ∇E)v₁ becomes small
2. Adaptive dt shrinks to prevent overshooting
3. Eventually dt → 0 and progress halts

v₂ kicking:
1. Perturbs the geometry discontinuously
2. Moves to a new region where GAD direction is non-zero
3. Allows dt to reset and dynamics to continue
4. Eventually, through enough kicks, finds path to index-1

### 4.3 Why Index Doesn't Decrease

The v₂ direction is perpendicular to v₁ by construction. Kicking along v₂:
- Doesn't directly address the 5 negative curvature directions
- Just moves "sideways" in configuration space
- The new point still has ~5 negative eigenvalues (similar PES topology)

But it DOES break out of local traps, allowing continued exploration.

---

## 5. The Correct Theoretical Approach

### 5.1 What We Should Do

To find index-1 from high-index regions, we need **nonlocal convergence**. Options:

1. **iHiSD Crossover** (Most Principled):
   - Start with gradient flow (α ≈ 0) to escape high-index regions
   - Crossover to 1-HiSD (α = 1) as we approach index-1 regions
   - Requires tuning the crossover schedule ε(α)

2. **Gradient Descent Kicks** (Simple Alternative):
   - When stuck, take gradient descent steps (not v₂ kicks)
   - Gradient descent naturally escapes saddles (all saddles are unstable)
   - Then resume GAD

3. **Adaptive k with k < Morse Index**:
   - Use k = 1 always (standard GAD)
   - The "adaptive" part should be in the escape mechanism, not the base dynamics

### 5.2 Why Gradient Descent Escapes Saddles

At any saddle point (index ≥ 1):
- Gradient descent direction: -∇E
- All saddles are unstable for gradient descent (saddles are not minima)
- GD naturally flows away from saddles toward minima

So a "kick" using gradient descent direction would escape the saddle. The question is whether it escapes toward lower or higher index regions.

### 5.3 The Correct Interpretation of "Adaptive"

| Wrong Interpretation | Correct Interpretation |
|---------------------|----------------------|
| k = Morse index (stabilizes current saddle) | k = 1 always (target index-1) |
| Higher index → more reflections | Higher index → need escape mechanism |
| k-HiSD escapes index-k saddles | k-HiSD FINDS index-k saddles |

---

## 6. Summary of What We've Learned

### 6.1 Theoretical Insights

1. **k-HiSD stabilizes index-k saddles** (Theorem 3.2)
   - Using k = Morse index is WRONG for escaping
   - Always use k = 1 to target index-1

2. **Singularities are NOT the main problem**
   - Only 26% of stalls near singularities
   - High-index saddles (Morse index 5-7) are the main issue

3. **v₂ kicking works empirically but not theoretically**
   - Doesn't reduce Morse index
   - Works by brute-force "unsticking"
   - 85% success rate despite theoretical opacity

### 6.2 Practical Implications

1. **Keep v₂ kicking for now** - it works empirically
2. **Don't use adaptive k = Morse index** - it's wrong
3. **For a principled approach**: Implement iHiSD crossover dynamics
4. **Alternative**: Gradient descent kicks (simpler than iHiSD)

### 6.3 Open Questions

1. **Why does v₂ direction specifically help?**
   - Is it better than random perturbation?
   - Is it better than gradient descent perturbation?
   - These are testable hypotheses

2. **Can we predict when kicks will succeed?**
   - Eigenvalue gap? Morse index? Gradient projection?

3. **Is iHiSD crossover faster than v₂ kicking?**
   - Fewer total steps?
   - More consistent convergence?

---

## 7. Corrected Understanding: The Role of Each Method

### 7.1 Methods for FINDING Saddles of Specific Index

| Method | Finds | Stable At |
|--------|-------|-----------|
| Gradient Descent | Minima (index-0) | Minima |
| 1-HiSD (GAD) | Index-1 saddles | Index-1 saddles |
| 2-HiSD | Index-2 saddles | Index-2 saddles |
| k-HiSD | Index-k saddles | Index-k saddles |

### 7.2 Methods for ESCAPING Saddles

| Current Location | Escape Method |
|-----------------|---------------|
| Index-k saddle (k > 1) | Gradient descent (all saddles unstable) |
| Index-k saddle (k > 1) | (k-1)-HiSD or lower |
| Near singularity | Any perturbation to break symmetry |
| Stuck with dt → 0 | Any perturbation to reset dynamics |

### 7.3 The v₂ Kicking Mechanism Revisited

v₂ kicking is best understood as:
1. **NOT** a principled index-reduction scheme
2. **YES** a brute-force escape-and-explore mechanism
3. **Works because**: Eventually finds a path to index-1 through random exploration
4. **Not optimal**: May require many kicks; path is not systematic

---

## 8. Recommended Next Steps

### 8.1 Immediate (Keep What Works)
- Continue using v₂ kicking - 85% success rate is good
- The mechanism isn't theoretically elegant but it works

### 8.2 Experiments to Run
1. **Gradient descent kicks vs v₂ kicks**
   - Replace v₂ with -∇E direction
   - Compare convergence rate and steps

2. **Random direction kicks**
   - Test if direction matters at all
   - May reveal that ANY perturbation works

3. **iHiSD crossover implementation**
   - α dynamics from gradient flow to 1-HiSD
   - Theoretically principled approach

### 8.3 Analysis to Do
1. **Trajectory visualization**
   - Plot Morse index vs step for successful runs
   - Understand the actual convergence path

2. **Kick effectiveness correlation**
   - What predicts successful vs unsuccessful kicks?
   - Eigenvalue gap? Energy change? Gradient alignment?

---

## Appendix: Mathematical Details

### A.1 GAD Direction Derivation

GAD wants to ascend along v₁ and descend elsewhere:
```
ẋ = -∇E + 2(v₁ᵀ∇E)v₁
   = -∇E + 2⟨∇E, v₁⟩v₁
```

Decomposing ∇E = (v₁ᵀ∇E)v₁ + ∇E_⊥:
```
ẋ = -(v₁ᵀ∇E)v₁ - ∇E_⊥ + 2(v₁ᵀ∇E)v₁
   = +(v₁ᵀ∇E)v₁ - ∇E_⊥
```

So GAD ascends along v₁ (+ sign) and descends perpendicular (- sign).

### A.2 k-HiSD Direction Derivation

With R = I - 2∑ᵢ₌₁ᵏ vᵢvᵢᵀ:
```
-R∇E = -(I - 2VₖVₖᵀ)∇E
     = -∇E + 2Vₖ(Vₖᵀ∇E)
```

Decomposing ∇E into components along v₁...vₖ and perpendicular:
```
-R∇E = +∑ᵢ₌₁ᵏ(vᵢᵀ∇E)vᵢ - ∇E_⊥
```

So k-HiSD ascends along v₁...vₖ and descends perpendicular.

### A.3 Why k = Morse Index Stabilizes Current Saddle

At an index-k saddle with eigenvalues λ₁ < ... < λₖ < 0 < λₖ₊₁ < ...:
- k-HiSD ascends along v₁...vₖ (all negative curvature directions)
- k-HiSD descends along vₖ₊₁... (all positive curvature directions)
- At the saddle, ∇E = 0, so the fixed point is stable for k-HiSD
- Small perturbations are corrected by the dynamics

This is exactly opposite of what we want for escaping!
