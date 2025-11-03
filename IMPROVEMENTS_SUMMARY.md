# Transition State Search Improvements

## Summary of Changes

This document describes the improvements made to address the issue of infinitesimally small eigenvalue products (~1e-16) and premature convergence in transition state search methods.

---

## Problem Diagnosis

**Original Issues:**
1. **GAD RK45**: Eigenvalue products plateaued near zero for extended periods before becoming slightly negative, triggering early stopping at numerical noise levels (~1e-16)
2. **Eigenvalue Descent**: ReLU loss function allowed convergence to infinitesimally small eigenvalues rather than meaningful TS signatures
3. **Both methods**: Lack of verification that λ₀ was becoming MORE negative (not just barely negative)

**Root Cause:** Methods were treating any negative eigenvalue product as success, when in reality, true first-order saddle points have λ₀ ≈ -0.01 to -0.1 eV/Å² (meaningfully negative).

---

## 1. GAD RK45 Search Improvements (`gad_rk45_search.py`)

### A. Multi-Stage Early Stopping

**Old Behavior:**
```python
if eig_prod < -1e-5:  # Stop immediately
    self.ts_found = True
```

**New Behavior:**
- **Stage 1**: Detect TS candidate when `eig_product < min_eig_product_threshold` (default: -1e-4)
- **Stage 2**: Wait `confirmation_steps` (default: 10) to verify eigenvalue product is INCREASING in magnitude
- **Confirmation Criteria**:
  - Eigenvalue product magnitude increases by >20%, OR
  - λ₀ becomes >10% more negative

**New Parameters:**
```bash
--min-eig-product-threshold -1e-4   # Require meaningfully negative product
--confirmation-steps 10              # Steps to wait for confirmation
```

### B. Enhanced Kick Mechanism

**Added Two Kick Triggers:**

1. **Local Minimum Detection** (original): Force magnitude low + both eigenvalues positive
2. **Stagnation Detection** (NEW): Eigenvalue product variance < threshold over window

**New Parameters:**
```bash
--stagnation-check-window 20         # Window size for stagnation check
--stagnation-variance-threshold 1e-6 # Variance threshold
```

### C. Individual Eigenvalue Tracking

**Added to trajectory:**
- `eig0`: Most negative eigenvalue at each step
- `eig1`: Second smallest eigenvalue at each step

**Enhanced plotting:**
- 5 subplots (added individual eigenvalue plot)
- Shows λ₀ and λ₁ evolution separately
- Displays TS threshold (-1e-4) as reference line

### D. Improved Summary Statistics

**Added to output JSON:**
```json
{
  "initial_eig0": -0.0001,
  "final_eig0": -0.0523,
  "initial_eig1": 0.0002,
  "final_eig1": 0.0834,
  "ts_candidate_step": 45,
  "ts_confirmed": true
}
```

---

## 2. Eigenvalue Descent Improvements (`gad_eigenvalue_descent.py`)

### A. New Loss Functions

**Original (Problematic):**
```python
loss = torch.relu(eigvals[0]) + torch.relu(-eigvals[1])
# Problem: Allows infinitesimal eigenvalues
```

**New Options:**

1. **`targeted_magnitude` (RECOMMENDED, now default):**
   ```python
   loss = (eigvals[0] - target_eig0)**2 + (eigvals[1] - target_eig1)**2
   # Targets: λ₀ = -0.05 eV/Å², λ₁ = 0.10 eV/Å²
   ```
   - Pushes eigenvalues to MEANINGFUL magnitudes
   - Prevents convergence to numerical noise

2. **`midpoint_squared`:**
   ```python
   midpoint = (eigvals[0] + eigvals[1]) / 2.0
   loss = midpoint**2
   # Minimizes midpoint between eigenvalues
   ```

3. **`relu` (original, for comparison):**
   ```python
   loss = torch.relu(eigvals[0]) + torch.relu(-eigvals[1])
   ```

### B. Configurable Targets

**New Parameters:**
```bash
--loss-type targeted_magnitude    # Choose loss function
--target-eig0 -0.05               # Target for λ₀ (eV/Å²)
--target-eig1 0.10                # Target for λ₁ (eV/Å²)
```

### C. Enhanced Output

**Added to summary:**
- Individual final eigenvalues (not just product)
- Comprehensive statistics:
  - Average λ₀, λ₁ across all samples
  - Distribution of negative eigenvalues
  - Success rate (samples with exactly 1 negative eigenvalue)

---

## Usage Examples

### GAD RK45 with Multi-Stage Stopping

```bash
# Basic usage with new defaults
python -m src.gad_rk45_search \
  --max-samples 30 \
  --start-from reactant \
  --stop-at-ts \
  --t-end 2.0

# With stricter TS detection
python -m src.gad_rk45_search \
  --max-samples 30 \
  --start-from reactant \
  --stop-at-ts \
  --min-eig-product-threshold -5e-4 \
  --confirmation-steps 15

# With stagnation-aware kicks
python -m src.gad_rk45_search \
  --max-samples 30 \
  --start-from reactant \
  --enable-kick \
  --kick-force-threshold 0.02 \
  --stagnation-variance-threshold 5e-7
```

### Eigenvalue Descent with Targeted Magnitudes

```bash
# Using targeted magnitude loss (RECOMMENDED)
python -m src.gad_eigenvalue_descent \
  --max-samples 30 \
  --start-from reactant \
  --n-steps-opt 200 \
  --lr 0.01 \
  --loss-type targeted_magnitude \
  --target-eig0 -0.05 \
  --target-eig1 0.10

# More aggressive targeting for difficult cases
python -m src.gad_eigenvalue_descent \
  --max-samples 30 \
  --start-from midpoint_rt \
  --n-steps-opt 300 \
  --lr 0.005 \
  --loss-type targeted_magnitude \
  --target-eig0 -0.10 \
  --target-eig1 0.15

# Compare with original ReLU loss
python -m src.gad_eigenvalue_descent \
  --max-samples 30 \
  --start-from reactant \
  --loss-type relu
```

---

## Expected Improvements

### Before (Your Stats):
```json
{
  "final_loss": 1.22e-15,
  "final_eig_product": 1.92e-16,  // Numerical noise!
  "rmsd_to_known_ts": 0.198
}
```

### After (Expected with targeted_magnitude):
```json
{
  "final_loss": 2.5e-4,
  "final_eig_product": -4.2e-3,   // Meaningfully negative!
  "final_eig0": -0.048,           // Close to target -0.05
  "final_eig1": 0.087,            // Close to target 0.10
  "rmsd_to_known_ts": 0.198
}
```

---

## Key Insights

1. **Eigenvalue Magnitude Matters**: A TS must have λ₀ ≈ -0.01 to -0.1 eV/Å², not just λ₀ < 0

2. **Confirmation is Critical**: Don't stop immediately when crossing zero—wait to see if λ₀ becomes MORE negative

3. **Stagnation Detection**: GAD can get trapped in flat regions where eigenvalue product hovers near small positive values

4. **Loss Function Design**: Targeting specific magnitudes is superior to just checking signs

---

## Monitoring Your Results

### Key Metrics to Watch:

1. **Final `eig0` magnitude**: Should be ≈ -0.05 to -0.10 eV/Å² for typical reactions
2. **Final `eig_product` magnitude**: Should be > 1e-3 in absolute value
3. **TS confirmation rate**: How many samples have `ts_confirmed = true`?
4. **RMSD consistency**: Meaningful eigenvalues should correlate with good RMSD

### Diagnostic Plots:

The new 5-panel plots show:
- Panel 3: Individual eigenvalue evolution (see if λ₀ stays meaningfully negative)
- Panel 4: Eigenvalue product with -1e-4 threshold line (see if you cross threshold)

---

## Tuning Recommendations

### If still getting tiny eigenvalue products:

**For GAD RK45:**
- Increase `--min-eig-product-threshold` to -5e-4 or -1e-3
- Increase `--confirmation-steps` to 20

**For Eigenvalue Descent:**
- Increase target magnitudes: `--target-eig0 -0.10 --target-eig1 0.15`
- Reduce learning rate: `--lr 0.005`
- Increase steps: `--n-steps-opt 300`

### If overshooting (λ₀ too negative):
- Relax targets: `--target-eig0 -0.03 --target-eig1 0.08`
- Check if you're starting too close to TS: try `--start-from reactant`

---

## Files Modified

1. **`src/gad_rk45_search.py`**:
   - Multi-stage early stopping logic
   - Enhanced kick mechanism with stagnation detection
   - Individual eigenvalue tracking
   - Improved plotting (5 subplots)
   - New command-line arguments

2. **`src/gad_eigenvalue_descent.py`**:
   - Three loss function options
   - Configurable eigenvalue targets
   - Enhanced summary statistics
   - Individual eigenvalue tracking

---

## Testing Your Installation

Quick validation test:
```bash
# Test GAD RK45 (should NOT stop at numerical noise)
python -m src.gad_rk45_search \
  --max-samples 5 \
  --start-from reactant \
  --stop-at-ts \
  --min-eig-product-threshold -1e-4

# Test eigenvalue descent with targeted loss
python -m src.gad_eigenvalue_descent \
  --max-samples 5 \
  --start-from reactant \
  --loss-type targeted_magnitude \
  --n-steps-opt 100
```

Check output JSON files for:
- `final_eig0` values around -0.05 (not -1e-16!)
- `final_eig_product` values around -0.005 (not -1e-16!)

---

## Questions to Address

1. **Are the default targets appropriate for your reactions?**
   - Default: λ₀ = -0.05, λ₁ = 0.10 eV/Å²
   - You may need to adjust based on typical barrier heights

2. **Should we make the loss function adaptive?**
   - E.g., reduce target magnitude as optimization progresses
   - Would require more complex logic but might be more robust

3. **Do you want eigenvector following mode?**
   - After TS candidate found, switch to following the lowest eigenvector
   - More computationally expensive but potentially more accurate

Let me know your feedback after testing these improvements!
