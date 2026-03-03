#!/bin/bash
#SBATCH --job-name=min_nr_v8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --gpus-per-node=0
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/min_nr_v8_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/min_nr_v8_%j.err
#SBATCH --account=rrg-aspuru

# =============================================================================
# Newton-Raphson Minimization — v8 iHiSD-Inspired Crossover
# =============================================================================
#
# Context — v7 Results Analysis:
#
#   | Noise | Trust Region | Line Search | LS + mw=200 |
#   |-------|-------------|-------------|-------------|
#   | 0.5 A | **77%**     | 72%         | 70%         |
#   | 1.0 A | **59%**     | 51%         | —           |
#   | 1.5 A | **41%**     | 31%         | —           |
#
#   Line search made things worse — trust region limits damage by capping
#   to 0.01 A when the step is bad; line search starts alpha=1.0 in a
#   *wrong direction*, and even alpha=0.001 in a wrong direction doesn't
#   help.
#
#   Two distinct failure regimes from trajectory images:
#   1. "Almost converged" (0.5-1.0 A): Forces decrease to ~0.03 eV/A but
#      ghost n_neg persist. Trust radius collapsed, taking 0.01 A steps
#      for thousands of iterations.
#   2. "Genuinely stuck" (1.5-2.0 A): Forces flat at 10-1000 eV/A for
#      10,000 steps. n_neg = 6-15 throughout. Condition numbers 10^8 to
#      10^26 (amplifying numerical noise by 10^24). Displacement maxed at
#      1.3 A cap. Zero progress — quadratic model completely invalid,
#      Newton DIRECTION is wrong.
#
#   Root cause: At high noise, Newton's step direction is unreliable
#   because the Hessian is not a meaningful curvature estimate. No step-
#   size control (trust region or line search) can fix a wrong direction.
#
# v8 Approach — iHiSD-Inspired Crossover:
#
#   Start with gradient descent (globally convergent) and smoothly
#   transition to Newton (locally fast) via adaptive damping:
#
#     mu = mu_max * (1 - alpha)
#     effective_lam_i = shifted_lam_i + mu
#     step_i = -g_i / effective_lam_i
#
#   alpha = morse_factor * force_factor, where:
#     morse_factor = max(0, 1 - n_neg / n_neg_ref)
#     force_factor = 1 / (1 + force_norm / force_ref)
#
#   At alpha=0 (far from min): step ≈ -g/mu_max (gradient descent)
#   At alpha=1 (at min):       step = standard shifted Newton
#
#   Always combined with Armijo line search (stateless, no trust region).
#
# Grid: 6 configs x 4 noise levels = 24 combos x 300 samples each
#
# | # | mu_max | n_neg_ref | force_ref | Description             |
# |---|--------|-----------|-----------|-------------------------|
# | 1 | 0      | —         | —         | v7 baseline: SN + LS    |
# | 2 | 0.1    | 3         | 0.1       | Light crossover         |
# | 3 | 0.5    | 3         | 0.1       | Moderate crossover      |
# | 4 | 1.0    | 3         | 0.1       | Strong crossover        |
# | 5 | 2.0    | 3         | 0.1       | Very strong crossover   |
# | 6 | 0.5    | 5         | 0.1       | Wider Morse transition  |
#
# Noise levels: 0.5, 1.0, 1.5, 2.0 A
# Total: 24 combos x 300 samples = 7200 trajectories
#
# =============================================================================

set -e

mkdir -p /scratch/memoozd/ts-tools-scratch/logs

PROJECT_DIR="/project/rrg-aspuru/memoozd/ts-tools"
SCRATCH_DIR="/scratch/memoozd/ts-tools-scratch"

cd "$SCRATCH_DIR"

module purge
module load StdEnv/2023
module load python/3.11.5

source "$PROJECT_DIR/.venv/bin/activate"

# Environment setup
TMP_BASE="${SLURM_TMPDIR:-${SCRATCH_DIR}/tmp}"
export TMPDIR="${TMP_BASE}/${SLURM_JOB_ID}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CACHE_HOME="${SCRATCH_DIR}/.cache"
export MPLCONFIGDIR="${SCRATCH_DIR}/.config/matplotlib"
export FONTCONFIG_PATH=/etc/fonts
export FONTCONFIG_FILE=/etc/fonts/fonts.conf

mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

# Thread configuration
TOTAL_CPUS="${SLURM_CPUS_ON_NODE:-192}"
export OMP_NUM_THREADS="$TOTAL_CPUS"
export OPENBLAS_NUM_THREADS="$TOTAL_CPUS"
export MKL_NUM_THREADS="$TOTAL_CPUS"
export NUMEXPR_NUM_THREADS="$TOTAL_CPUS"
export VECLIB_MAXIMUM_THREADS="$TOTAL_CPUS"
export MPLBACKEND=Agg

# Data paths
H5_PATH="/project/rrg-aspuru/memoozd/data/transition1x.h5"
OUT_DIR="${SCRATCH_DIR}/runs/min_nr_v8_${SLURM_JOB_ID}"

mkdir -p "$OUT_DIR"

# =============================================================================
# Fixed configuration
# =============================================================================
N_STEPS=10000
MAX_SAMPLES=300
SCINE_FUNCTIONAL="DFTB0"
FORCE_CONVERGED="1e-4"
MIN_INTERATOMIC_DIST="0.5"
LOG_SPECTRUM_K=10
SHIFT_EPSILON="1e-3"
MAD="1.3"

# Single seed from SLURM_JOB_ID
NOISE_SEED="${SLURM_JOB_ID:-$(date +%s)}"

# Parallel worker config
THREADS_PER_WORKER="${THREADS_PER_WORKER:-4}"
N_WORKERS=$((TOTAL_CPUS / THREADS_PER_WORKER))
if [ "$N_WORKERS" -lt 1 ]; then N_WORKERS=1; fi

echo "=============================================="
echo "Newton-Raphson Minimization — v8 Crossover"
echo "=============================================="
echo "Date:         $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "CPUs:         $TOTAL_CPUS"
echo "Workers:      $N_WORKERS  (${THREADS_PER_WORKER} threads each)"
echo "=============================================="
echo "Fixed config:"
echo "  SHIFT_EPSILON:      $SHIFT_EPSILON"
echo "  MAD:                $MAD"
echo "  NOISE_SEED:         $NOISE_SEED"
echo "  N_STEPS:            $N_STEPS"
echo "  MAX_SAMPLES:        $MAX_SAMPLES"
echo "  FORCE_CONVERGED:    $FORCE_CONVERGED"
echo "  LOG_SPECTRUM_K:     $LOG_SPECTRUM_K"
echo "Configs:  24 (6 crossover configs x 4 noise levels)"
echo "=============================================="

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

COMBO=0
FAILED=0

# =============================================================================
# Helper: run one combination
# Usage: run_v8 <noise_level> <tag> [extra_flags...]
# =============================================================================
run_v8() {
    local noise_level="$1"
    local tag="$2"
    shift 2

    local start_from="midpoint_rt_noise${noise_level}A"
    local run_out="${OUT_DIR}/${tag}"
    mkdir -p "$run_out"

    COMBO=$((COMBO + 1))
    echo "[combo ${COMBO}]  ${tag}  (noise=${noise_level}A, seed=${NOISE_SEED})"

    if python "$PROJECT_DIR/src/noisy/v2_tests/runners/run_minimization_parallel.py" \
            --h5-path "$H5_PATH" \
            --out-dir "$run_out" \
            --scine-functional "$SCINE_FUNCTIONAL" \
            --n-steps "$N_STEPS" \
            --max-samples "$MAX_SAMPLES" \
            --start-from "$start_from" \
            --noise-seed "$NOISE_SEED" \
            --n-workers "$N_WORKERS" \
            --threads-per-worker "$THREADS_PER_WORKER" \
            --method newton_raphson \
            --max-atom-disp "$MAD" \
            --force-converged "$FORCE_CONVERGED" \
            --min-interatomic-dist "$MIN_INTERATOMIC_DIST" \
            --project-gradient-and-v \
            --log-spectrum-k "$LOG_SPECTRUM_K" \
            --shift-epsilon "$SHIFT_EPSILON" \
            --step-control "line_search" \
            "$@"; then
        echo "  -> OK"
    else
        echo "  -> FAILED (exit $?)"
        FAILED=$((FAILED + 1))
    fi
    echo ""
}

# =============================================================================
# Grid: 6 configs x 4 noise levels = 24 combos
# =============================================================================

for NOISE in "0.5" "1.0" "1.5" "2.0"; do

    # Config 1: v7 baseline — pure shifted Newton + line search (no crossover)
    run_v8 "$NOISE" "n${NOISE}_mad${MAD}_se${SHIFT_EPSILON}_scls_cm0_pgtrue_phfalse"

    # Config 2: Light crossover (mu_max=0.1)
    run_v8 "$NOISE" "n${NOISE}_mad${MAD}_se${SHIFT_EPSILON}_scls_cm0.1_pgtrue_phfalse" \
        --crossover-mu-max 0.1

    # Config 3: Moderate crossover (mu_max=0.5)
    run_v8 "$NOISE" "n${NOISE}_mad${MAD}_se${SHIFT_EPSILON}_scls_cm0.5_pgtrue_phfalse" \
        --crossover-mu-max 0.5

    # Config 4: Strong crossover (mu_max=1.0)
    run_v8 "$NOISE" "n${NOISE}_mad${MAD}_se${SHIFT_EPSILON}_scls_cm1.0_pgtrue_phfalse" \
        --crossover-mu-max 1.0

    # Config 5: Very strong crossover (mu_max=2.0)
    run_v8 "$NOISE" "n${NOISE}_mad${MAD}_se${SHIFT_EPSILON}_scls_cm2.0_pgtrue_phfalse" \
        --crossover-mu-max 2.0

    # Config 6: Wider Morse transition (mu_max=0.5, n_neg_ref=5)
    run_v8 "$NOISE" "n${NOISE}_mad${MAD}_se${SHIFT_EPSILON}_scls_cm0.5_cnr5_pgtrue_phfalse" \
        --crossover-mu-max 0.5 \
        --crossover-n-neg-ref 5

done

# =============================================================================
# Summary
# =============================================================================
echo "=============================================="
echo "v8 grid completed at $(date)"
echo "  Total combinations: $COMBO"
echo "  Successful:         $((COMBO - FAILED))"
echo "  Failed:             $FAILED"
echo "  Noise seed:         $NOISE_SEED"
echo "  Results:            $OUT_DIR"
echo "=============================================="

# =============================================================================
# Analysis
# =============================================================================
ANALYSIS_DIR="${OUT_DIR}/analysis"
mkdir -p "$ANALYSIS_DIR"

echo "Running grid analysis ..."
python "$PROJECT_DIR/src/noisy/v2_tests/scripts/analyze_minimization_nr_grid.py" \
    --grid-dir "$OUT_DIR" \
    --result-glob "*/minimization_newton_raphson_*_results.json" \
    --output-dir "$ANALYSIS_DIR" \
    --top-k 24 > "${ANALYSIS_DIR}/report.txt" 2>&1 || true

echo "Running failure autopsy ..."
AUTOPSY_DIR="${OUT_DIR}/autopsy"
mkdir -p "$AUTOPSY_DIR"
python "$PROJECT_DIR/src/noisy/v2_tests/scripts/analyze_nr_failure_autopsy.py" \
    --grid-dir "$OUT_DIR" \
    --output-dir "$AUTOPSY_DIR" > "${AUTOPSY_DIR}/autopsy_report.txt" 2>&1 || true

echo "=============================================="
echo "All done.  Results in: $OUT_DIR"
echo "  Grid report:    ${ANALYSIS_DIR}/report.txt"
echo "  Autopsy report: ${AUTOPSY_DIR}/autopsy_report.txt"
echo "=============================================="
