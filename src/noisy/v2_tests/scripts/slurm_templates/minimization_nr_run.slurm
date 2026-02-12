#!/bin/bash
#SBATCH --job-name=min_nr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=0
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/min_nr_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/min_nr_%j.err
#SBATCH --account=rrg-aspuru

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
TOTAL_CPUS="${SLURM_CPUS_ON_NODE:-32}"
export OMP_NUM_THREADS="$TOTAL_CPUS"
export OPENBLAS_NUM_THREADS="$TOTAL_CPUS"
export MKL_NUM_THREADS="$TOTAL_CPUS"
export NUMEXPR_NUM_THREADS="$TOTAL_CPUS"
export VECLIB_MAXIMUM_THREADS="$TOTAL_CPUS"
export MPLBACKEND=Agg

# Data paths
H5_PATH="/project/rrg-aspuru/memoozd/data/transition1x.h5"
OUT_DIR="${SCRATCH_DIR}/runs/min_nr_${SLURM_JOB_ID}"

mkdir -p "$OUT_DIR"

# =============================================================================
# Configuration
# =============================================================================
N_STEPS="${N_STEPS:-5000}"
MAX_SAMPLES="${MAX_SAMPLES:-20}"
START_FROM="${START_FROM:-midpoint_rt_noise2.0A}"
NOISE_SEED="${NOISE_SEED:-42}"
SCINE_FUNCTIONAL="${SCINE_FUNCTIONAL:-DFTB0}"

MAX_ATOM_DISP="${MAX_ATOM_DISP:-0.3}"
FORCE_CONVERGED="${FORCE_CONVERGED:-1e-4}"
MIN_INTERATOMIC_DIST="${MIN_INTERATOMIC_DIST:-0.5}"
TR_THRESHOLD="${TR_THRESHOLD:-1e-6}"

# Eckart projection of gradient
# Submit with: PROJECT_GRADIENT_AND_V=true sbatch minimization_nr_run.slurm
PROJECT_GRADIENT_AND_V="${PROJECT_GRADIENT_AND_V:-false}"

# Hessian purification (enforce translational sum rules)
PURIFY_HESSIAN="${PURIFY_HESSIAN:-false}"

# =============================================================================
# Experiment submission examples
# =============================================================================
# No projection:
#   sbatch minimization_nr_run.slurm
#
# With Eckart projection of gradient:
#   PROJECT_GRADIENT_AND_V=true sbatch minimization_nr_run.slurm
#
# With Hessian purification:
#   PURIFY_HESSIAN=true sbatch minimization_nr_run.slurm
#
# Full projection + purification:
#   PROJECT_GRADIENT_AND_V=true PURIFY_HESSIAN=true sbatch minimization_nr_run.slurm

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

THREADS_PER_WORKER="${THREADS_PER_WORKER:-4}"
N_WORKERS=$((TOTAL_CPUS / THREADS_PER_WORKER))
if [ "$N_WORKERS" -lt 1 ]; then N_WORKERS=1; fi

python "$PROJECT_DIR/src/noisy/v2_tests/runners/run_minimization_parallel.py" \
    --h5-path "$H5_PATH" \
    --out-dir "$OUT_DIR" \
    --scine-functional "$SCINE_FUNCTIONAL" \
    --n-steps "$N_STEPS" \
    --max-samples "$MAX_SAMPLES" \
    --start-from "$START_FROM" \
    --noise-seed "$NOISE_SEED" \
    --n-workers "$N_WORKERS" \
    --threads-per-worker "$THREADS_PER_WORKER" \
    --method newton_raphson \
    --max-atom-disp "$MAX_ATOM_DISP" \
    --force-converged "$FORCE_CONVERGED" \
    --min-interatomic-dist "$MIN_INTERATOMIC_DIST" \
    --tr-threshold "$TR_THRESHOLD" \
    $([ "$PROJECT_GRADIENT_AND_V" = "true" ] && echo "--project-gradient-and-v") \
    $([ "$PURIFY_HESSIAN" = "true" ] && echo "--purify-hessian")
