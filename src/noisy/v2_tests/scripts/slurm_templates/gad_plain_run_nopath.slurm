#!/bin/bash
#SBATCH --job-name=gad_plain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=0
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/gad_plain_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/gad_plain_%j.err
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
OUT_DIR="${SCRATCH_DIR}/runs/gad_plain_${SLURM_JOB_ID}"

mkdir -p "$OUT_DIR"

# =============================================================================
# Configuration
# =============================================================================
N_STEPS="${N_STEPS:-10000}"
MAX_SAMPLES="${MAX_SAMPLES:-20}"
START_FROM="${START_FROM:-midpoint_rt_noise2.0A}"
NOISE_SEED="${NOISE_SEED:-42}"
SCINE_FUNCTIONAL="${SCINE_FUNCTIONAL:-DFTB0}"

DT="${DT:-0.00005}"
DT_CONTROL="${DT_CONTROL:-adaptive}"
DT_MIN="${DT_MIN:-1e-6}"
DT_MAX="${DT_MAX:-0.08}"
MAX_ATOM_DISP="${MAX_ATOM_DISP:-0.35}"
MIN_INTERATOMIC_DIST="${MIN_INTERATOMIC_DIST:-0.5}"
TS_EPS="${TS_EPS:-1e-5}"
TR_THRESHOLD="${TR_THRESHOLD:-1e-6}"

# Eckart projection options
# Set to "true" to enable full projection (gradient + guide vector v)
# This prevents TR leakage and improves numerical stability
# Submit with: PROJECT_GRADIENT_AND_V=true sbatch gad_plain_run_nopath.slurm
PROJECT_GRADIENT_AND_V="${PROJECT_GRADIENT_AND_V:-true}"

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

THREADS_PER_WORKER="${THREADS_PER_WORKER:-4}"
N_WORKERS=$((TOTAL_CPUS / THREADS_PER_WORKER))
if [ "$N_WORKERS" -lt 1 ]; then N_WORKERS=1; fi

python "$PROJECT_DIR/src/noisy/v2_tests/runners/run_gad_baselines_parallel_nopath.py" \
    --h5-path "$H5_PATH" \
    --out-dir "$OUT_DIR" \
    --scine-functional "$SCINE_FUNCTIONAL" \
    --n-steps "$N_STEPS" \
    --max-samples "$MAX_SAMPLES" \
    --start-from "$START_FROM" \
    --noise-seed "$NOISE_SEED" \
    --n-workers "$N_WORKERS" \
    --threads-per-worker "$THREADS_PER_WORKER" \
    --baseline plain \
    --dt "$DT" \
    --dt-control "$DT_CONTROL" \
    --dt-min "$DT_MIN" \
    --dt-max "$DT_MAX" \
    --max-atom-disp "$MAX_ATOM_DISP" \
    --min-interatomic-dist "$MIN_INTERATOMIC_DIST" \
    --ts-eps "$TS_EPS" \
    --tr-threshold "$TR_THRESHOLD" \
    $([ "$PROJECT_GRADIENT_AND_V" = "true" ] && echo "--project-gradient-and-v")
