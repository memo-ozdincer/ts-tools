#!/bin/bash
#SBATCH --job-name=grad_descent
#SBATCH --partition=cpunodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:0
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --output=logs/grad_descent_%j.out
#SBATCH --error=logs/grad_descent_%j.err

# --- Configuration ---
set -e
PROJECT_DIR="/h/327/memo/ts-tools"
VENV_DIR="${PROJECT_DIR}/.venv"
PYTHON_VERSION="3.11"
SCRATCH_DIR="/scratch/${USER}/ts-tools-scratch"

# --- Script Start ---
echo ">>> Starting job on node $(hostname)"
cd "${PROJECT_DIR}"
mkdir -p logs
mkdir -p "${SCRATCH_DIR}" "${SCRATCH_DIR}/logs"

echo ">>> Loading Python ${PYTHON_VERSION} module..."
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module purge
module load python/${PYTHON_VERSION}

echo ">>> Activating virtual environment..."
source "${VENV_DIR}/bin/activate"

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
TOTAL_CPUS="${SLURM_CPUS_PER_TASK:-32}"
export OMP_NUM_THREADS="$TOTAL_CPUS"
export OPENBLAS_NUM_THREADS="$TOTAL_CPUS"
export MKL_NUM_THREADS="$TOTAL_CPUS"
export NUMEXPR_NUM_THREADS="$TOTAL_CPUS"
export VECLIB_MAXIMUM_THREADS="$TOTAL_CPUS"
export MPLBACKEND=Agg

# Data paths
H5_PATH="${H5_PATH:-/project/memo/data/transition1x.h5}"
OUT_DIR="${SCRATCH_DIR}/runs/grad_descent_${SLURM_JOB_ID}"

mkdir -p "$OUT_DIR"

# =============================================================================
# Configuration
# =============================================================================
N_STEPS="${N_STEPS:-5000}"
MAX_SAMPLES="${MAX_SAMPLES:-20}"
START_FROM="${START_FROM:-midpoint_rt_noise1.0A}"
NOISE_SEED="${NOISE_SEED:-42}"
SCINE_FUNCTIONAL="${SCINE_FUNCTIONAL:-DFTB0}"

STEP_SIZE="${STEP_SIZE:-0.01}"
STEP_SIZE_MIN="${STEP_SIZE_MIN:-1e-6}"
STEP_SIZE_MAX="${STEP_SIZE_MAX:-0.1}"
MAX_ATOM_DISP="${MAX_ATOM_DISP:-0.3}"
FORCE_CONVERGED="${FORCE_CONVERGED:-1e-4}"
MIN_INTERATOMIC_DIST="${MIN_INTERATOMIC_DIST:-0.5}"
ADAPTIVE_STEP="${ADAPTIVE_STEP:-1}"
ARMIJO_C="${ARMIJO_C:-1e-4}"
BACKTRACK_FACTOR="${BACKTRACK_FACTOR:-0.5}"
MAX_BACKTRACK="${MAX_BACKTRACK:-10}"
TR_THRESHOLD="${TR_THRESHOLD:-1e-6}"

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

THREADS_PER_WORKER="${THREADS_PER_WORKER:-4}"
N_WORKERS=$((TOTAL_CPUS / THREADS_PER_WORKER))
if [ "$N_WORKERS" -lt 1 ]; then N_WORKERS=1; fi

CMD=(
    python "$PROJECT_DIR/src/noisy/v2_tests/runners/run_gradient_descent_parallel.py"
    --h5-path "$H5_PATH"
    --out-dir "$OUT_DIR"
    --scine-functional "$SCINE_FUNCTIONAL"
    --n-steps "$N_STEPS"
    --max-samples "$MAX_SAMPLES"
    --start-from "$START_FROM"
    --noise-seed "$NOISE_SEED"
    --n-workers "$N_WORKERS"
    --threads-per-worker "$THREADS_PER_WORKER"
    --step-size "$STEP_SIZE"
    --step-size-min "$STEP_SIZE_MIN"
    --step-size-max "$STEP_SIZE_MAX"
    --max-atom-disp "$MAX_ATOM_DISP"
    --force-converged "$FORCE_CONVERGED"
    --min-interatomic-dist "$MIN_INTERATOMIC_DIST"
    --armijo-c "$ARMIJO_C"
    --backtrack-factor "$BACKTRACK_FACTOR"
    --max-backtrack "$MAX_BACKTRACK"
    --tr-threshold "$TR_THRESHOLD"
)

if [ "$ADAPTIVE_STEP" -eq 1 ]; then
    CMD+=(--adaptive-step)
else
    CMD+=(--no-adaptive-step)
fi

"${CMD[@]}"
