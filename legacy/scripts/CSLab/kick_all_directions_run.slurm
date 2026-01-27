#!/bin/bash
#SBATCH --job-name=kick_all_dirs
#SBATCH --partition=cpunodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:0
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --output=logs/kick_all_dirs_%j.out
#SBATCH --error=logs/kick_all_dirs_%j.err

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
OUT_DIR="${SCRATCH_DIR}/runs/kick_all_dirs_${SLURM_JOB_ID}"

mkdir -p "$OUT_DIR"

# =============================================================================
# Configuration
# =============================================================================
N_STEPS="${N_STEPS:-15000}"
MAX_SAMPLES="${MAX_SAMPLES:-20}"
START_FROM="${START_FROM:-midpoint_rt_noise1.0A}"
NOISE_SEED="${NOISE_SEED:-42}"
SCINE_FUNCTIONAL="${SCINE_FUNCTIONAL:-DFTB0}"

DT="${DT:-0.0001}"
DT_MIN="${DT_MIN:-1e-6}"
DT_MAX="${DT_MAX:-0.08}"
MAX_ATOM_DISP="${MAX_ATOM_DISP:-0.35}"
ESCAPE_WINDOW="${ESCAPE_WINDOW:-10}"
ESCAPE_DISP_THRESHOLD="${ESCAPE_DISP_THRESHOLD:-1e-4}"
ESCAPE_NEG_VIB_STD="${ESCAPE_NEG_VIB_STD:-0.5}"
ESCAPE_DELTA="${ESCAPE_DELTA:-0.3}"
MAX_ESCAPE_CYCLES="${MAX_ESCAPE_CYCLES:-500}"
MIN_INTERATOMIC_DIST="${MIN_INTERATOMIC_DIST:-0.5}"
TS_EPS="${TS_EPS:-1e-5}"

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

THREADS_PER_WORKER="${THREADS_PER_WORKER:-4}"
N_WORKERS=$((TOTAL_CPUS / THREADS_PER_WORKER))
if [ "$N_WORKERS" -lt 1 ]; then N_WORKERS=1; fi

python "$PROJECT_DIR/src/noisy/v2_tests/runners/run_kick_comparison_parallel.py" \
    --h5-path "$H5_PATH" \
    --out-dir "$OUT_DIR" \
    --scine-functional "$SCINE_FUNCTIONAL" \
    --n-steps "$N_STEPS" \
    --max-samples "$MAX_SAMPLES" \
    --start-from "$START_FROM" \
    --noise-seed "$NOISE_SEED" \
    --n-workers "$N_WORKERS" \
    --threads-per-worker "$THREADS_PER_WORKER" \
    --strategies all \
    --dt "$DT" \
    --dt-min "$DT_MIN" \
    --dt-max "$DT_MAX" \
    --max-atom-disp "$MAX_ATOM_DISP" \
    --escape-window "$ESCAPE_WINDOW" \
    --escape-disp-threshold "$ESCAPE_DISP_THRESHOLD" \
    --escape-neg-vib-std "$ESCAPE_NEG_VIB_STD" \
    --escape-delta "$ESCAPE_DELTA" \
    --max-escape-cycles "$MAX_ESCAPE_CYCLES" \
    --min-interatomic-dist "$MIN_INTERATOMIC_DIST" \
    --ts-eps "$TS_EPS"
