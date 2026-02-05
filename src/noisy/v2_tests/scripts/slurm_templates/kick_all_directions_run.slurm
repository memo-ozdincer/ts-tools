#!/bin/bash
#SBATCH --job-name=kick_all_dirs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=0
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/memoozd/ts-tools-scratch/logs/kick_all_dirs_%j.out
#SBATCH --error=/scratch/memoozd/ts-tools-scratch/logs/kick_all_dirs_%j.err
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
OUT_DIR="${SCRATCH_DIR}/runs/kick_all_dirs_${SLURM_JOB_ID}"

mkdir -p "$OUT_DIR"

# =============================================================================
# Configuration (aligned with HPO-parallel defaults)
# =============================================================================
N_STEPS="${N_STEPS:-10000}"
MAX_SAMPLES="${MAX_SAMPLES:-20}"
START_FROM="${START_FROM:-midpoint_rt_noise2.0A}"
NOISE_SEED="${NOISE_SEED:-42}"
SCINE_FUNCTIONAL="${SCINE_FUNCTIONAL:-DFTB0}"

DT="${DT:-0.004998616}"
DT_MIN="${DT_MIN:-1e-6}"
DT_MAX="${DT_MAX:-0.0787679}"
MAX_ATOM_DISP="${MAX_ATOM_DISP:-0.3508912104455338}"
ESCAPE_WINDOW="${ESCAPE_WINDOW:-12}"
ESCAPE_DISP_THRESHOLD="${ESCAPE_DISP_THRESHOLD:-1e-2}"
ESCAPE_NEG_VIB_STD="${ESCAPE_NEG_VIB_STD:-0.7}"
ESCAPE_DELTA="${ESCAPE_DELTA:-1.0}"
MAX_ESCAPE_CYCLES="${MAX_ESCAPE_CYCLES:-1000}"
MIN_INTERATOMIC_DIST="${MIN_INTERATOMIC_DIST:-0.5247819497457936}"
TS_EPS="${TS_EPS:-1e-5}"
FORCE_ESCAPE_AFTER="${FORCE_ESCAPE_AFTER:-2000}"
TR_THRESHOLD="${TR_THRESHOLD:-1e-6}"

# Submit with: PROJECT_GRADIENT_AND_V=true sbatch kick_all_directions_run.slurm
PROJECT_GRADIENT_AND_V="${PROJECT_GRADIENT_AND_V:-true}"

# Optional grid search (set GRID=1 to enable)
GRID="${GRID:-1}"
ESCAPE_DELTA_GRID=("1e-5" "1e-4" "1e-3" "1e-2" "1e-1" "1.0")
ESCAPE_DISP_THRESHOLD_GRID=("1e-6" "1e-6" "1e-5" "1e-4" "1e-3" "1e-2")
MAX_ESCAPE_CYCLES_GRID=("5000" "4000" "3000" "2000" "1500" "1000")
PROFILE_NAMES=("high" "mid" "low")
ESCAPE_WINDOW_PROFILE=("8" "12" "20")
ESCAPE_NEG_VIB_STD_PROFILE=("0.9" "0.7" "0.4")

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

THREADS_PER_WORKER="${THREADS_PER_WORKER:-4}"
N_WORKERS=$((TOTAL_CPUS / THREADS_PER_WORKER))
if [ "$N_WORKERS" -lt 1 ]; then N_WORKERS=1; fi

if [[ "$GRID" == "1" ]]; then
    echo "Running kick-all-directions grid search (HPO-parallel style)..."
    for idx in "${!ESCAPE_DELTA_GRID[@]}"; do
        ESCAPE_DELTA="${ESCAPE_DELTA_GRID[$idx]}"
        ESCAPE_DISP_THRESHOLD="${ESCAPE_DISP_THRESHOLD_GRID[$idx]}"
        MAX_ESCAPE_CYCLES="${MAX_ESCAPE_CYCLES_GRID[$idx]}"
        for pidx in "${!PROFILE_NAMES[@]}"; do
            PROFILE_NAME="${PROFILE_NAMES[$pidx]}"
            ESCAPE_WINDOW="${ESCAPE_WINDOW_PROFILE[$pidx]}"
            ESCAPE_NEG_VIB_STD="${ESCAPE_NEG_VIB_STD_PROFILE[$pidx]}"
            RUN_TAG="ed${ESCAPE_DELTA}_edt${ESCAPE_DISP_THRESHOLD}_ew${ESCAPE_WINDOW}_env${ESCAPE_NEG_VIB_STD}_mec${MAX_ESCAPE_CYCLES}_${PROFILE_NAME}"
            RUN_OUT_DIR="${OUT_DIR}/${RUN_TAG}"
            mkdir -p "$RUN_OUT_DIR"
            python "$PROJECT_DIR/src/noisy/v2_tests/runners/run_kick_comparison_parallel.py" \
                --h5-path "$H5_PATH" \
                --out-dir "$RUN_OUT_DIR" \
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
                --ts-eps "$TS_EPS" \
                --tr-threshold "$TR_THRESHOLD" \
                --force-escape-after "$FORCE_ESCAPE_AFTER" \
                $([ "$PROJECT_GRADIENT_AND_V" = "true" ] && echo "--project-gradient-and-v")
        done
    done
else
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
        --ts-eps "$TS_EPS" \
        --tr-threshold "$TR_THRESHOLD" \
        --force-escape-after "$FORCE_ESCAPE_AFTER" \
        $([ "$PROJECT_GRADIENT_AND_V" = "true" ] && echo "--project-gradient-and-v")
fi
