#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/6101772/memoozd/ts-tools
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --job-name=hip_hpo
#SBATCH --output=logs/hip_multi_mode_eckartmw_hpo_%j.out
#SBATCH --error=logs/hip_multi_mode_eckartmw_hpo_%j.err

mkdir -p logs
source .venv/bin/activate

# Ensure CPU-side ops can use the allocated cores.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

H5_PATH="/project/6101772/memoozd/data/transition1x.h5"
CKPT_PATH="/project/6101772/memoozd/models/hip_v2.ckpt"
OUT_DIR="$HOME/scratch/ts-tools-output/hpo"

# Create output directory
mkdir -p "$OUT_DIR"

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo ">>> Running HIP multi-mode Eckart-MW Bayesian HPO..."

START_FROM="midpoint_rt_noise1.0A"
NOISE_SEED=42

# HPO parameters
N_TRIALS=${N_TRIALS:-100}
N_STEPS_PER_SAMPLE=${N_STEPS_PER_SAMPLE:-800}
N_SAMPLES=${N_SAMPLES:-15}
MAX_SAMPLES=${MAX_SAMPLES:-100}  # For initial difficult sample detection
DIFFICULTY_THRESHOLD=${DIFFICULTY_THRESHOLD:-0.5}

# Optuna storage (SQLite database for persistence across runs)
STORAGE="sqlite:///$OUT_DIR/hip_hpo_study.db"
STUDY_NAME="hip-multi-mode-eckartmw-hpo-${SLURM_JOB_ID}"

echo ">>> Bayesian HPO configuration:"
echo "    Trials: $N_TRIALS"
echo "    Steps per sample: $N_STEPS_PER_SAMPLE"
echo "    Samples per trial: $N_SAMPLES"
echo "    Difficulty threshold: $DIFFICULTY_THRESHOLD"
echo "    Study: $STUDY_NAME"
echo "    Storage: $STORAGE"

python -m src.experiments.2025.hip_multi_mode_eckartmw_hpo \
    --h5-path "$H5_PATH" \
    --checkpoint-path "$CKPT_PATH" \
    --out-dir "$OUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --start-from "$START_FROM" \
    --noise-seed "$NOISE_SEED" \
    --n-trials "$N_TRIALS" \
    --n-steps-per-sample "$N_STEPS_PER_SAMPLE" \
    --n-samples "$N_SAMPLES" \
    --difficulty-threshold "$DIFFICULTY_THRESHOLD" \
    --study-name "$STUDY_NAME" \
    --storage "$STORAGE"

echo "Job completed successfully."
echo "Results saved to: $OUT_DIR/hpo_results.json"
echo "Optuna study database: $STORAGE"
