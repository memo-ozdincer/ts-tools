#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/6101772/memoozd/ts-tools
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --job-name=scine_plateau_noisy_1A_30
#SBATCH --output=logs/scine_plateau_noisy_1A_30_%j.out
#SBATCH --error=logs/scine_plateau_noisy_1A_30_%j.err

# NOTE: No GPU needed for SCINE - it runs on CPU only

mkdir -p logs
source .venv/bin/activate

H5_PATH="/project/6101772/memoozd/data/transition1x.h5"
OUT_DIR="$HOME/scratch/ts-tools-output"

# W&B setup
# - Do NOT hardcode API keys in SLURM scripts; set WANDB_API_KEY in your shell or job env.
# - On clusters without outbound internet, W&B often needs offline mode.
export WANDB_DIR="$OUT_DIR/wandb"
mkdir -p "$WANDB_DIR"
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo ">>> WANDB_API_KEY not set; using WANDB_MODE=offline"
    export WANDB_MODE=offline
fi

export WANDB_RUN_GROUP="Experiment-plateau-noise-hip-scine"

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo ">>> Running SCINE noisy GAD plateau experiment (plateau+floor only)..."

# Noise level for this experiment
START_FROM="midpoint_rt_noise1.0A"
NOISE_SEED=42

# Experiment intent:
# - 30-step Euler-GAD
# - dt increases when neg-eig count plateaus (dt_control=neg_eig_plateau)
# - floor on step size via --min-mean-disp
N_STEPS=1500
MAX_SAMPLES=10

WANDB_NAME="plateau-noise__scine__${START_FROM}__seed${NOISE_SEED}__job${SLURM_JOB_ID}"

echo ">>> SCINE Euler + neg-eig plateau dt control (+ floor), start-from=$START_FROM, noise-seed=$NOISE_SEED"
python -m src.experiments.scine_gad_plateau \
    --method euler \
    --dt-control neg_eig_plateau \
    --min-mean-disp 0.0005 \
    --plateau-patience 10 \
    --plateau-boost 1.5 \
    --plateau-shrink 0.5 \
    --dt-min 1e-6 \
    --dt-max 0.05 \
    --max-atom-disp 0.25 \
    --scine-functional DFTB0 \
    --h5-path "$H5_PATH" \
    --out-dir "$OUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --n-steps "$N_STEPS" \
    --start-from "$START_FROM" \
    --noise-seed "$NOISE_SEED" \
    --stop-at-ts \
    --dt 0.001 \
    --wandb \
    --wandb-project "gad-noise-experiments" \
    --wandb-name "$WANDB_NAME"

echo "✅ Job completed successfully."
