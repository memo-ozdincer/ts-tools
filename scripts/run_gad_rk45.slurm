#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=gad_rk45      # Updated job name
#SBATCH --output=logs/gad_rk45_%j.out  # Updated log file
#SBATCH --error=logs/gad_rk45_%j.err   # Updated log file

# --- Configuration ---
set -e
PROJECT_DIR="/project/memo/code/ts-tools"
VENV_DIR="${PROJECT_DIR}/.venv"
PYTHON_VERSION="3.11"

# --- Script Start ---
echo ">>> Starting job on node $(hostname)"
echo ">>> Project Directory: ${PROJECT_DIR}"

cd ${PROJECT_DIR}

echo ">>> Loading Python ${PYTHON_VERSION} module..."
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module purge
module load python/${PYTHON_VERSION}

echo ">>> Activating virtual environment..."
source ${VENV_DIR}/bin/activate

echo ">>> Running GAD-RK45 Search script..."

# --- EXPERIMENTS ---
#
# This script runs the new GAD-RK45 adaptive solver.
# Note that `--dt` is NOT used, as the step size is now adaptive.
# Instead, we set a total integration time with `--t-end`.
#
# New arguments:
# --force-kick-thresh : RMS force below which we switch to eigenvector kicks.
# --kick-scale        : How "hard" the eigenvector kick is.

# --- Experiment 1: Start from Reactant, search for TS (RECOMMENDED) ---
# This is a powerful test to see if the method can climb from a minimum to a saddle point.
python -m src.gad_rk45_search \
    --max-samples 300 \
    --start-from reactant \
    --stop-at-ts \
    --t-end 2.0 \
    --force-kick-thresh 0.05 \
    --kick-scale 0.2


# --- Experiment 2: Start from Midpoint, search for TS ---
# To run this, comment out Experiment 1 and uncomment the block below.
# python -m src.gad_rk45_search \
#     --max-samples 300 \
#     --start-from midpoint_rt \
#     --stop-at-ts \
#     --t-end 2.0 \
#     --force-kick-thresh 0.05 \
#     --kick-scale 0.2


# --- Experiment 3: Start from TS guess, refine for a fixed time (NO early stopping) ---
# This tests how a TS guess behaves under the dynamics.
# python -m src.gad_rk45_search \
#     --max-samples 300 \
#     --start-from ts \
#     --t-end 1.0 \
#     --force-kick-thresh 0.05 \
#     --kick-scale 0.2


echo "✅ Job completed successfully."