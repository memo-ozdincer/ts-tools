#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --job-name=freq_analysis
#SBATCH --output=logs/freq_analysis_%j.out
#SBATCH --error=logs/freq_analysis_%j.err

cd ~/project/memo/ts-tools

# Load Compute Canada stack
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load python/3.11

# Activate env
PYTHONBIN=~/project/memo/ts-tools/.venv/bin
source ${PYTHONBIN}/activate

# Run analysis
${PYTHONBIN}/python ~/project/memo/ts-tools/scripts/gad_frequency_analysis.py