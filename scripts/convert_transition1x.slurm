#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=convert_t1x
#SBATCH --output=logs/convert_t1x_%j.out
#SBATCH --error=logs/convert_t1x_%j.err

cd ~/project/memo/ts-tools

# Load Compute Canada stack
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load python/3.11

# Activate env
PYTHONBIN=~/project/memo/ts-tools/.venv/bin
source ${PYTHONBIN}/activate

# Run the converter script
${PYTHONBIN}/python ~/project/memo/ts-tools/scripts/convert_transition1x.py --limit 5000