#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /project/aip-aspuru/aburger/gad-ff
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32GB
#SBATCH --job-name=gad
# Jobs must write their output to your scratch or project directory (home is read-only on compute nodes).
#SBATCH --output=/project/aip-aspuru/aburger/gad-ff/outslurm/slurm-%j.txt
#SBATCH --error=/project/aip-aspuru/aburger/gad-ff/outslurm/slurm-%j.txt
# activate venv
source .venv/bin/activate
echo `date`: Job $SLURM_JOB_ID is allocated resources.
uv run "$@"