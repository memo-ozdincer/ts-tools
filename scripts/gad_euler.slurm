#!/bin/bash
#SBATCH --job-name=gad_euler
#SBATCH --gres=gpu:1
#SBATCH --ntasks=4
#SBATCH --mem=16000
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# modules
module purge
module load python/3.10
module load cuda/11.1

# activate venv
source ~/ts-tools/.venv/bin/activate

# go to script folder
cd ~/ts-tools/scripts

# make sure logs dir exists
mkdir -p ~/ts-tools/logs
exec > >(tee ~/ts-tools/logs/gad_euler_$SLURM_JOB_ID.out) 2>&1

# run
python gad_gad_euler_rmsd.py