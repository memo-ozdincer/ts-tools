#!/bin/bash
set -euo pipefail

# Submits the two isolated plateau experiments (HIP + SCINE), 30 samples each, 1.0Å noise.
# Usage:
#   bash scripts/killarney/experiments/submit_noisy_1A_30.sh

sbatch scripts/killarney/experiments/scine_gad_plateau_noisy.slurm
sbatch scripts/killarney/experiments/hip_gad_plateau_noisy.slurm
